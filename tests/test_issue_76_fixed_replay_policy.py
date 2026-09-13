import unittest

from scripts.issue_76_fixed_replay_policy import FixedReplayPolicy


class FixedReplayPolicyTests(unittest.TestCase):
    def setUp(self):
        self.action = dict(drag_x=-10, drag_y=-80, release_time_ms=600, tap_time_ms=0)

    def test_genuine_decision_precedes_executed_segment_and_images_do_not_select(self):
        policy = FixedReplayPolicy(self.action)
        policy.observe(b"pre-decision agent image", 6.)
        decision = policy.choose()
        self.assertEqual(decision["action"], self.action)
        self.assertFalse(decision["observations_influence_action"])
        policy.observe(b"actual prelaunch image", 6.02)
        policy.executed(decision["action"], 6.02)
        policy.observe(b"later observed image", 6.04)
        evidence = policy.evidence()
        self.assertEqual(evidence["observations"], 3)
        self.assertEqual(evidence["executed_actions"], 1)
        self.assertEqual(evidence["decision_time"], 6.)
        self.assertEqual(evidence["executed_action_time"], 6.02)
        policy.reset()
        policy.observe(b"different image", 1.)
        self.assertEqual(policy.choose()["action"], self.action)

    def test_external_mutation_cannot_change_assignment(self):
        policy = FixedReplayPolicy(self.action)
        self.action["drag_x"] = -160
        policy.observe(b"image", 1.)
        chosen = policy.choose()
        self.assertEqual(chosen["action"]["drag_x"], -10)
        chosen["action"]["drag_x"] = -60
        self.assertEqual(policy.action["drag_x"], -10)

    def test_missing_observation_repeated_decision_and_mismatched_action_fail(self):
        policy = FixedReplayPolicy(self.action)
        with self.assertRaises(ValueError):
            policy.choose()
        policy.observe(b"image", 1.)
        policy.choose()
        with self.assertRaises(ValueError):
            policy.choose()
        with self.assertRaises(ValueError):
            policy.executed(dict(self.action, drag_x=-60), 1.)
        with self.assertRaises(ValueError):
            policy.executed(self.action, 1.1)
        policy.executed(self.action, 1.)
        with self.assertRaises(ValueError):
            policy.executed(self.action, 1.)
        with self.assertRaises(ValueError):
            policy.observe(b"earlier image", .9)


if __name__ == "__main__":
    unittest.main()
