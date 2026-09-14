import unittest
from scripts.prepare_issue_76_angular_replay import angular_actions, angular_members


class AngularInventoryTests(unittest.TestCase):
    def test_frozen_grid_and_source_preservation(self):
        expected = [(-80,7),(-78,17),(-76,26),(-72,35),(-67,44),(-61,51),
                    (-55,59),(-47,65),(-39,70),(-30,74),(-21,77),(-11,79)]
        self.assertEqual([(a["drag_x"], a["drag_y"]) for a in angular_actions()], expected)
        original = [{"identity": f"issue-76-action-replay-001-a{i:02d}",
                     "original_action_reference": i == 0, "candidate_ordinal": i,
                     "source_member_identity": "source", "xml": "unchanged", "engine_seed": 1,
                     "actions": [{"drag_x": -80, "drag_y": 10}]} for i in range(13)]
        members = angular_members(original)
        self.assertEqual(members[0]["actions"], original[0]["actions"])
        for member, source in zip(members, original):
            self.assertNotEqual(member["identity"], source["identity"])
            for key in ("source_member_identity", "xml", "engine_seed"):
                self.assertEqual(member[key], source[key])
        self.assertEqual(original[1]["actions"][0]["drag_x"], -80)
