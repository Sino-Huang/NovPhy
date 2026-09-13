from copy import deepcopy
import unittest

from scripts.prepare_issue_76_action_replay import FAMILIES, replay_members


class ActionReplayInventoryTests(unittest.TestCase):
    def test_source_lineages_are_preserved_and_reference_is_not_a_ranked_candidate(self):
        sources = [dict(identity=f"source-{i}", ordinal=i + 1, generator_family=family,
                        exposure_role="training", novelty_level=0, base_cluster=f"lineage-{i}",
                        generation_seed=100 + i, engine_seed=200 + i, xml="unchanged XML",
                        scenario={"identity": f"scenario-{i}"}, generated_slots=["pig:0000"],
                        actions=[dict(drag_x=-80, drag_y=10, release_time_ms=600, tap_time_ms=0)])
                   for i, family in enumerate(FAMILIES)]
        original = deepcopy(sources)
        members = replay_members(sources)
        self.assertEqual(sources, original)
        self.assertEqual(len(members), 65)
        self.assertEqual(len({member["identity"] for member in members}), 65)
        self.assertEqual(len({member["base_cluster"] for member in members}), 5)
        for index, source in enumerate(sources):
            group = members[index * 13:(index + 1) * 13]
            self.assertEqual([member["candidate_ordinal"] for member in group], list(range(13)))
            self.assertEqual(sum(member["original_action_reference"] for member in group), 1)
            for member in group:
                for key in ("xml", "scenario", "base_cluster", "generation_seed", "engine_seed", "exposure_role"):
                    self.assertEqual(member[key], source[key])
                self.assertEqual(member["maximum_shots"], 1)
            self.assertEqual(group[0]["actions"], source["actions"])
            self.assertEqual(group[1]["actions"][0]["drag_y"], -80)

    def test_first_training_assignment_is_not_replaced_by_a_later_usable_one(self):
        sources = []
        for index, family in enumerate(FAMILIES):
            for offset in (0, 1):
                sources.append(dict(identity=f"{family}-{offset}", ordinal=index * 2 + offset + 1,
                    generator_family=family, exposure_role="training", novelty_level=0, usable=bool(offset),
                    actions=[dict(drag_x=-80, drag_y=10, release_time_ms=600, tap_time_ms=0)]))
        members = replay_members(sources)
        self.assertTrue(all(member["source_member_identity"].endswith("-0") for member in members))


if __name__ == "__main__":
    unittest.main()
