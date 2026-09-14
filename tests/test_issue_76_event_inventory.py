import unittest

from scripts.prepare_issue_76_event_development import FAMILIES, ROLES, assignments, materialize_assignment, select_sources


class EventInventoryTests(unittest.TestCase):
    def sources(self):
        return [{"identity": f"source-{family}-{i:03d}", "ordinal": family * 70 + i + 1,
                 "generator_family": name, "exposure_role": "training" if i < 50 else "calibration" if i < 60 else "model_selection",
                 "actions": [{"drag_x": -80, "drag_y": 10, "release_time_ms": 600, "tap_time_ms": 0}],
                 "scenario": {"fixture": i}, "base_cluster": f"base-{family}-{i}", "engine_seed": i,
                 "xml": "retained-source"}
                for family, name in enumerate(FAMILIES) for i in range(70)]

    def test_exact_role_preserving_outcome_independent_selection(self):
        sources = self.sources()
        selected = select_sources(sources)
        self.assertEqual(len(selected), 200)
        for family in FAMILIES:
            for role, count in zip(ROLES, (20, 10, 10)):
                self.assertEqual(sum(s["generator_family"] == family and s["exposure_role"] == role for s in selected), count)
        for source in sources:
            source["gameplay_success"] = True
        self.assertEqual([s["identity"] for s in select_sources(sources)], [s["identity"] for s in selected])
        self.assertEqual([s["scenario"]["fixture"] for s in selected[:20]], [i * 49 // 19 for i in range(20)])

    def test_assignments_bind_original_sources_without_role_or_scenario_changes(self):
        sources = select_sources(self.sources())
        rows = assignments(sources)
        self.assertEqual(len(rows), 2600)
        self.assertEqual(len({row["identity"] for row in rows}), 2600)
        member = materialize_assignment(sources[0], rows[5])
        for key in ("scenario", "xml", "base_cluster", "engine_seed", "exposure_role"):
            self.assertEqual(member[key], sources[0][key])
        self.assertEqual(member["actions"], [rows[5]["action"]])
        self.assertEqual(member["maximum_shots"], 1)
        self.assertEqual(sum(row["original_action_reference"] for row in rows), 200)

    def test_missing_role_member_is_not_replaced(self):
        with self.assertRaises(ValueError):
            select_sources(self.sources()[:-1])
