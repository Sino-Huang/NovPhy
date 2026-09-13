import unittest
from scripts.validate_issue_76_action_replay import observation_source_matches


class ObservationLineageBindingTests(unittest.TestCase):
    def test_derived_observation_lineage_is_not_the_source_scenario_lineage(self):
        value = {"exposure_role": "training", "scenario_lineage_identity": "derived-observation-lineage",
                 "source_bindings": {"source_scenario_lineage_identity": "source-lineage"}}
        self.assertTrue(observation_source_matches(value, "source-lineage"))
        self.assertFalse(observation_source_matches(value, "another-source"))
        value["exposure_role"] = "model_selection"
        self.assertFalse(observation_source_matches(value, "source-lineage"))


if __name__ == "__main__":
    unittest.main()
