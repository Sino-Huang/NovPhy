from io import StringIO
import unittest
from unittest.mock import patch

from scripts.validate_issue_76_angular_collection import failed_outcome, group_report, outcome, viability


class AngularCollectionTests(unittest.TestCase):
    def test_censoring_overrides_clear_and_applies_failure_penalty(self):
        result = {"segments": [{"native_root": "/fixture", "summary": {"censored": True}}]}
        manifest = {"chunks": [{"path": "last.json.gz"}], "last_fixed_step": 10}
        with patch("scripts.validate_issue_76_angular_collection.terminal_evidence",
                   return_value={"reason": "level_clear"}), patch(
                       "scripts.validate_issue_76_angular_collection.angular.supervisor.files.read",
                       return_value=manifest), patch(
                           "scripts.validate_issue_76_angular_collection.gzip.open",
                           return_value=StringIO('{"fixed_step_samples": [{"fixed_step": 10, "entities": []}]}')):
            value = outcome(result)
        self.assertEqual(value["ranking_cost"], 1e9)
        self.assertFalse(value["native_level_clear"])
        self.assertIsNone(value["settled_count_cost"])

    def test_failed_assignment_retained_and_whole_lineage_excluded(self):
        members = [{"identity": str(i), "source_member_identity": "lineage",
                    "original_action_reference": i == 0} for i in range(3)]
        rows = [{**member, "candidate_ordinal": i, **failed_outcome("fixture")}
                for i, member in enumerate(members)]
        with patch("scripts.validate_issue_76_angular_collection.angular.validation.comparison.compare",
                   return_value={"comparable": True, "failures": []}) as compare:
            groups = group_report(members, {"0": {}, "1": {}}, rows, {})
        self.assertEqual(compare.call_count, 2)
        self.assertEqual(len(groups[0]["rows"]), 3)
        self.assertFalse(groups[0]["paired_ranking_admissible"])
        self.assertEqual(groups[0]["rows"][2]["ranking_cost"], 1e9)
        self.assertEqual(groups[0]["candidate_count"], 2)

    def test_reference_clear_does_not_satisfy_grid_criterion(self):
        members = [{"identity": str(i), "source_member_identity": "lineage",
                    "original_action_reference": i == 0} for i in range(2)]
        rows = [{**member, "candidate_ordinal": i, **failed_outcome("fixture"),
                 "native_level_clear": i == 0} for i, member in enumerate(members)]
        groups = group_report(members, {}, rows, {})
        self.assertEqual(groups[0]["native_level_clears"], 0)

    def test_viability_requires_both_thresholds_and_never_authorizes_advancement(self):
        groups = [{"paired_ranking_admissible": i < 4, "native_level_clears": int(i < 3)}
                  for i in range(5)]
        result = viability(groups)
        self.assertTrue(result["action_design_viable"])
        self.assertFalse(result["advancement_authorized"])
        self.assertFalse(result["training_authorized"])
        groups[0]["paired_ranking_admissible"] = False
        self.assertFalse(viability(groups)["action_design_viable"])
        groups[0]["paired_ranking_admissible"] = True
        groups[2]["native_level_clears"] = 0
        self.assertFalse(viability(groups)["action_design_viable"])
