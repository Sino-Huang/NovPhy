from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_issue_44_probe_plan import build_issue_44_probe_plan
from scripts.collection_plan import load_collection_plan


class Issue44ProbePlanTests(unittest.TestCase):
    def test_builds_the_five_frozen_nonfinal_probe_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "probe-plan.json"
            result = build_issue_44_probe_plan(output)
            loaded = load_collection_plan(output)
            cases = {
                intervention.id
                for scenario in loaded.plan.scenarios
                for intervention in scenario.interventions
            }
            self.assertEqual(cases, {
                "no-contact", "collision", "support", "support-change", "stable-terminal",
            })
            self.assertEqual(
                {scenario.exposure_role for scenario in loaded.plan.scenarios},
                {"training", "calibration"},
            )
            scenarios = {scenario.exposure_role: scenario for scenario in loaded.plan.scenarios}
            self.assertEqual(
                scenarios["training"].scenario_manifest_projection["scenario_lineage_identity"],
                "scenario-lineage-v1:sha256:ef31cc858b84d09d853dbf5957e163e34f7154509db9bfbfbf84a1edd3c4ccba",
            )
            by_case = {
                intervention.id: intervention
                for scenario in loaded.plan.scenarios
                for intervention in scenario.interventions
            }
            self.assertEqual(by_case["collision"].engine_relative_action["release_offset"], (-80, 8))
            self.assertIn("nearest-active-target-center:block:0002@[-6.1118,-3.09]",
                by_case["collision"].source_provenance["stratum"])
            self.assertEqual(by_case["support-change"].engine_relative_action["release_offset"], (-80, 3))
            self.assertIn("lowest-id-supported-entity:block:0000@[5.4977,-3.1]",
                by_case["support-change"].source_provenance["stratum"])
            self.assertIn("empty-space-engine-target-v1:[8.0,6.0]",
                by_case["no-contact"].source_provenance["selection_rule"])
            self.assertIn("initial-persistent-support-window",
                by_case["support"].source_provenance["selection_rule"])
            self.assertEqual(result["plan_identity"], loaded.plan.identity)
            self.assertEqual(result["probe_cases"], sorted(cases))
            first = output.read_bytes()
            self.assertEqual(build_issue_44_probe_plan(output)["plan_identity"], loaded.plan.identity)
            self.assertEqual(output.read_bytes(), first)
            self.assertEqual(json.loads(first)["identity"], loaded.plan.identity)


if __name__ == "__main__":
    unittest.main()
