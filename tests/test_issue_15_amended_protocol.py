from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.final_evaluation_access import FinalEvaluationWorkflowAccessManifest
from scripts.issue_15_amended_protocol import (
    FINAL_SEED,
    build_plan,
    build_protocol,
    materialize_final_authority,
    validate_protocol,
)


class Issue15AmendedProtocolTests(unittest.TestCase):
    def test_amendment_freezes_new_attempts_and_capacity_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            scenario, _manifest, _xml = materialize_final_authority(Path(directory))
            plan = build_plan(scenario)
            protocol = build_protocol(
                plan, implementation_commit="commit:fixture"
            )

        collection = plan["confirmatory-plan.json"]
        workflow = FinalEvaluationWorkflowAccessManifest.from_dict(
            plan["final-evaluation-workflow-access-manifest.json"]
        )
        self.assertEqual(FINAL_SEED, 4505)
        self.assertEqual(len(collection["attempt_ids"]), 6)
        self.assertEqual(len(set(collection["attempt_ids"])), 6)
        self.assertTrue(all("seed-4505" in item for item in collection["attempt_ids"]))
        self.assertTrue(all("seed-4504" not in item for item in collection["attempt_ids"]))
        self.assertEqual(workflow.authorization_state, "pending")
        self.assertEqual(
            protocol["calibration_basis"]["capacity"],
            {"max_entities": 15, "latent_dim": 197},
        )
        self.assertFalse(
            protocol["exposure_audit"][
                "new_seed_4505_scenario_outcomes_or_metrics_accessed"
            ]
        )
        self.assertEqual(
            protocol["replicate_and_seed_policy"]["fixed_attempt_ids"],
            collection["attempt_ids"],
        )
        issue_15 = protocol["experiment_matrix"][
            "confirmatory_oracle_symbol_issue_15"
        ]
        issue_16 = protocol["experiment_matrix"][
            "learned_feature_symbol_stress_issue_16"
        ]
        self.assertEqual(
            [item["budget"] for item in issue_16["comparisons"]],
            [item["budget"] for item in issue_15["comparisons"]],
        )
        self.assertIn(
            "not_supported_by_this_experiment",
            protocol["required_outputs"]["decision_values"],
        )
        self.assertNotIn(
            "unsupported", protocol["required_outputs"]["decision_values"]
        )
        self.assertEqual(validate_protocol(protocol), protocol)


if __name__ == "__main__":
    unittest.main()
