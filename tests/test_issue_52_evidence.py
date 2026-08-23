from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from scripts.build_issue_52_evidence import validate_issue_52_evidence
from scripts.cohort_v2_production_plans import (
    CENTRAL_STRATA,
    COLLECTION_IDENTITY,
    PARAMETER_IDENTITY,
    ROOT,
    derive_issue_52_payloads,
    validate_issue_52_payloads,
)


ISSUE_52 = ROOT / "data/runtime_evidence/issue-52"


class Issue52ProductionPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payloads = derive_issue_52_payloads(ROOT, validate_pilot=False)

    def test_plans_freeze_all_quota_dimensions_and_sealed_assignments(self) -> None:
        collection = self.payloads["collection-plan.json"]
        parameters = self.payloads["production-parameter-plan.json"]

        self.assertEqual(collection["identity"], COLLECTION_IDENTITY)
        self.assertEqual(parameters["identity"], PARAMETER_IDENTITY)
        self.assertEqual(
            list(collection["quotas"]["central_coverage_stratum"]),
            list(CENTRAL_STRATA),
        )
        self.assertEqual(
            collection["quotas"]["total_planned_rollouts"]["quota"], 24
        )
        self.assertEqual(
            set(collection["quotas"]),
            {
                "total_planned_rollouts",
                "benchmark_condition",
                "exposure_role",
                "instance_held_out_partition",
                "scenario_template_level_instance",
                "intervention_source",
                "central_coverage_stratum",
                "termination_class",
                "required_capability_coverage",
            },
        )
        final = collection["assignments"][-1]
        self.assertEqual(final["exposure_role"], "final_evaluation")
        self.assertIn("sealed_scenario_manifest_reference", final)
        self.assertNotIn("scenario_manifest_reference", final)
        self.assertNotIn("scenario_manifest", final)

    def test_actions_and_attempt_policy_are_fixed_before_outcomes(self) -> None:
        collection = self.payloads["collection-plan.json"]
        for intervention in collection["interventions"]:
            self.assertEqual(
                intervention["interface_action"]["drag_release"],
                intervention["engine_relative_action"]["drag_delta_canvas_pixels"],
            )
        self.assertEqual(
            {item["intervention_source"] for item in collection["interventions"]},
            {"geometry_stratified", "targeted_rare"},
        )
        self.assertEqual(collection["attempt_policy"]["retry_counts"], {})
        self.assertFalse(collection["attempt_policy"]["quota_fill_resampling"])
        self.assertTrue(collection["attempt_policy"]["outcome_independent_accounting"])
        self.assertEqual(collection["bounded_negative"]["cap"], 0)
        self.assertIn("not_required", collection["bounded_negative"]["status"])

    def test_every_parameter_value_resolves_complete_non_final_evidence(self) -> None:
        plan = self.payloads["production-parameter-plan.json"]
        evidence = plan["evidence"]

        def visit(value):
            if isinstance(value, dict):
                if set(value) == {"value", "unit", "evidence_id"}:
                    source = evidence[value["evidence_id"]]
                    self.assertEqual(source["plan_version"], plan["plan_version"])
                    self.assertTrue(source["analysis_method"])
                    self.assertTrue(source["observed_range_or_uncertainty"])
                    self.assertTrue(source["decision_rule"])
                    self.assertTrue(source["rationale"])
                    self.assertTrue(source["source_record_ids"])
                    self.assertEqual(
                        {
                            plan["source_records"][source_record_id][
                                "exposure_boundary"
                            ]
                            for source_record_id in source["source_record_ids"]
                        },
                        {"no_final_evaluation_outcomes"},
                    )
                    return
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(plan["parameters"])

    def test_unknown_identity_missing_evidence_quota_drift_and_parameter_drift_fail(self) -> None:
        cases = []

        unknown = deepcopy(self.payloads)
        unknown["collection-plan.json"]["identity"] = "unknown-plan"
        cases.append((unknown, "identity or version is unknown"))

        missing_evidence = deepcopy(self.payloads)
        missing_evidence["production-parameter-plan.json"]["evidence"][
            "capture_stride"
        ]["source_record_ids"] = []
        cases.append((missing_evidence, "not source-bound"))

        quota_drift = deepcopy(self.payloads)
        quota_drift["collection-plan.json"]["quotas"]["termination_class"][
            "stable_entered"
        ]["quota"] = 11
        cases.append((quota_drift, "termination_class quotas"))

        assignment_drift = deepcopy(self.payloads)
        assignment_drift["collection-plan.json"]["assignments"][0][
            "scenario_lineage_identity"
        ] = "scenario-lineage-v1:mutated"
        cases.append((assignment_drift, "frozen exposure manifest"))

        parameter_drift = deepcopy(self.payloads)
        parameter_drift["production-parameter-plan.json"]["parameters"]["capture"][
            "rollout_ceiling_fixed_steps"
        ]["value"] = 601
        cases.append((parameter_drift, "outside their source-bound decision rules"))

        for payloads, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_issue_52_payloads(payloads, ROOT)

    def test_published_bundle_exactly_rederives(self) -> None:
        if not ISSUE_52.is_dir():
            self.skipTest("issue-52 immutable bundle is not published")
        result = validate_issue_52_evidence(
            ISSUE_52,
            repository_root=ROOT,
            revalidate_pilot=False,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["planned_rollouts"], 24)


if __name__ == "__main__":
    unittest.main()
