import unittest
from pathlib import Path

from scripts.build_issue_54_evidence import (
    build_ingestion_evidence,
    run_adversarial_ingestion_checks,
)
from world_model.data import (
    CohortV2IngestionError,
    CohortV2OracleWindowDataset,
    CohortV2ReleaseReader,
    probe_cohort_v2_final_access,
)
from world_model.data.cohort_v2 import CENTRAL_LABELS
from world_model.training import (
    build_cohort_v2_oracle_window_loader,
    score_cohort_v2_endpoints,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_RELEASE = ROOT / "data/runtime_evidence/issue-53-mixed-termination-v5"
CAPABILITY_DECLARATION = ROOT / "docs/data_contracts/cohort_v2_capabilities_v1.json"
PRODUCTION_PLAN = ROOT / "data/runtime_evidence/issue-53-plan-v5"
SEALED_RELEASE = ROOT / ".local-artifacts/issue-53-mixed-termination-final-release-v5"


class CohortV2ReleaseReaderTests(unittest.TestCase):
    def test_training_reader_exposes_observation_backed_central_windows(self) -> None:
        reader = CohortV2ReleaseReader(
            PUBLIC_RELEASE,
            capability_declaration_path=CAPABILITY_DECLARATION,
            production_plan_root=PRODUCTION_PLAN,
            workflow_kind="training",
            influence="learned_parameters",
        )

        self.assertEqual(reader.release_identity, "representative-cohort-v2-release-v5:issue-53:mixed-termination")
        self.assertEqual(len(reader.rollouts), 6)
        self.assertEqual(
            {rollout.coverage_stratum for rollout in reader.rollouts},
            {
                "no-contact/miss",
                "collision",
                "persistent support",
                "support change",
                "destruction",
                "stability transitions",
            },
        )

        dataset = CohortV2OracleWindowDataset(reader)
        self.assertEqual(len(dataset), 6)
        loader = build_cohort_v2_oracle_window_loader(dataset)
        example = next(iter(loader))[0]
        self.assertEqual(set(example.context.labels), set(CENTRAL_LABELS))
        self.assertEqual(example.target.fixed_step, example.context.fixed_step + 1)
        self.assertTrue(example.agent_observation.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(example.intervention["interface_action"]["coordinate_frame"], "slingshot_relative")
        self.assertIsNone(example.context.labels["steady-state"]["value"])
        self.assertNotEqual(example.context.labels["steady-state"]["availability"], "available")

    def test_access_and_capability_boundaries_fail_before_examples(self) -> None:
        with self.assertRaisesRegex(CohortV2IngestionError, "capabilit"):
            CohortV2ReleaseReader(
                PUBLIC_RELEASE,
                capability_declaration_path=CAPABILITY_DECLARATION,
                production_plan_root=PRODUCTION_PLAN,
                workflow_kind="training",
                influence="learned_parameters",
                requested_capabilities=("physical_regime_gate",),
            )
        with self.assertRaisesRegex(CohortV2IngestionError, "sealed final"):
            CohortV2ReleaseReader(
                PUBLIC_RELEASE,
                capability_declaration_path=CAPABILITY_DECLARATION,
                production_plan_root=PRODUCTION_PLAN,
                workflow_kind="final_evaluation",
                influence="frozen_final_metrics_after_authorization",
            )
        with self.assertRaisesRegex(CohortV2IngestionError, "not permitted"):
            CohortV2ReleaseReader(
                PUBLIC_RELEASE,
                capability_declaration_path=CAPABILITY_DECLARATION,
                production_plan_root=PRODUCTION_PLAN,
                workflow_kind="calibration",
                influence="learned_parameters",
            )

        reader = CohortV2ReleaseReader(
            PUBLIC_RELEASE,
            capability_declaration_path=CAPABILITY_DECLARATION,
            production_plan_root=PRODUCTION_PLAN,
            workflow_kind="training",
            influence="learned_parameters",
        )
        with self.assertRaisesRegex(CohortV2IngestionError, "canonical observation"):
            reader.load_observation(reader.rollouts[0], observation_role="canonical")

    def test_endpoint_scoring_consumes_available_values_and_skips_unavailable(self) -> None:
        reader = CohortV2ReleaseReader(
            PUBLIC_RELEASE,
            capability_declaration_path=CAPABILITY_DECLARATION,
            production_plan_root=PRODUCTION_PLAN,
            workflow_kind="model_selection",
            influence="configuration_selection",
        )

        def all_false(frame_record):
            return {
                "contact": (),
                "supports": (),
                "steady-state": False,
                "structure-unstable": False,
                "excess_penetration": False,
                "unsupported_stationary_or_floating_body": {
                    item["entity_id"]: False
                    for item in frame_record.labels[
                        "unsupported_stationary_or_floating_body"
                    ]
                },
            }

        score = score_cohort_v2_endpoints(reader, all_false)
        self.assertEqual(score.endpoint_count, 6)
        self.assertGreater(score.scored_value_count, 0)
        self.assertGreater(score.unavailable_value_count, 0)
        self.assertGreater(score.scored_relation_count, 0)
        self.assertLessEqual(score.correct_value_count, score.scored_value_count)

    def test_adversarial_ingestion_suite_covers_capabilities_and_boundaries(self) -> None:
        reader = CohortV2ReleaseReader(
            PUBLIC_RELEASE,
            capability_declaration_path=CAPABILITY_DECLARATION,
            production_plan_root=PRODUCTION_PLAN,
            workflow_kind="training",
            influence="learned_parameters",
        )
        checks = run_adversarial_ingestion_checks(
            PUBLIC_RELEASE, CAPABILITY_DECLARATION, PRODUCTION_PLAN, reader
        )

        self.assertEqual(len(checks), 16)
        self.assertTrue(all(item["passed"] for item in checks))

    @unittest.skipUnless(SEALED_RELEASE.is_dir(), "sealed operator evidence is local")
    def test_authorized_final_probe_returns_audit_receipt_without_final_examples(self) -> None:
        receipt = probe_cohort_v2_final_access(PUBLIC_RELEASE, SEALED_RELEASE)

        self.assertTrue(receipt.passed)
        self.assertEqual(receipt.authorization_state, "authorized")
        self.assertEqual(receipt.observed_access_count, 1)
        self.assertFalse(hasattr(receipt, "rollouts"))

    @unittest.skipUnless(SEALED_RELEASE.is_dir(), "sealed operator evidence is local")
    def test_evidence_builder_runs_every_role_and_adversarial_boundary(self) -> None:
        report = build_ingestion_evidence(
            repository_root=ROOT,
            release_root=PUBLIC_RELEASE,
            sealed_root=SEALED_RELEASE,
            code_revision="test-revision",
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["counts"]["rollouts"], 18)
        self.assertEqual(report["counts"]["oracle_training_windows"], 18)
        self.assertEqual(set(report["roles"]), {"training", "calibration", "model_selection"})
        self.assertEqual(len(report["adversarial_checks"]), 16)
        self.assertTrue(all(item["passed"] for item in report["adversarial_checks"]))


if __name__ == "__main__":
    unittest.main()
