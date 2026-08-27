from __future__ import annotations

import json
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.cohort_v2_statistical_protocol import load_protocol
from scripts.final_evaluation_access import FinalEvaluationWorkflowAccessManifest
from scripts.run_cohort_v2_confirmatory import _compact_summary
from world_model.data import CohortV2FinalEvaluationReader, CohortV2IngestionError
from world_model.training.cohort_v2_confirmatory import (
    CANDIDATE_ID,
    CohortV2ConfirmatoryRecord,
    analyze_cohort_v2_confirmatory,
    validate_cohort_v2_confirmatory_evidence,
    write_cohort_v2_confirmatory_evidence,
)
from world_model.training import cohort_v2_confirmatory as confirmatory
from world_model.training.cohort_v2_evaluation import (
    _validate_reader_bindings,
    cohort_v2_evaluation_state_set_identity,
)
from world_model.training.cohort_v2_integrated import (
    CohortV2RecursiveRolloutRecord,
)
from tests.test_cohort_v2_macro_training import _frame


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/runtime_evidence/issue-34/cohort-v2-prospective-statistical-protocol-v1.json"


class CohortV2ConfirmatoryTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(PROTOCOL)
        self.attempts = self.protocol["replicate_and_seed_policy"]["fixed_attempt_ids"]

    def _records(self, candidate_error: float):
        records = []
        comparisons = self.protocol["experiment_matrix"][
            "confirmatory_oracle_symbol_issue_15"
        ]["comparisons"]
        for comparison in comparisons:
            budget = comparison["budget"]
            comparator_id = comparison["strongest_comparator_id"]
            for index, attempt in enumerate(self.attempts):
                common = {
                    "protocol_identity": self.protocol["artifact_identity"],
                    "release_identity": "release:fixture",
                    "partition_identity": "partition:fixture",
                    "code_revision": "commit:fixture",
                    "budget": budget,
                    "attempt_id": attempt,
                    "exposure_role": "final_evaluation",
                    "coverage_stratum": f"stratum:{index}",
                    "state_count": 2,
                    "mean_endpoint_violation_rate": 0.0,
                    "mean_policy_compute_per_simulated_frame": budget / 2,
                    "mean_full_compute_per_simulated_frame": budget / 2,
                }
                records.append(CohortV2ConfirmatoryRecord(
                    configuration_id=CANDIDATE_ID,
                    comparison_role="candidate",
                    checkpoint_identity="checkpoint:candidate",
                    seed=10,
                    mean_endpoint_prediction_error=candidate_error,
                    **common,
                ))
                records.append(CohortV2ConfirmatoryRecord(
                    configuration_id=comparator_id,
                    comparison_role="comparator",
                    checkpoint_identity=f"checkpoint:{comparator_id}",
                    seed=10,
                    mean_endpoint_prediction_error=0.1,
                    **common,
                ))
        return tuple(records)

    def _recursive(self):
        return tuple(
            CohortV2RecursiveRolloutRecord(
                checkpoint_identity="checkpoint:candidate",
                exposure_role="final_evaluation",
                attempt_id=attempt,
                scenario_lineage_identity=f"lineage:{index}",
                coverage_stratum=f"stratum:{index}",
                requested_horizon=15,
                simulated_duration=2,
                effective_horizons=(2,),
                cumulative_horizons=(2,),
                authoritative_endpoint_identities=(f"endpoint:{index}",),
                endpoint_mse_curve=(0.01,),
                terminal_mse=0.01,
                error_auc=0.01,
                total_compute=100.0,
            )
            for index, attempt in enumerate(self.attempts)
        )

    def test_frozen_rule_supports_only_a_budget_that_passes_both_bounds(self):
        report = analyze_cohort_v2_confirmatory(
            self._records(candidate_error=0.01),
            self._recursive(),
            self.protocol,
            source_bindings={"fixture": True},
        )

        self.assertEqual(report["decision"], "supported")
        self.assertTrue(all(
            item["gain_rule_passed"] for item in report["budget_decisions"]
        ))
        self.assertTrue(all(
            item["violation_rule_passed"] for item in report["budget_decisions"]
        ))

    def test_no_passing_budget_marks_the_hypothesis_unsupported(self):
        report = analyze_cohort_v2_confirmatory(
            self._records(candidate_error=0.2),
            self._recursive(),
            self.protocol,
            source_bindings={"fixture": True},
        )

        self.assertEqual(report["decision"], "unsupported")
        self.assertFalse(any(
            item["budget_rule_passed"] for item in report["budget_decisions"]
        ))

    def test_evidence_validates_by_exact_recomputation(self):
        records = self._records(candidate_error=0.01)
        recursive = self._recursive()
        report = analyze_cohort_v2_confirmatory(
            records,
            recursive,
            self.protocol,
            source_bindings={"fixture": True},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evidence"
            written = write_cohort_v2_confirmatory_evidence(
                root,
                records,
                recursive,
                report,
                implementation_revision="commit:fixture",
            )
            validated = validate_cohort_v2_confirmatory_evidence(
                root, self.protocol
            )
            self.assertEqual(
                validated["artifact_identity"], written["artifact_identity"]
            )

    def test_exhaustive_evaluator_accepts_one_authorized_final_reader(self):
        reader = SimpleNamespace(
            release_identity="release:fixture",
            partition_identity="partition:fixture",
            rollouts=(SimpleNamespace(exposure_role="final_evaluation"),),
        )
        release, partition, roles = _validate_reader_bindings((reader,))

        self.assertEqual(release, "release:fixture")
        self.assertEqual(partition, "partition:fixture")
        self.assertEqual(roles, ("final_evaluation",))
        public = cohort_v2_evaluation_state_set_identity(
            release, partition, ("state:1",)
        )
        final = cohort_v2_evaluation_state_set_identity(
            release, partition, ("state:1",), roles
        )
        self.assertNotEqual(public, final)

    def test_final_reader_rejects_pending_workflow_before_sealed_validation(self):
        pending = FinalEvaluationWorkflowAccessManifest.from_dict(
            json.loads((
                ROOT
                / "data/runtime_evidence/issue-53-plan-v5/"
                "final-evaluation-workflow-access-manifest.json"
            ).read_bytes())
        )
        artifact = pending.authorized_artifacts[0]
        observed = {
            "workflow_identity": pending.workflow_identity,
            "operator_identity": pending.operator_identity,
            "artifact_identity": artifact.artifact_identity,
            "source_scenario_lineage_identities": list(
                artifact.source_scenario_lineage_identities
            ),
            "accessed_at": "2026-08-27T00:00:00Z",
            "authorization_identity": "authorization:fixture",
            "consumer_exposure_role": "final_evaluation",
        }
        with patch(
            "world_model.data.cohort_v2.validate_published_issue_53_evidence"
        ) as sealed_validation:
            with self.assertRaises(CohortV2IngestionError):
                CohortV2FinalEvaluationReader(
                    ROOT / "data/runtime_evidence/issue-53-mixed-termination-v5",
                    ROOT / "does-not-exist",
                    capability_declaration_path=(
                        ROOT / "docs/data_contracts/cohort_v2_capabilities_v1.json"
                    ),
                    production_plan_root=(
                        ROOT / "data/runtime_evidence/issue-53-plan-v5"
                    ),
                    access_manifest=pending,
                    observed_accesses=[observed],
                )
        sealed_validation.assert_not_called()

    def test_entity_capacity_audit_catches_the_pre_evaluation_codec_failure(self):
        frame = _frame("frame:overflow", 0, steady=False, unstable=False)
        template = frame.engine_state["entities"][0]
        entities = tuple(
            {
                **template,
                "entity_id": f"entity:{index:02d}",
                "scenario_object_id": f"object:{index:02d}",
            }
            for index in range(15)
        )
        overflow = replace(
            frame,
            engine_state={
                "entities": entities,
                "world": frame.engine_state["world"],
            },
        )
        reader = SimpleNamespace(rollouts=(SimpleNamespace(
            attempt_id="attempt:final",
            coverage_stratum="collision",
            frame_records=(overflow,),
        ),))

        audit = confirmatory.audit_final_entity_capacity(reader, max_entities=12)

        self.assertEqual(audit[0]["attempt_id"], "attempt:final")
        self.assertEqual(audit[0]["maximum_entity_count"], 15)
        self.assertEqual(audit[0]["declared_entity_slots"], 12)
        self.assertEqual(audit[0]["failure_code"], "entity_slot_capacity_exceeded")

    def test_capacity_failures_are_retained_and_make_both_budgets_unsupported(self):
        capacity_audit = tuple({
            "attempt_id": attempt,
            "coverage_stratum": f"stratum:{index}",
            "frame_record_count": 10,
            "maximum_entity_count": 15,
            "declared_entity_slots": 12,
            "overflow_frame_count": 10,
            "failure_code": "entity_slot_capacity_exceeded",
            "passed": False,
        } for index, attempt in enumerate(self.attempts))

        report = analyze_cohort_v2_confirmatory(
            (),
            (),
            self.protocol,
            source_bindings={"failed_implementation_commit": "commit:fixture"},
            capacity_audit=capacity_audit,
        )

        self.assertEqual(report["decision"], "unsupported")
        self.assertEqual(report["bootstrap"]["replicates"], 0)
        self.assertEqual(len(report["failed_missing_or_excluded_runs"]), 6)
        self.assertTrue(all(
            not item["budget_rule_passed"] for item in report["budget_decisions"]
        ))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "failure-evidence"
            written = write_cohort_v2_confirmatory_evidence(
                root,
                (),
                (),
                report,
                implementation_revision="commit:finalizer",
                capacity_audit=capacity_audit,
            )
            validated = validate_cohort_v2_confirmatory_evidence(
                root, self.protocol
            )
            self.assertEqual(
                validated["artifact_identity"], written["artifact_identity"]
            )

    def test_compact_summary_digests_large_source_identities(self):
        large = "source:" + "x" * 100_000
        report = {
            "decision": "unsupported",
            "decision_rationale": "fixture",
            "budget_decisions": [],
            "fixed_h15_complete_rollout": {"status": "not_run"},
            "failed_missing_or_excluded_runs": [],
            "source_bindings": {
                "access_audit": {
                    "schema": "final_evaluation_workflow_access_audit_v1",
                    "authorization_state": "authorized",
                    "authorization_identity": "authorization:fixture",
                    "observed_access_count": 1,
                    "passed": True,
                    "workflow_identity": "workflow:fixture",
                    "partition_identity": large,
                    "workflow_manifest_identity": large,
                },
                "access_manifest_identity": large,
                "candidate_checkpoint_identity": large,
                "failed_implementation_commit": "commit:failed",
                "failure_exception": "failure:fixture",
                "failure_phase": "pre_evaluation",
                "finalization_implementation_commit": "commit:finalizer",
                "sealed_bundle_identity": "sealed:fixture",
            },
        }

        compact = _compact_summary(
            {"artifact_identity": "artifact:fixture"},
            report,
            self.protocol,
            "commit:finalizer",
        )

        encoded = json.dumps(compact)
        self.assertLess(len(encoded), 10_000)
        self.assertNotIn(large, encoded)


if __name__ == "__main__":
    unittest.main()
