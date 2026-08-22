import copy
import tempfile
import unittest
from pathlib import Path

from scripts.cohort_partition import create_cohort_partition_manifest
from scripts.cohort_v2_partition import create_cohort_v2_partition_exposure_manifest
from scripts.final_evaluation_access import (
    FinalEvaluationWorkflowAccessRejected,
    FinalEvaluationWorkflowAccessManifest,
    audit_final_evaluation_workflow_access,
    audit_final_evaluation_access,
    audit_ordinary_workflow_access,
    authorize_final_evaluation_workflow_access,
    create_final_evaluation_workflow_access_manifest,
    create_final_evaluation_access_manifest,
    load_final_evaluation_access_manifest,
    write_final_evaluation_access_manifest,
)
from tests.test_cohort_partition import entry, fixture_projection, provenance
from tests.test_cohort_v2_partition import ROLES, inventory_entries, partition_manifest


class FinalEvaluationAccessTests(unittest.TestCase):
    def setUp(self):
        self.training = fixture_projection(seed=1, template="training-template")
        self.final = fixture_projection(seed=2, template="final-template")
        self.final_lineage = str(self.final["scenario_lineage_identity"])
        self.partition = create_cohort_partition_manifest(
            partition_version=1,
            split_regime="template_held_out",
            held_out_roles=["final_evaluation"],
            entries=[
                entry(self.training, "training", "training"),
                entry(self.final, "final", "final_evaluation"),
            ],
            provenance_records=[
                provenance(
                    self.training,
                    artifact_identity="artifact:training",
                ),
                provenance(
                    self.final,
                    artifact_identity="artifact:final",
                ),
            ],
        )
        self.access = create_final_evaluation_access_manifest(
            self.partition,
            access_version=1,
            workflow_identity="final-workflow:v1",
        )
        self.observed = [{
            "artifact_identity": "artifact:final",
            "workflow_identity": "final-workflow:v1",
            "consumer_exposure_role": "final_evaluation",
            "source_scenario_lineage_identities": [self.final_lineage],
        }]

    def test_manifest_is_versioned_partition_bound_and_round_trips(self):
        self.assertEqual(
            [item.artifact_identity for item in self.access.authorized_artifacts],
            ["artifact:final"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "final-access.json"
            write_final_evaluation_access_manifest(self.access, path)
            self.assertEqual(load_final_evaluation_access_manifest(path), self.access)

    def test_ordinary_workflows_accept_only_declared_nonfinal_access(self):
        training_lineage = str(self.training["scenario_lineage_identity"])
        observed = [{
            "artifact_identity": "artifact:training",
            "source_scenario_lineage_identities": [training_lineage],
        }]
        for workflow_kind in (
            "training",
            "calibration",
            "model_selection",
            "pilot",
        ):
            with self.subTest(workflow_kind=workflow_kind):
                audit = audit_ordinary_workflow_access(
                    self.partition,
                    workflow_kind=workflow_kind,
                    observed_scenario_lineage_identities=[training_lineage],
                    observed_artifact_accesses=observed,
                )
                self.assertTrue(audit["passed"])

    def test_ordinary_workflows_reject_final_and_undeclared_access(self):
        training_lineage = str(self.training["scenario_lineage_identity"])
        cases = (
            (
                [self.final_lineage],
                [],
                "final_evaluation lineage",
            ),
            (
                [training_lineage],
                [{
                    "artifact_identity": "artifact:final",
                    "source_scenario_lineage_identities": [self.final_lineage],
                }],
                "final_evaluation data",
            ),
            (
                [training_lineage],
                [{
                    "artifact_identity": "artifact:unknown",
                    "source_scenario_lineage_identities": [training_lineage],
                }],
                "undeclared artifact",
            ),
            (
                [training_lineage],
                [{
                    "artifact_identity": "artifact:training",
                    "source_scenario_lineage_identities": [self.final_lineage],
                }],
                "source provenance differs",
            ),
        )
        for lineages, artifacts, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    audit_ordinary_workflow_access(
                        self.partition,
                        workflow_kind="training",
                        observed_scenario_lineage_identities=lineages,
                        observed_artifact_accesses=artifacts,
                    )

    def test_declared_final_workflow_access_passes(self):
        audit = audit_final_evaluation_access(
            self.partition,
            self.access,
            observed_accesses=self.observed,
            ordinary_workflow_lineage_identities=[
                str(self.training["scenario_lineage_identity"])
            ],
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["observed_artifact_count"], 1)

    def test_role_workflow_artifact_and_source_leakage_fail_closed(self):
        mutations = {
            "ordinary role": ("consumer_exposure_role", "training", "ordinary workflow"),
            "wrong workflow": ("workflow_identity", "other", "undeclared workflow"),
            "unknown artifact": ("artifact_identity", "other", "undeclared artifact"),
            "wrong source": (
                "source_scenario_lineage_identities",
                [str(self.training["scenario_lineage_identity"])],
                "source provenance",
            ),
        }
        for name, (field, value, message) in mutations.items():
            with self.subTest(name=name):
                observed = copy.deepcopy(self.observed)
                observed[0][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    audit_final_evaluation_access(
                        self.partition,
                        self.access,
                        observed_accesses=observed,
                    )

    def test_ordinary_lineage_inventory_cannot_include_final_evaluation(self):
        with self.assertRaisesRegex(ValueError, "Ordinary workflow"):
            audit_final_evaluation_access(
                self.partition,
                self.access,
                observed_accesses=[],
                ordinary_workflow_lineage_identities=[self.final_lineage],
            )

    def test_access_manifest_tampering_and_partition_mismatch_fail_closed(self):
        payload = self.access.to_dict()
        payload["workflow_identity"] = "changed"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tampered.json"
            path.write_text(__import__("json").dumps(payload), encoding="utf-8")
            changed_access = load_final_evaluation_access_manifest(path)
            self.assertEqual(changed_access.workflow_identity, "changed")
            with self.assertRaisesRegex(ValueError, "undeclared workflow"):
                audit_final_evaluation_access(
                    self.partition,
                    changed_access,
                    observed_accesses=self.observed,
                )

        other = create_cohort_partition_manifest(
            partition_version=2,
            split_regime="template_held_out",
            held_out_roles=["final_evaluation"],
            entries=[
                entry(self.training, "training", "training"),
                entry(self.final, "final", "final_evaluation"),
            ],
            provenance_records=[],
        )
        with self.assertRaisesRegex(ValueError, "different partition"):
            audit_final_evaluation_access(
                other,
                self.access,
                observed_accesses=[],
            )


class FinalEvaluationWorkflowAccessTests(unittest.TestCase):
    def setUp(self):
        self.partition = partition_manifest()
        final = next(
            entry
            for entry in self.partition.entries
            if entry.exposure_role == "final_evaluation"
        )
        self.workflow = create_final_evaluation_workflow_access_manifest(
            self.partition,
            workflow_version=1,
            workflow_identity="central-v2-final-evaluation-workflow-v1",
            operator_identity="novphy-operator-v1:final-evaluation-custodian",
            frozen_at="2026-08-22T01:00:00Z",
            authorized_artifacts=[{
                "artifact_kind": "scenario_manifest",
                "artifact_identity": final.scenario_manifest_identity,
                "source_scenario_lineage_identities": [
                    final.scenario_lineage_identity
                ],
            }],
        )
        self.record = {
            "workflow_identity": self.workflow.workflow_identity,
            "operator_identity": self.workflow.operator_identity,
            "artifact_identity": final.scenario_manifest_identity,
            "source_scenario_lineage_identities": [final.scenario_lineage_identity],
            "accessed_at": "2026-08-22T03:00:00Z",
            "authorization_identity": "github-issue-authorization-v1:47:release",
            "consumer_exposure_role": "final_evaluation",
        }

    def test_pending_workflow_rejects_access_then_exact_authorized_record_passes(self):
        self.assertEqual(
            FinalEvaluationWorkflowAccessManifest.from_dict(self.workflow.to_dict()),
            self.workflow,
        )
        self.assertEqual(self.workflow.authorization_state, "pending")
        with self.assertRaisesRegex(ValueError, "not authorized") as rejected:
            audit_final_evaluation_workflow_access(
                self.partition,
                self.workflow,
                observed_accesses=[self.record],
            )
        self.assertIsInstance(rejected.exception, FinalEvaluationWorkflowAccessRejected)
        self.assertEqual(
            rejected.exception.audit_record["artifact_identity"],
            self.record["artifact_identity"],
        )
        self.assertEqual(
            rejected.exception.audit_record["authorization_identity"],
            self.record["authorization_identity"],
        )
        self.assertFalse(rejected.exception.audit_record["passed"])

        authorized = authorize_final_evaluation_workflow_access(
            self.workflow,
            authorization_identity="github-issue-authorization-v1:47:release",
            authorized_at="2026-08-22T02:00:00Z",
        )
        report = audit_final_evaluation_workflow_access(
            self.partition,
            authorized,
            observed_accesses=[self.record],
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["observed_access_count"], 1)

        tampered = self.workflow.to_dict()
        tampered["operator_identity"] = "novphy-operator-v1:forged"
        with self.assertRaisesRegex(ValueError, "identity"):
            FinalEvaluationWorkflowAccessManifest.from_dict(tampered)

    def test_workflow_rejects_artifact_absent_from_the_frozen_final_partition(self):
        final = next(
            entry for entry in self.partition.entries
            if entry.exposure_role == "final_evaluation"
        )
        with self.assertRaisesRegex(ValueError, "frozen final partition"):
            create_final_evaluation_workflow_access_manifest(
                self.partition,
                workflow_version=1,
                workflow_identity="central-v2-final-evaluation-workflow-v1",
                operator_identity="novphy-operator-v1:final-evaluation-custodian",
                frozen_at="2026-08-22T01:00:00Z",
                authorized_artifacts=[{
                    "artifact_kind": "derivation_artifact",
                    "artifact_identity": "derivation:forged",
                    "source_scenario_lineage_identities": [
                        final.scenario_lineage_identity
                    ],
                }],
            )

    def test_workflow_operator_artifact_source_time_and_authorization_fail_closed(self):
        authorized = authorize_final_evaluation_workflow_access(
            self.workflow,
            authorization_identity="github-issue-authorization-v1:47:release",
            authorized_at="2026-08-22T02:00:00Z",
        )
        cases = (
            ("workflow_identity", "other", "wrong workflow"),
            ("operator_identity", "other", "wrong operator"),
            ("artifact_identity", "other", "undeclared artifact"),
            (
                "source_scenario_lineage_identities",
                ["scenario-lineage:training"],
                "source provenance",
            ),
            ("accessed_at", "2026-08-22T01:30:00Z", "predates authorization"),
            ("authorization_identity", "other", "wrong authorization"),
            ("consumer_exposure_role", "training", "non-final workflow"),
        )
        for field, value, message in cases:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                record = copy.deepcopy(self.record)
                record[field] = value
                audit_final_evaluation_workflow_access(
                    self.partition,
                    authorized,
                    observed_accesses=[record],
                )

        other_partition = create_cohort_v2_partition_exposure_manifest(
            partition_version=2,
            source_inventory_identity=self.partition.source_inventory_identity,
            source_inventory_review_url=self.partition.source_inventory_review_url,
            inventory_entries=inventory_entries(),
            lineage_quotas={role: 1 for role in ROLES},
        )
        with self.assertRaisesRegex(ValueError, "different partition"):
            audit_final_evaluation_workflow_access(
                other_partition,
                authorized,
                observed_accesses=[],
            )


if __name__ == "__main__":
    unittest.main()
