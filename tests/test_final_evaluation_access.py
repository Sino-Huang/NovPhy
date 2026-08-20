import copy
import tempfile
import unittest
from pathlib import Path

from scripts.cohort_partition import create_cohort_partition_manifest
from scripts.final_evaluation_access import (
    audit_final_evaluation_access,
    audit_ordinary_workflow_access,
    create_final_evaluation_access_manifest,
    load_final_evaluation_access_manifest,
    write_final_evaluation_access_manifest,
)
from tests.test_cohort_partition import entry, fixture_projection, provenance


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


if __name__ == "__main__":
    unittest.main()
