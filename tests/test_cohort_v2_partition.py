from __future__ import annotations

import unittest

from scripts.cohort_v2_partition import (
    CohortV2PartitionExposureManifest,
    audit_cohort_v2_partition_exposure,
    audit_cohort_v2_workflow_influence,
    create_cohort_v2_partition_exposure_manifest,
)
from scripts.cohort_v2_scenarios import ScenarioInventoryEntry


ROLES = ("training", "calibration", "model_selection", "final_evaluation")


def inventory_entries() -> tuple[ScenarioInventoryEntry, ...]:
    entries = []
    for index, role in enumerate(ROLES, start=1):
        common = {
            "exposure_role": role,
            "inventory_state": (
                "sealed_final" if role == "final_evaluation" else "planned_non_final"
            ),
            "scenario_manifest_identity": f"scenario-manifest:{role}",
            "benchmark_condition_identity": f"benchmark-condition:{index}",
            "scenario_template_identity": f"scenario-template:{(index - 1) % 2}",
            "level_instance_identity": f"level-instance:{role}",
            "scenario_specification_identity": f"scenario-specification:{role}",
            "scenario_lineage_identity": f"scenario-lineage:{role}",
            "declared_initial_engine_state_identity": f"initial-state:{role}",
        }
        if role == "final_evaluation":
            common["sealed_scenario_manifest_reference"] = "sealed:final"
        else:
            common["scenario_manifest_reference"] = f"{role}.json"
        entries.append(ScenarioInventoryEntry.from_dict(common))
    return tuple(entries)


def partition_manifest() -> CohortV2PartitionExposureManifest:
    return create_cohort_v2_partition_exposure_manifest(
        partition_version=1,
        source_inventory_identity="central-v2-scenario-inventory-v1:issue-45-reviewed",
        source_inventory_review_url=(
            "https://github.com/Sino-Huang/NovPhy/issues/45#issuecomment-5358450102"
        ),
        inventory_entries=inventory_entries(),
        lineage_quotas={role: 1 for role in ROLES},
    )


def provenance_records() -> list[dict[str, object]]:
    records = []
    for entry in partition_manifest().entries:
        for kind in (
            "derivation_artifact",
            "generation_seed",
            "intervention",
            "observation_configuration",
            "observation_variant",
            "replay",
            "rerun",
        ):
            records.append({
                "artifact_kind": kind,
                "artifact_identity": f"{kind}:{entry.exposure_role}",
                "source_scenario_lineage_identity": entry.scenario_lineage_identity,
                "level_instance_identity": entry.level_instance_identity,
                "scenario_template_identity": entry.scenario_template_identity,
                "dataset_partition": entry.dataset_partition,
                "exposure_role": entry.exposure_role,
            })
    return records


class CohortV2PartitionExposureTests(unittest.TestCase):
    def test_manifest_freezes_all_four_roles_permissions_and_lineage_quotas(self) -> None:
        manifest = partition_manifest()

        self.assertEqual(
            CohortV2PartitionExposureManifest.from_dict(manifest.to_dict()),
            manifest,
        )
        self.assertEqual(manifest.split_regime, "instance_held_out")
        self.assertEqual(manifest.quota_scope, "partition_lineage_membership")
        self.assertEqual(
            manifest.central_evidence_floor,
            {
                "minimum_level_instances": 2,
                "minimum_non_final_scenario_lineages": 2,
                "minimum_scenario_templates": 2,
            },
        )
        self.assertEqual(
            {entry.exposure_role for entry in manifest.entries},
            set(ROLES),
        )
        self.assertEqual(
            next(
                entry for entry in manifest.entries if entry.exposure_role == "training"
            ).may_influence,
            ("learned_parameters",),
        )
        self.assertEqual(
            next(
                entry
                for entry in manifest.entries
                if entry.exposure_role == "final_evaluation"
            ).inventory_state,
            "sealed_final",
        )
        next_version = create_cohort_v2_partition_exposure_manifest(
            partition_version=2,
            source_inventory_identity=manifest.source_inventory_identity,
            source_inventory_review_url=manifest.source_inventory_review_url,
            inventory_entries=inventory_entries(),
            lineage_quotas={role: 1 for role in ROLES},
        )
        self.assertNotEqual(next_version.identity, manifest.identity)

        tampered_fields = {
            "scenario_lineage_identity": "scenario-lineage:forged",
            "level_instance_identity": "level-instance:forged",
            "scenario_manifest_reference": "forged.json",
            "lineage_quota": 2,
        }
        for field, value in tampered_fields.items():
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity|quota"):
                tampered = manifest.to_dict()
                tampered["entries"][0][field] = value
                CohortV2PartitionExposureManifest.from_dict(tampered)

    def test_auditor_accepts_template_reuse_and_complete_same_role_provenance(self) -> None:
        manifest = partition_manifest()
        records = provenance_records()

        report = audit_cohort_v2_partition_exposure(
            manifest,
            declared_provenance_records=records,
            observed_artifact_identities=[
                str(record["artifact_identity"]) for record in records
            ],
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["scenario_lineage_count"], 4)
        self.assertEqual(report["level_instance_count"], 4)
        self.assertEqual(report["scenario_template_count"], 2)
        self.assertEqual(
            report["shared_scenario_template_identities"],
            ["scenario-template:0", "scenario-template:1"],
        )
        self.assertFalse(report["template_held_out_claim"])
        self.assertFalse(report["template_held_out_score"])

    def test_auditor_rejects_cross_role_or_undeclared_artifacts(self) -> None:
        manifest = partition_manifest()
        records = provenance_records()
        leaked = [dict(record) for record in records]
        leaked[0]["exposure_role"] = "calibration"

        with self.assertRaisesRegex(ValueError, "inherit"):
            audit_cohort_v2_partition_exposure(
                manifest,
                declared_provenance_records=leaked,
                observed_artifact_identities=[
                    str(record["artifact_identity"]) for record in leaked
                ],
            )

    def test_workflow_influence_is_limited_to_its_exact_role_and_permission(self) -> None:
        manifest = partition_manifest()
        records = provenance_records()
        training = next(
            entry for entry in manifest.entries if entry.exposure_role == "training"
        )
        training_artifact = next(
            record for record in records
            if record["artifact_kind"] == "observation_configuration"
            and record["exposure_role"] == "training"
        )
        report = audit_cohort_v2_workflow_influence(
            manifest,
            workflow_kind="training",
            influence="learned_parameters",
            declared_provenance_records=records,
            observed_scenario_lineage_identities=[training.scenario_lineage_identity],
            observed_artifact_identities=[str(training_artifact["artifact_identity"])],
        )
        self.assertTrue(report["passed"])

        calibration = next(
            entry for entry in manifest.entries if entry.exposure_role == "calibration"
        )
        with self.assertRaisesRegex(ValueError, "exposure role"):
            audit_cohort_v2_workflow_influence(
                manifest,
                workflow_kind="training",
                influence="learned_parameters",
                declared_provenance_records=records,
                observed_scenario_lineage_identities=[calibration.scenario_lineage_identity],
                observed_artifact_identities=[],
            )
        with self.assertRaisesRegex(ValueError, "permission"):
            audit_cohort_v2_workflow_influence(
                manifest,
                workflow_kind="calibration",
                influence="learned_parameters",
                declared_provenance_records=records,
                observed_scenario_lineage_identities=[calibration.scenario_lineage_identity],
                observed_artifact_identities=[],
            )

        with self.assertRaisesRegex(ValueError, "undeclared provenance"):
            audit_cohort_v2_partition_exposure(
                manifest,
                declared_provenance_records=records[1:],
                observed_artifact_identities=[
                    str(record["artifact_identity"]) for record in records
                ],
            )


if __name__ == "__main__":
    unittest.main()
