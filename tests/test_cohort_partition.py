import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.cohort_partition import (
    CohortPartitionManifest,
    audit_cohort_partition_manifest,
    create_cohort_partition_manifest,
    load_cohort_partition_manifest,
    write_cohort_partition_manifest,
)
from scripts.scenario_manifest import (
    BenchmarkCondition,
    ELIGIBLE,
    SMOKE_ONLY,
    create_generated_manifest,
    import_legacy_manifest,
    scenario_manifest_projection,
)


XML = b'''<?xml version="1.0" encoding="utf-8"?>
<Level width="2">
  <Camera maxWidth="30" minWidth="20" />
  <Birds><Bird type="BirdRed" /></Birds>
  <Slingshot x="-8" y="-2" />
  <GameObjects><Pig type="BasicSmall" x="1" y="-3" rotation="0" /></GameObjects>
</Level>
'''


def fixture_projection(
    *,
    template: str = "scenario-template-v1:fixture",
    seed: int = 41,
    xml: bytes = XML,
    eligibility: str = ELIGIBLE,
) -> dict[str, object]:
    manifest = create_generated_manifest(
        xml,
        benchmark_condition=BenchmarkCondition("novelty_level_1", "type0101"),
        template_identity=template,
        generator_identity="novphy-task-generator",
        generator_version="canonical-v1",
        generation_seed=seed,
        declared_inputs={"layout_choice": seed},
        parameter_realization={"shift_x": seed / 100},
        eligibility=eligibility,
        eligibility_reason="fixture smoke artifact" if eligibility == SMOKE_ONLY else None,
    )
    return scenario_manifest_projection(manifest, f"fixtures/{template}/{seed}.scenario.json")


def unavailable_template_projection() -> dict[str, object]:
    manifest = import_legacy_manifest(
        XML,
        benchmark_condition=BenchmarkCondition("novelty_level_1", "type0101"),
        source_path="fixtures/legacy.xml",
        eligibility=ELIGIBLE,
    )
    return scenario_manifest_projection(manifest, "fixtures/legacy.scenario.json")


def entry(projection: dict[str, object], partition: str, role: str) -> dict[str, object]:
    return {
        "dataset_partition": partition,
        "exposure_role": role,
        **copy.deepcopy(projection),
    }


def provenance(
    projection: dict[str, object],
    *,
    kind: str = "derivation_artifact",
    artifact_identity: str = "artifact-v1:fixture",
    consumer: str | None = None,
    sources: list[str] | None = None,
) -> dict[str, object]:
    lineage = str(projection["scenario_lineage_identity"])
    return {
        "artifact_kind": kind,
        "artifact_identity": artifact_identity,
        "consumer_scenario_lineage_identity": consumer or lineage,
        "source_scenario_lineage_identities": sources if sources is not None else [lineage],
    }


def four_role_entries() -> list[dict[str, object]]:
    roles = ("training", "calibration", "model_selection", "final_evaluation")
    return [
        entry(fixture_projection(seed=index), f"partition-{role}", role)
        for index, role in enumerate(roles, start=1)
    ]


class CohortPartitionTests(unittest.TestCase):
    def test_round_trip_write_load_and_permutation_have_deterministic_identity(self) -> None:
        training_one = fixture_projection(seed=1, template="nonheld-template")
        training_two = fixture_projection(seed=2, template="nonheld-template")
        calibration = fixture_projection(seed=3, template="nonheld-template")
        model_selection = fixture_projection(seed=4, template="heldout-template")
        final_evaluation = fixture_projection(seed=5, template="heldout-template")
        entries = [
            entry(training_one, "partition-training", "training"),
            entry(training_two, "partition-training", "training"),
            entry(calibration, "partition-calibration", "calibration"),
            entry(model_selection, "partition-model-selection", "model_selection"),
            entry(final_evaluation, "partition-final-evaluation", "final_evaluation"),
        ]
        records = [
            provenance(
                training_one,
                kind="replay",
                artifact_identity="artifact-v1:training",
                sources=[
                    str(training_two["scenario_lineage_identity"]),
                    str(training_one["scenario_lineage_identity"]),
                ],
            ),
            provenance(calibration, artifact_identity="artifact-v1:calibration"),
            provenance(model_selection, artifact_identity="artifact-v1:model-selection"),
            provenance(final_evaluation, artifact_identity="artifact-v1:final-evaluation"),
        ]
        canonical_records = copy.deepcopy(records)
        canonical_records[0]["source_scenario_lineage_identities"].reverse()
        first = create_cohort_partition_manifest(
            partition_version=1,
            split_regime="template_held_out",
            held_out_roles=["final_evaluation", "model_selection"],
            entries=list(reversed(entries)),
            provenance_records=list(reversed(records)),
        )
        second = create_cohort_partition_manifest(
            partition_version=1,
            split_regime="template_held_out",
            held_out_roles=["model_selection", "final_evaluation"],
            entries=entries,
            provenance_records=canonical_records,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.identity, second.identity)
        self.assertTrue(
            first.identity.startswith(
                "cohort-partition-manifest-v1:1:template_held_out:"
            )
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        with self.assertRaises(TypeError):
            first.entries[0].scenario_manifest_projection["scenario_lineage_identity"] = "changed"
        with self.assertRaises(TypeError):
            first.entries[0].scenario_manifest_projection["scenario_manifest"]["schema"] = "changed"

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cohort-partition.json"
            write_cohort_partition_manifest(first, path)
            original_bytes = path.read_bytes()
            loaded = load_cohort_partition_manifest(path)
            self.assertEqual(loaded, first)
            write_cohort_partition_manifest(loaded, path)
            self.assertEqual(path.read_bytes(), original_bytes)

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["identity"] = ""
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity must be a nonempty string"):
                load_cohort_partition_manifest(path)

    def test_exact_keys_are_fail_closed_at_each_public_boundary(self) -> None:
        manifest = create_cohort_partition_manifest(
            partition_version=1,
            split_regime="instance_held_out",
            held_out_roles=[],
            entries=[entry(fixture_projection(), "partition-a", "training")],
            provenance_records=[],
        )
        cases = []
        top = manifest.to_dict()
        top["unknown"] = True
        cases.append(top)
        missing_entry_key = manifest.to_dict()
        del missing_entry_key["entries"][0]["dataset_partition"]
        cases.append(missing_entry_key)
        entry_with_unknown = manifest.to_dict()
        entry_with_unknown["entries"][0]["unknown"] = True
        cases.append(entry_with_unknown)
        provenance_with_unknown = manifest.to_dict()
        provenance_with_unknown["provenance_records"] = [
            provenance(fixture_projection())
        ]
        provenance_with_unknown["provenance_records"][0]["unknown"] = True
        cases.append(provenance_with_unknown)
        projection_with_unknown = manifest.to_dict()
        projection_with_unknown["entries"][0]["scenario_manifest"]["unknown"] = True
        cases.append(projection_with_unknown)

        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, "incomplete or contains unknown"):
                    CohortPartitionManifest.from_dict(payload)

    def test_all_roles_are_accepted_and_invalid_role_is_rejected(self) -> None:
        manifest = create_cohort_partition_manifest(
            partition_version=1,
            split_regime="instance_held_out",
            held_out_roles=[],
            entries=four_role_entries(),
            provenance_records=[],
        )
        self.assertEqual(
            {item.exposure_role for item in manifest.entries},
            {"training", "calibration", "model_selection", "final_evaluation"},
        )

        invalid = [entry(fixture_projection(), "partition-a", "research")]
        with self.assertRaisesRegex(ValueError, "exposure role"):
            create_cohort_partition_manifest(
                partition_version=1,
                split_regime="instance_held_out",
                held_out_roles=[],
                entries=invalid,
                provenance_records=[],
            )

    def test_lineage_partition_role_and_level_instance_conflicts_are_rejected(self) -> None:
        projection = fixture_projection()
        with self.assertRaisesRegex(ValueError, "scenario lineage identities must be unique"):
            create_cohort_partition_manifest(
                partition_version=1,
                split_regime="instance_held_out",
                held_out_roles=[],
                entries=[entry(projection, "partition-a", "training"), entry(projection, "partition-a", "training")],
                provenance_records=[],
            )
        with self.assertRaisesRegex(ValueError, "scenario lineage identities must be unique"):
            create_cohort_partition_manifest(
                partition_version=1,
                split_regime="instance_held_out",
                held_out_roles=[],
                entries=[
                    entry(projection, "partition-a", "training"),
                    entry(projection, "partition-b", "final_evaluation"),
                ],
                provenance_records=[],
            )

        with self.assertRaisesRegex(ValueError, "dataset partition must map to one exposure role"):
            create_cohort_partition_manifest(
                partition_version=1,
                split_regime="instance_held_out",
                held_out_roles=[],
                entries=[
                    entry(fixture_projection(seed=1), "partition-a", "training"),
                    entry(fixture_projection(seed=2), "partition-a", "calibration"),
                ],
                provenance_records=[],
            )

        first_level = fixture_projection(seed=1)
        second_lineage_same_level = fixture_projection(seed=2)
        second_lineage_same_level["level_instance_identity"] = first_level[
            "level_instance_identity"
        ]
        second_lineage_same_level["scenario_manifest"]["level_instance"][
            "identity"
        ] = first_level["level_instance_identity"]
        with self.assertRaisesRegex(ValueError, "level instance must not map"):
            create_cohort_partition_manifest(
                partition_version=1,
                split_regime="instance_held_out",
                held_out_roles=[],
                entries=[
                    entry(first_level, "partition-a", "training"),
                    entry(
                        second_lineage_same_level,
                        "partition-b",
                        "final_evaluation",
                    ),
                ],
                provenance_records=[],
            )

    def test_instance_held_out_allows_template_sharing(self) -> None:
        manifest = create_cohort_partition_manifest(
            partition_version=1,
            split_regime="instance_held_out",
            held_out_roles=[],
            entries=[
                entry(fixture_projection(seed=1), "partition-a", "training"),
                entry(fixture_projection(seed=2), "partition-b", "final_evaluation"),
            ],
            provenance_records=[],
        )
        self.assertEqual(len(manifest.entries), 2)

    def test_template_held_out_boundary_and_within_side_sharing(self) -> None:
        leaked = [
            entry(fixture_projection(seed=1, template="shared-template"), "partition-a", "training"),
            entry(fixture_projection(seed=2, template="shared-template"), "partition-b", "final_evaluation"),
        ]
        with self.assertRaisesRegex(ValueError, "template-held-out boundary"):
            create_cohort_partition_manifest(
                partition_version=1,
                split_regime="template_held_out",
                held_out_roles=["final_evaluation"],
                entries=leaked,
                provenance_records=[],
            )

        manifest = create_cohort_partition_manifest(
            partition_version=1,
            split_regime="template_held_out",
            held_out_roles=["final_evaluation"],
            entries=[
                entry(fixture_projection(seed=1, template="training-template"), "partition-a", "training"),
                entry(fixture_projection(seed=2, template="training-template"), "partition-b", "calibration"),
                entry(fixture_projection(seed=3, template="heldout-template"), "partition-c", "final_evaluation"),
                entry(fixture_projection(seed=4, template="heldout-template"), "partition-d", "final_evaluation"),
            ],
            provenance_records=[],
        )
        self.assertEqual(manifest.held_out_roles, ("final_evaluation",))

    def test_template_held_out_requires_nonempty_proper_roles_and_available_templates(self) -> None:
        entries = [entry(fixture_projection(), "partition-a", "training")]
        with self.assertRaisesRegex(ValueError, "split regime"):
            create_cohort_partition_manifest(
                partition_version=1,
                split_regime="rollout_held_out",
                held_out_roles=[],
                entries=entries,
                provenance_records=[],
            )
        with self.assertRaisesRegex(ValueError, "instance_held_out requires held_out_roles to be empty"):
            create_cohort_partition_manifest(
                partition_version=1,
                split_regime="instance_held_out",
                held_out_roles=["final_evaluation"],
                entries=entries,
                provenance_records=[],
            )
        for held_out_roles, message in (([], "nonempty proper subset"), (["training", "calibration", "model_selection", "final_evaluation"], "nonempty proper subset")):
            with self.subTest(held_out_roles=held_out_roles), self.assertRaisesRegex(ValueError, message):
                create_cohort_partition_manifest(
                    partition_version=1,
                    split_regime="template_held_out",
                    held_out_roles=held_out_roles,
                    entries=entries,
                    provenance_records=[],
                )

        with self.assertRaisesRegex(ValueError, "available template"):
            create_cohort_partition_manifest(
                partition_version=1,
                split_regime="template_held_out",
                held_out_roles=["final_evaluation"],
                entries=[entry(unavailable_template_projection(), "partition-a", "training")],
                provenance_records=[],
            )

    def test_provenance_rules_apply_to_derivation_and_replay_records(self) -> None:
        training_a = fixture_projection(seed=1)
        training_b = fixture_projection(seed=2)
        final = fixture_projection(seed=3)
        entries = [
            entry(training_a, "partition-a", "training"),
            entry(training_b, "partition-b", "training"),
            entry(final, "partition-c", "final_evaluation"),
        ]
        line_a = str(training_a["scenario_lineage_identity"])
        line_b = str(training_b["scenario_lineage_identity"])
        line_final = str(final["scenario_lineage_identity"])

        for invalid_kind in ("derived_artifact", "unknown_artifact"):
            with self.subTest(invalid_kind=invalid_kind), self.assertRaisesRegex(ValueError, "artifact kind"):
                create_cohort_partition_manifest(
                    partition_version=1,
                    split_regime="instance_held_out",
                    held_out_roles=[],
                    entries=entries,
                    provenance_records=[provenance(training_a, kind=invalid_kind)],
                )

        for kind in ("derivation_artifact", "replay"):
            valid = create_cohort_partition_manifest(
                partition_version=1,
                split_regime="instance_held_out",
                held_out_roles=[],
                entries=entries,
                provenance_records=[provenance(training_a, kind=kind, sources=[line_a])],
            )
            self.assertEqual(valid.provenance_records[0].source_scenario_lineage_identities, (line_a,))

            cases = (
                (provenance(training_a, kind=kind, consumer="unknown", sources=[line_a]), "consumer"),
                (provenance(training_a, kind=kind, sources=["unknown"]), "source"),
                (provenance(training_a, kind=kind, artifact_identity=""), "nonempty"),
                (provenance(training_a, kind=kind, sources=[]), "nonempty"),
                (provenance(training_a, kind=kind, sources=[line_a, line_a]), "unique"),
                (provenance(training_a, kind=kind, sources=[line_final]), "same dataset partition and exposure role"),
                (provenance(training_a, kind=kind, sources=[line_b]), "same dataset partition and exposure role"),
            )
            for record, message in cases:
                with self.subTest(kind=kind, message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        create_cohort_partition_manifest(
                            partition_version=1,
                            split_regime="instance_held_out",
                            held_out_roles=[],
                            entries=entries,
                            provenance_records=[record],
                        )

            duplicate = [
                provenance(training_a, kind=kind, artifact_identity="duplicate", sources=[line_a]),
                provenance(training_b, kind=kind, artifact_identity="duplicate", sources=[line_b]),
            ]
            with self.assertRaisesRegex(ValueError, "artifact identities must be unique"):
                create_cohort_partition_manifest(
                    partition_version=1,
                    split_regime="instance_held_out",
                    held_out_roles=[],
                    entries=entries,
                    provenance_records=duplicate,
                )

    def test_audit_requires_complete_admitted_lineage_and_provenance_inventories(self) -> None:
        training_a = fixture_projection(seed=1)
        training_b = fixture_projection(seed=2)
        line_a = str(training_a["scenario_lineage_identity"])
        line_b = str(training_b["scenario_lineage_identity"])
        entries = [
            entry(training_a, "partition-training", "training"),
            entry(training_b, "partition-training", "training"),
        ]
        records = [
            provenance(
                training_a,
                artifact_identity="artifact-v1:all-training",
                sources=[line_a, line_b],
            )
        ]
        manifest = create_cohort_partition_manifest(
            partition_version=1,
            split_regime="instance_held_out",
            held_out_roles=[],
            entries=entries,
            provenance_records=records,
        )

        audit_cohort_partition_manifest(
            manifest,
            admitted_scenario_lineage_identities=[line_b, line_a],
            admitted_provenance_records=[
                provenance(
                    training_a,
                    artifact_identity="artifact-v1:all-training",
                    sources=[line_b, line_a],
                )
            ],
        )

        for inventory, message in (
            ([], "nonempty"),
            ([line_a], "lineage inventory mismatch"),
            ([line_a, line_b, "extra"], "lineage inventory mismatch"),
            ([line_a, line_a, line_b], "unique"),
        ):
            with self.subTest(lineage_inventory=inventory), self.assertRaisesRegex(ValueError, message):
                audit_cohort_partition_manifest(
                    manifest,
                    admitted_scenario_lineage_identities=inventory,
                    admitted_provenance_records=records,
                )

        provenance_cases = (
            ([], "provenance inventory mismatch"),
            (
                records
                + [provenance(training_b, artifact_identity="artifact-v1:extra", sources=[line_b])],
                "provenance inventory mismatch",
            ),
            (
                [provenance(training_a, artifact_identity="artifact-v1:all-training", sources=[line_a])],
                "provenance inventory mismatch",
            ),
        )
        for admitted_records, message in provenance_cases:
            with self.subTest(admitted_records=admitted_records), self.assertRaisesRegex(ValueError, message):
                audit_cohort_partition_manifest(
                    manifest,
                    admitted_scenario_lineage_identities=[line_a, line_b],
                    admitted_provenance_records=admitted_records,
                )

        with self.assertRaisesRegex(ValueError, "artifact kind"):
            audit_cohort_partition_manifest(
                manifest,
                admitted_scenario_lineage_identities=[line_a, line_b],
                admitted_provenance_records=[provenance(training_a, kind="derived_artifact")],
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            audit_cohort_partition_manifest(
                manifest,
                admitted_scenario_lineage_identities=[line_a, line_b],
                admitted_provenance_records=records
                + [provenance(training_b, artifact_identity="artifact-v1:all-training", sources=[line_b])],
            )

    def test_smoke_only_projection_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "smoke_only"):
            create_cohort_partition_manifest(
                partition_version=1,
                split_regime="instance_held_out",
                held_out_roles=[],
                entries=[entry(fixture_projection(eligibility=SMOKE_ONLY), "partition-a", "training")],
                provenance_records=[],
            )


if __name__ == "__main__":
    unittest.main()
