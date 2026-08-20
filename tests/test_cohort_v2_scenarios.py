from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path

from scripts.cohort_v2_scenarios import (
    ScenarioLineageError,
    create_scenario_template_constraints,
    create_central_v2_scenario_inventory_draft,
    create_changed_declared_input_receipt,
    create_cohort_v2_scenario_manifest,
    create_identical_input_reproduction_receipt,
    create_unity_reset_reproduction_receipt,
    create_reviewed_central_v2_scenario_inventory,
    create_scenario_inventory_entry,
    create_scenario_template_record,
    load_cohort_v2_scenario_manifest,
    load_scenario_template_record,
    materialize_template_bound_level_instance,
    validate_central_v2_scenario_inventory,
    validate_central_v2_scenario_inventory_draft,
    validate_deterministic_scenario_receipt,
    validate_scenario_template_constraints_workbook,
    write_cohort_v2_scenario_manifest,
    write_scenario_template_record,
)
from scripts.scenario_manifest import BenchmarkCondition, ELIGIBLE, create_generated_manifest, import_legacy_manifest
from tasks.task_generator.canonical_materialization import CanonicalMaterializationRequest


TEMPLATE_A = b'''<?xml version="1.0" encoding="utf-8"?>
<Level width="2"><Camera maxWidth="30" minWidth="20"/><Birds><Bird type="BirdRed"/></Birds><Slingshot x="-8" y="-2"/><GameObjects><Pig type="BasicSmall" x="1" y="-3" rotation="0"/><Block type="SquareSmall" material="wood" x="2" y="-3" rotation="0"/></GameObjects></Level>\n'''
TEMPLATE_B = TEMPLATE_A.replace(b'x="2"', b'x="3"')
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAINING_TEMPLATE_PATH = REPOSITORY_ROOT / "tasks/task_templates/novelty_level_0/type010101/Levels/00001_0_1_010101_0_1.xml"
CALIBRATION_TEMPLATE_PATH = REPOSITORY_ROOT / "tasks/task_templates/novelty_level_0/type010102/Levels/00001_0_1_010102_0_2.xml"
CONSTRAINTS_WORKBOOK_PATH = REPOSITORY_ROOT / "tasks/task_generator/template_constraints.xlsx"


class CohortV2ScenarioTests(unittest.TestCase):
    def _record(self, content: bytes, name: str):
        return create_scenario_template_record(
            content,
            source_reference=f"tasks/task_templates/{name}.xml",
            benchmark_conditions=[BenchmarkCondition("novelty_level_1", "type0101")],
        )

    def _scenario(self, record, *, seed: int, layout_choice: int):
        xml = TEMPLATE_A.replace(b'width="2"', f'width="{seed + 2}"'.encode("ascii"))
        manifest = create_generated_manifest(
            xml,
            benchmark_condition=BenchmarkCondition("novelty_level_1", "type0101"),
            template_identity=record.identity,
            generator_identity="novphy-task-generator",
            generator_version="canonical_materialization_v1",
            generation_seed=seed,
            declared_inputs={
                "template_content_identity": record.source_content_identity,
                "layout_choice": layout_choice,
            },
            parameter_realization={"layout_choice": layout_choice},
            eligibility=ELIGIBLE,
        )
        return create_cohort_v2_scenario_manifest(record, manifest, xml_content=xml)

    def test_real_training_materialization_binds_reviewed_source_and_workbook_row(self) -> None:
        constraints = create_scenario_template_constraints(
            CONSTRAINTS_WORKBOOK_PATH.read_bytes(),
            source_reference="tasks/task_generator/template_constraints.xlsx",
            sheet_name="Task Variations",
            row_number=3,
            canonical_generator_template_name="0_1_010101_0_1",
            reference_point=(1.00798, -2.1274),
            min_coordinate=(-7.88, -2.39049),
            max_coordinate=(1.229969, 1.809741),
        )
        record = create_scenario_template_record(
            TRAINING_TEMPLATE_PATH.read_bytes(),
            source_reference="tasks/task_templates/novelty_level_0/type010101/Levels/00001_0_1_010101_0_1.xml",
            benchmark_conditions=[BenchmarkCondition("novelty_level_0", "type010101")],
            generation_constraints=constraints,
        )

        self.assertEqual(
            record.source_content_identity,
            "xml_bytes_v1:sha256:ede3a16435534ebbb3b37dccfeb7652ce2a56c50a1f5b826abede79692ba9220",
        )
        self.assertEqual(
            constraints.source_content_identity,
            "xlsx_bytes_v1:sha256:4635c9c072eb36481815ac17cfd1cdd958773fc400d99ac579e0561ba045bff1",
        )
        self.assertEqual(constraints.sheet_name, "Task Variations")
        self.assertEqual(constraints.row_number, 3)
        self.assertEqual(type(record).from_dict(record.to_dict()), record)
        with self.assertRaises(ScenarioLineageError) as forged_row:
            validate_scenario_template_constraints_workbook(
                replace(constraints, reference_point=(1.5, -2.1274)),
                CONSTRAINTS_WORKBOOK_PATH,
            )
        self.assertEqual(forged_row.exception.reason, "unresolved_source_provenance")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = CanonicalMaterializationRequest(
                template_path=TRAINING_TEMPLATE_PATH,
                output_xml_path=root / "training.xml",
                output_manifest_path=root / "training.scenario.json",
                template_name=constraints.canonical_generator_template_name,
                benchmark_condition=BenchmarkCondition("novelty_level_0", "type010101"),
                template_identity=record.identity,
                generation_seed=4401,
                reference_point=constraints.reference_point,
                min_coordinate=constraints.min_coordinate,
                max_coordinate=constraints.max_coordinate,
                restricted_objects=(),
            )
            first, first_scenario = materialize_template_bound_level_instance(
                request,
                record,
                constraints_workbook_path=CONSTRAINTS_WORKBOOK_PATH,
                publish=False,
            )
            second, second_scenario = materialize_template_bound_level_instance(
                request,
                record,
                constraints_workbook_path=CONSTRAINTS_WORKBOOK_PATH,
                publish=False,
            )

            self.assertEqual(first.xml_content, second.xml_content)
            self.assertEqual(first_scenario.identity, second_scenario.identity)
            self.assertEqual(first.manifest.generation.generation_seed, 4401)

            with self.assertRaises(ScenarioLineageError) as missing_identity:
                materialize_template_bound_level_instance(
                    replace(request, template_identity=""),
                    record,
                    constraints_workbook_path=CONSTRAINTS_WORKBOOK_PATH,
                    publish=False,
                )
            self.assertEqual(missing_identity.exception.reason, "missing_template_identity")

            with self.assertRaises(ScenarioLineageError) as unresolved:
                materialize_template_bound_level_instance(request, record, publish=False)
            self.assertEqual(unresolved.exception.reason, "unresolved_source_provenance")

            changed_workbook = root / "changed.xlsx"
            changed_workbook.write_bytes(CONSTRAINTS_WORKBOOK_PATH.read_bytes() + b"drift")
            with self.assertRaises(ScenarioLineageError) as drift:
                materialize_template_bound_level_instance(
                    request,
                    record,
                    constraints_workbook_path=changed_workbook,
                    publish=False,
                )
            self.assertEqual(drift.exception.reason, "content_drift")

    def test_real_calibration_materialization_binds_utf16_source_and_workbook_row(self) -> None:
        source_content = CALIBRATION_TEMPLATE_PATH.read_bytes()
        self.assertTrue(source_content.startswith(b"\xff\xfe"))
        constraints = create_scenario_template_constraints(
            CONSTRAINTS_WORKBOOK_PATH.read_bytes(),
            source_reference="tasks/task_generator/template_constraints.xlsx",
            sheet_name="Task Variations",
            row_number=4,
            canonical_generator_template_name="0_1_010102_0_2",
            reference_point=(1.02408, -1.84657),
            min_coordinate=(-7.235919, -1.95804),
            max_coordinate=(1.444081, 1.53147),
        )
        record = create_scenario_template_record(
            source_content,
            source_reference="tasks/task_templates/novelty_level_0/type010102/Levels/00001_0_1_010102_0_2.xml",
            benchmark_conditions=[BenchmarkCondition("novelty_level_0", "type010102")],
            generation_constraints=constraints,
        )

        self.assertEqual(
            record.source_content_identity,
            "xml_bytes_v1:sha256:36627b24c7c3b84799f876da79dc6cd554c0fa8fd2b4c0f3c5afb2a049376b37",
        )
        self.assertEqual(
            constraints.source_content_identity,
            "xlsx_bytes_v1:sha256:4635c9c072eb36481815ac17cfd1cdd958773fc400d99ac579e0561ba045bff1",
        )
        self.assertEqual(constraints.sheet_name, "Task Variations")
        self.assertEqual(constraints.row_number, 4)
        self.assertEqual(type(record).from_dict(record.to_dict()), record)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = CanonicalMaterializationRequest(
                template_path=CALIBRATION_TEMPLATE_PATH,
                output_xml_path=root / "calibration.xml",
                output_manifest_path=root / "calibration.scenario.json",
                template_name=constraints.canonical_generator_template_name,
                benchmark_condition=BenchmarkCondition("novelty_level_0", "type010102"),
                template_identity=record.identity,
                generation_seed=4501,
                reference_point=constraints.reference_point,
                min_coordinate=constraints.min_coordinate,
                max_coordinate=constraints.max_coordinate,
                restricted_objects=(),
            )
            first, first_scenario = materialize_template_bound_level_instance(
                request,
                record,
                constraints_workbook_path=CONSTRAINTS_WORKBOOK_PATH,
                publish=False,
            )
            second, second_scenario = materialize_template_bound_level_instance(
                request,
                record,
                constraints_workbook_path=CONSTRAINTS_WORKBOOK_PATH,
                publish=False,
            )

            self.assertEqual(first.xml_content, second.xml_content)
            self.assertEqual(first_scenario.identity, second_scenario.identity)
            self.assertEqual(first.manifest.generation.generation_seed, 4501)
            self.assertEqual(
                first.manifest.generation.declared_inputs["template_name"],
                "0_1_010102_0_2",
            )

            with self.assertRaises(ScenarioLineageError) as unresolved:
                materialize_template_bound_level_instance(
                    replace(request, template_name="0_1_010101_0_1"),
                    record,
                    constraints_workbook_path=CONSTRAINTS_WORKBOOK_PATH,
                    publish=False,
                )
            self.assertEqual(unresolved.exception.reason, "unresolved_source_provenance")

            changed_source = root / "changed.xml"
            changed_source.write_bytes(source_content + b"drift")
            with self.assertRaises(ScenarioLineageError) as drift:
                materialize_template_bound_level_instance(
                    replace(request, template_path=changed_source),
                    record,
                    constraints_workbook_path=CONSTRAINTS_WORKBOOK_PATH,
                    publish=False,
                )
            self.assertEqual(drift.exception.reason, "content_drift")

    def test_model_selection_seed_change_produces_distinct_lineage_receipt(self) -> None:
        source_content = TRAINING_TEMPLATE_PATH.read_bytes()
        constraints = create_scenario_template_constraints(
            CONSTRAINTS_WORKBOOK_PATH.read_bytes(),
            source_reference="tasks/task_generator/template_constraints.xlsx",
            sheet_name="Task Variations",
            row_number=3,
            canonical_generator_template_name="0_1_010101_0_1",
            reference_point=(1.00798, -2.1274),
            min_coordinate=(-7.88, -2.39049),
            max_coordinate=(1.229969, 1.809741),
        )
        record = create_scenario_template_record(
            source_content,
            source_reference="tasks/task_templates/novelty_level_0/type010101/Levels/00001_0_1_010101_0_1.xml",
            benchmark_conditions=[BenchmarkCondition("novelty_level_0", "type010101")],
            generation_constraints=constraints,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_request = CanonicalMaterializationRequest(
                template_path=TRAINING_TEMPLATE_PATH,
                output_xml_path=root / "training.xml",
                output_manifest_path=root / "training.scenario.json",
                template_name=constraints.canonical_generator_template_name,
                benchmark_condition=BenchmarkCondition("novelty_level_0", "type010101"),
                template_identity=record.identity,
                generation_seed=4401,
                reference_point=constraints.reference_point,
                min_coordinate=constraints.min_coordinate,
                max_coordinate=constraints.max_coordinate,
                restricted_objects=(),
            )
            model_selection_request = replace(
                training_request,
                output_xml_path=root / "model-selection.xml",
                output_manifest_path=root / "model-selection.scenario.json",
                generation_seed=4402,
            )
            _, training = materialize_template_bound_level_instance(
                training_request,
                record,
                constraints_workbook_path=CONSTRAINTS_WORKBOOK_PATH,
                publish=False,
            )
            _, model_selection = materialize_template_bound_level_instance(
                model_selection_request,
                record,
                constraints_workbook_path=CONSTRAINTS_WORKBOOK_PATH,
                publish=False,
            )

            receipt = create_changed_declared_input_receipt(
                training,
                model_selection,
                input_key="generation_seed",
            )
            self.assertEqual(validate_deterministic_scenario_receipt(receipt), receipt)
            self.assertEqual(receipt["original_value"], 4401)
            self.assertEqual(receipt["changed_value"], 4402)
            self.assertEqual(training.template_record, model_selection.template_record)
            self.assertEqual(
                training.template_record.generation_constraints,
                model_selection.template_record.generation_constraints,
            )
            self.assertNotEqual(
                training.scenario_manifest.scenario_specification.identity,
                model_selection.scenario_manifest.scenario_specification.identity,
            )
            self.assertNotEqual(
                training.scenario_manifest.scenario_lineage.identity,
                model_selection.scenario_manifest.scenario_lineage.identity,
            )

            reused_identity_manifest = replace(
                model_selection.scenario_manifest,
                scenario_specification=training.scenario_manifest.scenario_specification,
                scenario_lineage=training.scenario_manifest.scenario_lineage,
            )
            with self.assertRaises(ScenarioLineageError) as reused:
                create_changed_declared_input_receipt(
                    training,
                    replace(model_selection, scenario_manifest=reused_identity_manifest),
                    input_key="generation_seed",
                )
            self.assertEqual(reused.exception.reason, "cross_lineage_reuse")

            changed_source = root / "changed-template.xml"
            changed_source_content = source_content + b"\n"
            changed_source.write_bytes(changed_source_content)
            changed_record = create_scenario_template_record(
                changed_source_content,
                source_reference=record.source_reference,
                benchmark_conditions=record.benchmark_conditions,
                generation_constraints=constraints,
            )
            _, changed_template_scenario = materialize_template_bound_level_instance(
                replace(
                    model_selection_request,
                    template_path=changed_source,
                    template_identity=changed_record.identity,
                ),
                changed_record,
                constraints_workbook_path=CONSTRAINTS_WORKBOOK_PATH,
                publish=False,
            )
            with self.assertRaises(ScenarioLineageError) as drift:
                create_changed_declared_input_receipt(
                    training,
                    changed_template_scenario,
                    input_key="generation_seed",
                )
            self.assertEqual(drift.exception.reason, "content_drift")

    def test_template_record_and_v2_manifest_bind_exact_sources_without_changing_v1(self) -> None:
        record = self._record(TEMPLATE_A, "template-a")
        scenario = self._scenario(record, seed=1, layout_choice=1)

        malformed_record = record.to_dict()
        malformed_record["source_content_identity"] = "xml_bytes_v1:sha256:not-a-digest"
        with self.assertRaisesRegex(ValueError, "source_content_identity"):
            type(record).from_dict(malformed_record)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template_path = root / "template-a.xml"
            record_path = root / "template-a.record.json"
            xml_path = root / "level.xml"
            scenario_path = root / "level.cohort-v2-scenario.json"
            template_path.write_bytes(TEMPLATE_A)
            xml_path.write_bytes(TEMPLATE_A.replace(b'width="2"', b'width="3"'))
            write_scenario_template_record(record, record_path)
            write_cohort_v2_scenario_manifest(scenario, scenario_path)

            self.assertEqual(load_scenario_template_record(record_path, source_path=template_path), record)
            self.assertEqual(
                load_cohort_v2_scenario_manifest(
                    scenario_path,
                    xml_path=xml_path,
                    template_source_path=template_path,
                ),
                scenario,
            )
            template_path.write_bytes(TEMPLATE_B)
            with self.assertRaisesRegex(ValueError, "content identity"):
                load_scenario_template_record(record_path, source_path=template_path)

    def test_receipts_prove_declared_reproduction_and_changed_input_divergence(self) -> None:
        record = self._record(TEMPLATE_A, "template-a")
        original = self._scenario(record, seed=1, layout_choice=1)
        reproduction = self._scenario(record, seed=1, layout_choice=1)
        changed = self._scenario(record, seed=2, layout_choice=2)

        identical = create_identical_input_reproduction_receipt(original, reproduction)
        changed_input = create_changed_declared_input_receipt(
            original, changed, input_key="layout_choice"
        )
        self.assertEqual(validate_deterministic_scenario_receipt(identical), identical)
        self.assertEqual(validate_deterministic_scenario_receipt(changed_input), changed_input)
        self.assertNotEqual(
            original.scenario_manifest.scenario_lineage.identity,
            changed.scenario_manifest.scenario_lineage.identity,
        )

        malformed = deepcopy(identical)
        malformed["declared_initial_engine_state_identity"] = "inferred-from-filename"
        with self.assertRaisesRegex(ValueError, "identity is stale"):
            validate_deterministic_scenario_receipt(malformed)

    def test_unity_reset_receipt_requires_identical_normalized_initial_state(self) -> None:
        record = self._record(TEMPLATE_A, "template-a")
        scenario = self._scenario(record, seed=1, layout_choice=1)
        receipt = create_unity_reset_reproduction_receipt(
            scenario,
            first_capture_sha256="a" * 64,
            second_capture_sha256="b" * 64,
            first_initial_engine_state_identity="normalized-initial-engine-state-v1:sha256:" + "c" * 64,
            second_initial_engine_state_identity="normalized-initial-engine-state-v1:sha256:" + "c" * 64,
        )
        self.assertEqual(validate_deterministic_scenario_receipt(receipt), receipt)
        with self.assertRaisesRegex(ScenarioLineageError, "initial_state_mismatch"):
            create_unity_reset_reproduction_receipt(
                scenario,
                first_capture_sha256="a" * 64,
                second_capture_sha256="b" * 64,
                first_initial_engine_state_identity="normalized-initial-engine-state-v1:sha256:" + "c" * 64,
                second_initial_engine_state_identity="normalized-initial-engine-state-v1:sha256:" + "d" * 64,
            )

    def test_legacy_static_requires_its_actual_record_and_smoke_only_is_rejected(self) -> None:
        record = create_scenario_template_record(
            TEMPLATE_A,
            source_reference="legacy/level.xml",
            benchmark_conditions=[BenchmarkCondition("novelty_level_1", "type0101")],
        )
        imported = import_legacy_manifest(
            TEMPLATE_A,
            benchmark_condition=BenchmarkCondition("novelty_level_1", "type0101"),
            source_path="legacy/level.xml",
            eligibility=ELIGIBLE,
        )
        bound = create_cohort_v2_scenario_manifest(record, imported, xml_content=TEMPLATE_A)
        self.assertEqual(bound.scenario_manifest.generation.mode, "legacy_static")

        smoke = import_legacy_manifest(
            TEMPLATE_A,
            benchmark_condition=BenchmarkCondition("novelty_level_1", "type0101"),
            source_path="legacy/level.xml",
            eligibility="smoke_only",
            eligibility_reason="staged source",
        )
        with self.assertRaises(ScenarioLineageError) as smoke_only:
            create_cohort_v2_scenario_manifest(record, smoke, xml_content=TEMPLATE_A)
        self.assertEqual(smoke_only.exception.reason, "smoke_only")

    def test_template_bound_materialization_rejects_unresolved_source_provenance(self) -> None:
        record = self._record(TEMPLATE_A, "template-a")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template_path = root / "template.xml"
            template_path.write_bytes(TEMPLATE_A)
            request = CanonicalMaterializationRequest(
                template_path=template_path,
                output_xml_path=root / "level.xml",
                output_manifest_path=root / "level.scenario.json",
                template_name="0_1_0101_1_5",
                benchmark_condition=BenchmarkCondition("novelty_level_1", "type0101"),
                template_identity=record.identity,
                generation_seed=1729,
                reference_point=(1.0, -3.0),
                min_coordinate=(-1.0, -3.0),
                max_coordinate=(1.0, -1.0),
                restricted_objects=(),
            )
            materialized, scenario = materialize_template_bound_level_instance(request, record, publish=False)
            self.assertEqual(materialized.manifest.scenario_template.identity, record.identity)
            self.assertEqual(scenario.template_record, record)
            template_path.write_bytes(TEMPLATE_B)
            with self.assertRaisesRegex(ValueError, "content identity"):
                materialize_template_bound_level_instance(request, record, publish=False)

    def test_inventory_resolves_nonfinal_manifests_and_keeps_final_manifest_sealed(self) -> None:
        family_a_constraints = create_scenario_template_constraints(
            CONSTRAINTS_WORKBOOK_PATH.read_bytes(),
            source_reference="tasks/task_generator/template_constraints.xlsx",
            sheet_name="Task Variations",
            row_number=3,
            canonical_generator_template_name="0_1_010101_0_1",
            reference_point=(1.00798, -2.1274),
            min_coordinate=(-7.88, -2.39049),
            max_coordinate=(1.229969, 1.809741),
        )
        family_b_constraints = create_scenario_template_constraints(
            CONSTRAINTS_WORKBOOK_PATH.read_bytes(),
            source_reference="tasks/task_generator/template_constraints.xlsx",
            sheet_name="Task Variations",
            row_number=4,
            canonical_generator_template_name="0_1_010102_0_2",
            reference_point=(1.02408, -1.84657),
            min_coordinate=(-7.235919, -1.95804),
            max_coordinate=(1.444081, 1.53147),
        )
        family_a_record = create_scenario_template_record(
            TRAINING_TEMPLATE_PATH.read_bytes(),
            source_reference="tasks/task_templates/novelty_level_0/type010101/Levels/00001_0_1_010101_0_1.xml",
            benchmark_conditions=[BenchmarkCondition("novelty_level_0", "type010101")],
            generation_constraints=family_a_constraints,
        )
        family_b_record = create_scenario_template_record(
            CALIBRATION_TEMPLATE_PATH.read_bytes(),
            source_reference="tasks/task_templates/novelty_level_0/type010102/Levels/00001_0_1_010102_0_2.xml",
            benchmark_conditions=[BenchmarkCondition("novelty_level_0", "type010102")],
            generation_constraints=family_b_constraints,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def materialize(role: str, seed: int, template_path: Path, record, constraints):
                request = CanonicalMaterializationRequest(
                    template_path=template_path,
                    output_xml_path=root / f"{role}.xml",
                    output_manifest_path=root / f"{role}.scenario.json",
                    template_name=constraints.canonical_generator_template_name,
                    benchmark_condition=record.benchmark_conditions[0],
                    template_identity=record.identity,
                    generation_seed=seed,
                    reference_point=constraints.reference_point,
                    min_coordinate=constraints.min_coordinate,
                    max_coordinate=constraints.max_coordinate,
                    restricted_objects=(),
                )
                return materialize_template_bound_level_instance(
                    request,
                    record,
                    constraints_workbook_path=CONSTRAINTS_WORKBOOK_PATH,
                    publish=False,
                )[1]

            scenarios = {
                "training": materialize("training", 4401, TRAINING_TEMPLATE_PATH, family_a_record, family_a_constraints),
                "calibration": materialize("calibration", 4501, CALIBRATION_TEMPLATE_PATH, family_b_record, family_b_constraints),
                "model_selection": materialize("model-selection", 4402, TRAINING_TEMPLATE_PATH, family_a_record, family_a_constraints),
                "final_evaluation": materialize("final-evaluation", 4502, CALIBRATION_TEMPLATE_PATH, family_b_record, family_b_constraints),
            }
            self.assertEqual(
                scenarios["final_evaluation"].scenario_manifest.generation.generation_seed,
                4502,
            )

            manifest_root = root / "manifests"
            references = {
                "training": "training.json",
                "calibration": "calibration.json",
                "model_selection": "model-selection.json",
            }
            for role, reference in references.items():
                write_cohort_v2_scenario_manifest(
                    scenarios[role],
                    manifest_root / reference,
                )

            entries = [
                create_scenario_inventory_entry(
                    role,
                    "planned_non_final",
                    scenarios[role],
                    scenario_manifest_reference=reference,
                )
                for role, reference in references.items()
            ]
            entries.append(create_scenario_inventory_entry(
                "final_evaluation",
                "sealed_final",
                scenarios["final_evaluation"],
                sealed_scenario_manifest_reference="sealed-final-evaluation-v1:issue-45",
            ))
            draft = create_central_v2_scenario_inventory_draft(
                entries,
                manifest_root=manifest_root,
            )
            self.assertEqual(
                draft["identity"],
                "central-v2-scenario-inventory-draft-v1:sha256:993d27c535c100e73209d2d0da33169cb313a5b72d902ee23ad6170bdc481400",
            )
            self.assertEqual(
                validate_central_v2_scenario_inventory_draft(
                    draft,
                    manifest_root=manifest_root,
                ),
                draft,
            )

            final_entry = draft["entries"][3]
            self.assertEqual(set(final_entry), {
                "exposure_role", "inventory_state", "scenario_manifest_identity",
                "scenario_manifest_digest", "benchmark_condition_identity",
                "scenario_template_identity", "level_instance_identity",
                "scenario_specification_identity", "scenario_lineage_identity",
                "declared_initial_engine_state_identity", "sealed_scenario_manifest_reference",
            })
            sealed_projection = json.dumps(final_entry, sort_keys=True)
            self.assertNotIn("generation_seed", sealed_projection)
            self.assertNotIn("declared_inputs", sealed_projection)
            self.assertNotIn("parameter_realization", sealed_projection)
            self.assertNotIn("scenario_manifest_reference", final_entry)

            reviewed = create_reviewed_central_v2_scenario_inventory(
                draft,
                review_author="Sino-Huang",
                review_url="https://github.com/Sino-Huang/NovPhy/issues/45#issuecomment-123456789",
                manifest_root=manifest_root,
            )
            self.assertEqual(reviewed["approved_draft_identity"], draft["identity"])
            self.assertEqual(
                validate_central_v2_scenario_inventory(reviewed, manifest_root=manifest_root),
                reviewed,
            )
            with self.assertRaisesRegex(ValueError, "review authority"):
                create_reviewed_central_v2_scenario_inventory(
                    draft,
                    review_author="",
                    review_url="https://github.com/Sino-Huang/NovPhy/issues/45#issuecomment-123456789",
                    manifest_root=manifest_root,
                )
            wrong_draft = deepcopy(reviewed)
            wrong_draft["approved_draft_identity"] = "forged"
            with self.assertRaisesRegex(ValueError, "approved draft"):
                validate_central_v2_scenario_inventory(
                    wrong_draft,
                    manifest_root=manifest_root,
                )

            initial_state_mismatch = deepcopy(draft)
            initial_state_mismatch["entries"][0]["declared_initial_engine_state_identity"] = "forged"
            with self.assertRaises(ScenarioLineageError) as mismatch:
                validate_central_v2_scenario_inventory_draft(
                    initial_state_mismatch,
                    manifest_root=manifest_root,
                )
            self.assertEqual(mismatch.exception.reason, "initial_state_mismatch")

            conflicting = deepcopy(draft)
            conflicting["entries"][3]["inventory_state"] = "planned_non_final"
            with self.assertRaisesRegex(ValueError, "sealing state"):
                validate_central_v2_scenario_inventory_draft(
                    conflicting,
                    manifest_root=manifest_root,
                )

            reused = list(entries[:3])
            reused.append(replace(
                entries[0],
                exposure_role="final_evaluation",
                inventory_state="sealed_final",
                scenario_manifest_reference=None,
                sealed_scenario_manifest_reference="sealed-final-evaluation-v1:reused",
            ))
            with self.assertRaises(ScenarioLineageError) as cross_lineage:
                create_central_v2_scenario_inventory_draft(
                    reused,
                    manifest_root=manifest_root,
                )
            self.assertEqual(cross_lineage.exception.reason, "cross_lineage_reuse")


if __name__ == "__main__":
    unittest.main()
