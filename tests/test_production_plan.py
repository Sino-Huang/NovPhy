import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import quote

from scripts.collection_plan import (
    create_collection_plan,
    load_collection_plan,
    write_collection_plan,
)
from scripts.physics_macro_labels import (
    Availability,
    DERIVATION_SPEC_VERSION,
    SemanticStatus,
    derivation_spec_json,
)
from scripts.physics_material_damage import (
    DamageSourceRecord,
    MATERIAL_DAMAGE_MAPPING_SCHEMA_VERSION,
    MATERIAL_UNAVAILABLE_LABEL,
    MAPPING_SOURCE_FACTS,
    SUPPORTED_DAMAGE_LIFECYCLE_MAPPING,
    source_cohort_identity_for_records,
)
from scripts.production_plan import (
    ProductionPlan,
    create_production_plan,
    execute_production_plan,
    load_production_plan,
    production_plan_path,
    write_production_plan,
)
from scripts.representative_pilot import PilotReport
from scripts.scenario_manifest import (
    BenchmarkCondition,
    create_generated_manifest,
    scenario_manifest_projection,
)


XML = b'''<?xml version="1.0" encoding="utf-8"?>
<Level width="2"><Camera maxWidth="30" minWidth="20"/><Score highScore="0"/>
<Birds><Bird type="BirdRed"/></Birds><Slingshot x="-8" y="-2"/>
<GameObjects><Pig type="BasicSmall" material="" x="1" y="-3" rotation="0"/></GameObjects></Level>
'''


def _scenario(
    scenario_id: str,
    exposure_role: str,
    seed: int,
    *,
    negative_cap: int = 1,
    max_attempts: int = 3,
    transient_codes: tuple[str, ...] = ("engine_start_timeout", "transport_unavailable"),
) -> dict[str, object]:
    manifest = create_generated_manifest(
        XML,
        benchmark_condition=BenchmarkCondition("novelty_level_1", "type0101"),
        template_identity=f"scenario-template-v1:{scenario_id}",
        generator_identity="fixture-generator",
        generator_version="v1",
        generation_seed=seed,
        declared_inputs={"seed": seed},
        parameter_realization={"offset": seed},
    )
    return {
        "scenario_id": scenario_id,
        "exposure_role": exposure_role,
        **scenario_manifest_projection(manifest, f"fixtures/{scenario_id}.scenario.json"),
        "expected_initial_engine_state_identity": f"initial-state:{scenario_id}",
        "retry_policy": {
            "max_attempts": max_attempts,
            "transient_failure_codes": list(transient_codes),
            "stopping_rule": "execute_all_interventions",
        },
        "negative_specification": {
            "cap": negative_cap,
            "intervention_ids": [f"miss-{scenario_id}"],
            "semantic_justification": "bounded negative fixture intervention",
        },
        "interventions": [
            {
                "id": f"shot-{scenario_id}",
                "ordinal": 1,
                "intended_coverage_stratum": "collision",
                "source": "geometry_stratified",
                "interface_action": {
                    "action_type": "drag_hold_release",
                    "coordinate_frame": "slingshot_relative",
                    "drag_start": [100, 200],
                    "drag_release": [30, 50],
                    "tapTime": 0,
                    "releaseTime": 600,
                    "frame_height": 480,
                    "socket_command": {"x": 130, "y": 329, "tapTime": 0, "releaseTime": 600},
                },
                "engine_relative_action": {
                    "coordinate_frame": "slingshot_relative",
                    "release_offset": [30, 50],
                    "release_point": [130, 150],
                    "tap_time_ms": 0,
                    "release_time_ms": 600,
                },
                "mapping_version": "science-birds-slingshot-relative-v1",
                "slingshot_reference": {"gameX": 100, "gameY": 200},
                "source_provenance": {
                    "scenario_geometry_identity": f"geometry:{scenario_id}",
                    "stratum": "fixture",
                    "feasibility_rule": "fixture-v1",
                },
            },
            {
                "id": f"miss-{scenario_id}",
                "ordinal": 2,
                "intended_coverage_stratum": "no-contact/miss",
                "source": "targeted_rare",
                "interface_action": {
                    "action_type": "drag_hold_release",
                    "coordinate_frame": "slingshot_relative",
                    "drag_start": [100, 200],
                    "drag_release": [-20, 40],
                    "tapTime": 0,
                    "releaseTime": 600,
                    "frame_height": 480,
                    "socket_command": {"x": 80, "y": 319, "tapTime": 0, "releaseTime": 600},
                },
                "engine_relative_action": {
                    "coordinate_frame": "slingshot_relative",
                    "release_offset": [-20, 40],
                    "release_point": [80, 160],
                    "tap_time_ms": 0,
                    "release_time_ms": 600,
                },
                "mapping_version": "science-birds-slingshot-relative-v1",
                "slingshot_reference": {"gameX": 100, "gameY": 200},
                "source_provenance": {
                    "target_stratum": "no-contact/miss",
                    "selection_rule": "fixture-clearance-v1",
                },
            },
        ],
        "source_dispositions": {
            "geometry_stratified": {"status": "included"},
            "targeted_rare": {"status": "included"},
            "benchmark_agent_replay": {"status": "unavailable", "rationale": "not needed"},
        },
        "coverage_strata": {
            "no-contact/miss": {"status": "targeted", "intervention_ids": [f"miss-{scenario_id}"]},
            "collision": {"status": "targeted", "intervention_ids": [f"shot-{scenario_id}"]},
            "persistent support": {"status": "inapplicable", "rationale": "not needed"},
            "support change": {"status": "inapplicable", "rationale": "not needed"},
            "destruction": {"status": "inapplicable", "rationale": "not needed"},
            "pig removal": {"status": "inapplicable", "rationale": "not needed"},
            "explosion": {"status": "inapplicable", "rationale": "not needed"},
            "stability transitions": {"status": "inapplicable", "rationale": "not needed"},
            "level clear": {"status": "inapplicable", "rationale": "not needed"},
            "level fail": {"status": "inapplicable", "rationale": "not needed"},
        },
    }


def _collection_plan(
    *,
    training_cap: int = 1,
    final_cap: int = 1,
    training_max_attempts: int = 3,
    final_max_attempts: int = 3,
    training_codes: tuple[str, ...] = ("engine_start_timeout", "transport_unavailable"),
    final_codes: tuple[str, ...] = ("engine_start_timeout", "transport_unavailable"),
):
    return create_collection_plan(
        plan_version=3,
        scenarios=[
            _scenario(
                "training",
                "training",
                1,
                negative_cap=training_cap,
                max_attempts=training_max_attempts,
                transient_codes=training_codes,
            ),
            _scenario(
                "final",
                "final_evaluation",
                2,
                negative_cap=final_cap,
                max_attempts=final_max_attempts,
                transient_codes=final_codes,
            ),
        ],
    )


def _pilot_report(collection_plan, *, accepted: bool = True, marker: str = "fixture") -> PilotReport:
    def macro_evidence(attempt_id: str, role: str) -> dict[str, object]:
        return {
            "attempt_id": attempt_id,
            "capture_id": f"fixture-{role}-capture",
            "shot_id": f"fixture-{role}-shot",
            "derivation_spec_version": DERIVATION_SPEC_VERSION,
            "value_summary": {"true": 0, "false": 1, "null": 0},
            "availability_summary": {
                Availability.AVAILABLE.value: 1,
                Availability.UNAVAILABLE_NO_PREDECESSOR.value: 0,
                Availability.UNAVAILABLE_INSUFFICIENT_STATE_EVIDENCE.value: 0,
            },
        }

    macro_evidence_rows = [
        macro_evidence("attempt-training", "training"),
        macro_evidence("attempt-final", "final"),
    ]
    macro_evidence_by_attempt = {
        str(row["attempt_id"]): row for row in macro_evidence_rows
    }
    canonical_predicates = derivation_spec_json()["pending_predicates"]
    macro_semantics = {
        "schema": "representative_macro_semantics_v1",
        "derivation_spec_version": DERIVATION_SPEC_VERSION,
        "predicates": {
            name: {
                "status": SemanticStatus.HYPOTHESIS_PENDING_REPRESENTATIVE_VALIDATION.value,
                "definition": canonical_predicates[name]["definition"],
                "prerequisites": canonical_predicates[name]["prerequisites"],
                "unavailable_cases": canonical_predicates[name]["unavailable_cases"],
                "failure_cases": canonical_predicates[name]["failure_cases"],
                "pending_reason": (
                    "no authorized non-fixture representative engine evidence is recorded; "
                    "fixture-derived labels are diagnostic only and do not validate semantics"
                ),
                "evidence": [dict(row) for row in macro_evidence_rows],
            }
            for name in ("cascade-active", "collapsed", "pigs-cleared")
        },
    }
    version_envelope = {
        "generator_version": "v1",
    }
    source_record_objects = tuple(sorted(
        (
            DamageSourceRecord("fixture-training-capture", "fixture-training-shot", (), ()),
            DamageSourceRecord("fixture-final-capture", "fixture-final-shot", (), ()),
        ),
        key=lambda record: (record.capture_id, record.shot_id),
    ))
    source_records = [record.to_json() for record in source_record_objects]
    cohort_context = {
        "plan_identity": collection_plan.identity,
        "report_version": "3",
        "version_envelope": json.dumps(version_envelope, sort_keys=True, separators=(",", ":")),
    }
    source_cohort_identity = source_cohort_identity_for_records(
        source_record_objects,
        cohort_context=cohort_context,
    )
    pending_status = SemanticStatus.HYPOTHESIS_PENDING_REPRESENTATIVE_VALIDATION.value
    damage_evidence_by_attempt = {
        attempt_id: {
            "attempt_id": attempt_id,
            "capture_id": capture_id,
            "shot_id": shot_id,
            "record_count": 0,
            "mapping_version": SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.mapping_version,
            "source_cohort_identity": source_cohort_identity,
            "receipt_status": pending_status,
            "receipt_cohort_context": dict(cohort_context),
            "receipt_source_records": source_records,
        }
        for attempt_id, capture_id, shot_id in (
            ("attempt-training", "fixture-training-capture", "fixture-training-shot"),
            ("attempt-final", "fixture-final-capture", "fixture-final-shot"),
        )
    }
    atomic_validation = [
        {
            "attempt_id": attempt_id,
            "scenario_id": scenario_id,
            "accepted": True,
            "capture_id": damage_evidence_by_attempt[attempt_id]["capture_id"],
            "shot_id": damage_evidence_by_attempt[attempt_id]["shot_id"],
            "macro_semantics_evidence": {
                name: dict(macro_evidence_by_attempt[attempt_id])
                for name in ("cascade-active", "collapsed", "pigs-cleared")
            },
            "material_damage_evidence": damage_evidence_by_attempt[attempt_id],
        }
        for attempt_id, scenario_id in (
            ("attempt-training", "training"),
            ("attempt-final", "final"),
        )
    ]
    payload = {
        "schema": "representative_pilot_report_v3",
        "report_version": 3,
        "identity": "",
        "version_envelope": version_envelope,
        "plan_identity": collection_plan.identity,
        "plan_version": collection_plan.plan_version,
        "scenarios": [
            {"scenario_id": "training", "exposure_role": "training"},
            {"scenario_id": "final", "exposure_role": "final_evaluation"},
        ],
        "attempts": {
            "pilot_evidence": {
                "accepted_count": 2,
                "excluded_count": 0,
                "accepted_attempt_ids": ["attempt-training", "attempt-final"],
                "exclusions": [],
            },
            "atomic_validation": atomic_validation,
        },
        "coverage": {"fixture_marker": marker},
        "replays": [],
        "initial_state_identities": [],
        "partition_audits": [],
        "supervision": [],
        "macro_semantics": macro_semantics,
        "material_damage_semantics": {
            "schema": "representative_material_damage_semantics_v1",
            "source_cohort_identity": source_cohort_identity,
            "material": {
                "availability": "unavailable_missing_engine_material_field",
                "label": MATERIAL_UNAVAILABLE_LABEL,
                "reason": "physics_capture_v1 does not export a material field",
                "status": "unavailable",
            },
            "damage": {
                "availability": "unavailable_insufficient_damage_lifecycle_evidence",
                "mapping_schema_version": MATERIAL_DAMAGE_MAPPING_SCHEMA_VERSION,
                "mapping_version": SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.mapping_version,
                "source_facts": list(MAPPING_SOURCE_FACTS),
                "status": pending_status,
                "source_cohort_identity": source_cohort_identity,
                "cohort_context": cohort_context,
                "source_records": source_records,
                "evidence": list(damage_evidence_by_attempt.values()),
            },
        },
        "available_capabilities": [],
        "unavailable_capabilities": [{
            "capability": "representative_macro_semantics",
            "reason": "fixture report keeps representative macro semantics unavailable",
        }],
        "unavailable_labels": {
            "material": "no accepted engine-verified material mapping exists",
            "damage": "no representative engine lifecycle evidence verifies the damage mapping",
        },
        "permanent_or_systematic_exporter_defects": [],
        "pilot_status": "accepted" if accepted else "rejected",
        "acceptance_decision": {"accepted": accepted, "reasons": [] if accepted else ["fixture rejection"]},
    }
    payload["identity"] = (
        "representative-pilot-report-v3:3:"
        + quote(collection_plan.identity, safe="-._~")
    )
    return PilotReport.from_dict(payload)


def _parameters() -> dict[str, object]:
    return {
        "capture": {"capture_stride": 2, "stability_window": 12, "rollout_ceiling": 600},
        "tolerances": {"geometric": 0.01, "motion": 0.02, "numeric": 1e-6},
        "prospective_quotas": {"collision": 20},
        "bounded_negative_cap": 1,
        "transient_retry_counts": {"engine_start_timeout": 2, "transport_unavailable": 2},
    }


def _evidence(attempt_id: str = "attempt-training") -> dict[str, object]:
    def evidence_for(value, path: str):
        if isinstance(value, dict):
            return {key: evidence_for(item, f"{path}.{key}" if path else key) for key, item in value.items()}
        return {
            "attempt_ids": [attempt_id],
            "rationale": f"pilot evidence for {path}",
            "derivation": f"select {path} from accepted pilot measurements",
        }

    return evidence_for(_parameters(), "")


class ProductionPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.collection_plan = _collection_plan()
        self.report = _pilot_report(self.collection_plan)

    def test_successful_roundtrip_is_canonical_identity_bound_and_immutable(self) -> None:
        parameters = _parameters()
        evidence = _evidence()
        plan = create_production_plan(
            plan_version=1,
            pilot_report=self.report,
            collection_plan=self.collection_plan,
            parameters=parameters,
            evidence=evidence,
        )
        parameters["capture"]["capture_stride"] = 99
        evidence["capture"]["capture_stride"]["attempt_ids"].append("changed")
        self.assertEqual(plan.parameters["capture"]["capture_stride"], 2)
        with self.assertRaises(TypeError):
            plan.parameters["prospective_quotas"]["collision"] = 100

        with tempfile.TemporaryDirectory() as temporary:
            publication_dir = Path(temporary) / "nested"
            path = write_production_plan(plan, publication_dir)
            self.assertEqual(path, production_plan_path(publication_dir, plan))
            first = path.read_bytes()
            loaded = load_production_plan(path)
            self.assertEqual(loaded, plan)
            self.assertEqual(loaded.identity, plan.identity)
            self.assertEqual(write_production_plan(plan, publication_dir), path)
            self.assertEqual(path.read_bytes(), first)
            self.assertEqual(set(json.loads(first)), {
                "schema", "plan_version", "identity", "source_pilot_report",
                "source_collection_plan", "parameters", "evidence",
            })

    def test_rejects_rejected_pilot_and_mismatched_collection_plan(self) -> None:
        with self.assertRaisesRegex(ValueError, "accepted"):
            create_production_plan(
                plan_version=1,
                pilot_report=_pilot_report(self.collection_plan, accepted=False),
                collection_plan=self.collection_plan,
                parameters=_parameters(),
                evidence=_evidence(),
            )
        other_plan = create_collection_plan(plan_version=4, scenarios=[_scenario("other", "training", 3)])
        with self.assertRaisesRegex(ValueError, "collection plan"):
            create_production_plan(
                plan_version=1,
                pilot_report=self.report,
                collection_plan=other_plan,
                parameters=_parameters(),
                evidence=_evidence(),
            )

    def test_rejects_invalid_or_missing_parameters_and_evidence(self) -> None:
        invalid_values = (-1, math.inf, math.nan, True)
        for value in invalid_values:
            with self.subTest(value=value):
                parameters = _parameters()
                parameters["tolerances"]["numeric"] = value
                with self.assertRaises(ValueError):
                    create_production_plan(
                        plan_version=1,
                        pilot_report=self.report,
                        collection_plan=self.collection_plan,
                        parameters=parameters,
                        evidence=_evidence(),
                    )
        for mutation in (
            "missing-group",
            "empty-quota",
            "missing-evidence-leaf",
            "extra-evidence-leaf",
            "missing-derivation",
            "unknown-attempt",
        ):
            with self.subTest(mutation=mutation):
                parameters = _parameters()
                evidence = _evidence()
                if mutation == "missing-group":
                    parameters.pop("capture")
                elif mutation == "empty-quota":
                    parameters["prospective_quotas"] = {}
                elif mutation == "missing-evidence-leaf":
                    evidence["tolerances"].pop("numeric")
                elif mutation == "extra-evidence-leaf":
                    evidence["prospective_quotas"]["not-a-quota"] = evidence["prospective_quotas"]["collision"]
                elif mutation == "missing-derivation":
                    evidence["capture"]["capture_stride"].pop("derivation")
                else:
                    evidence["capture"]["capture_stride"]["attempt_ids"] = ["not-accepted"]
                with self.assertRaises(ValueError):
                    create_production_plan(
                        plan_version=1,
                        pilot_report=self.report,
                        collection_plan=self.collection_plan,
                        parameters=parameters,
                        evidence=evidence,
                    )

    def test_accepts_explicitly_empty_retry_counts_when_pilot_declared_none(self) -> None:
        collection_plan = _collection_plan(
            training_max_attempts=1,
            final_max_attempts=1,
            training_codes=(),
            final_codes=(),
        )
        parameters = _parameters()
        parameters["transient_retry_counts"] = {}
        evidence = _evidence()
        evidence["transient_retry_counts"] = {}

        plan = create_production_plan(
            plan_version=1,
            pilot_report=_pilot_report(collection_plan),
            collection_plan=collection_plan,
            parameters=parameters,
            evidence=evidence,
        )

        self.assertEqual(dict(plan.parameters["transient_retry_counts"]), {})

    def test_rejects_final_evaluation_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "final_evaluation"):
            create_production_plan(
                plan_version=1,
                pilot_report=self.report,
                collection_plan=self.collection_plan,
                parameters=_parameters(),
                evidence=_evidence("attempt-final"),
            )

    def test_new_version_with_changed_values_preserves_prior_plan(self) -> None:
        first_parameters = _parameters()
        first = create_production_plan(
            plan_version=1,
            pilot_report=self.report,
            collection_plan=self.collection_plan,
            parameters=first_parameters,
            evidence=_evidence(),
        )
        second_parameters = _parameters()
        second_parameters["capture"]["capture_stride"] = 3
        second = create_production_plan(
            plan_version=2,
            pilot_report=self.report,
            collection_plan=self.collection_plan,
            parameters=second_parameters,
            evidence=_evidence(),
        )
        self.assertEqual(first.plan_version, 1)
        self.assertEqual(first.parameters["capture"]["capture_stride"], 2)
        self.assertEqual(second.parameters["capture"]["capture_stride"], 3)
        self.assertNotEqual(first.identity, second.identity)
        self.assertIsInstance(second, ProductionPlan)

    def test_write_is_idempotent_but_never_overwrites_different_bytes(self) -> None:
        first = create_production_plan(
            plan_version=1,
            pilot_report=self.report,
            collection_plan=self.collection_plan,
            parameters=_parameters(),
            evidence=_evidence(),
        )
        changed_parameters = _parameters()
        changed_parameters["capture"]["capture_stride"] = 3
        changed_same_version = create_production_plan(
            plan_version=1,
            pilot_report=self.report,
            collection_plan=self.collection_plan,
            parameters=changed_parameters,
            evidence=_evidence(),
        )
        changed_version = create_production_plan(
            plan_version=2,
            pilot_report=self.report,
            collection_plan=self.collection_plan,
            parameters=_parameters(),
            evidence=_evidence(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            publication_dir = Path(temporary)
            path = write_production_plan(first, publication_dir)
            original = path.read_bytes()
            self.assertEqual(write_production_plan(first, publication_dir), path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(production_plan_path(publication_dir, changed_same_version), path)
            with self.assertRaisesRegex(ValueError, "different bytes"):
                write_production_plan(changed_same_version, publication_dir)
            self.assertEqual(path.read_bytes(), original)
            version_two_path = write_production_plan(changed_version, publication_dir)
            self.assertNotEqual(version_two_path, path)
            self.assertTrue(version_two_path.is_file())

    def test_execution_requires_existing_bound_production_plan_before_delegating(self) -> None:
        production_plan = create_production_plan(
            plan_version=1,
            pilot_report=self.report,
            collection_plan=self.collection_plan,
            parameters=_parameters(),
            evidence=_evidence(),
        )
        other_collection = create_collection_plan(
            plan_version=4,
            scenarios=[
                _scenario("training", "training", 4),
                _scenario("final", "final_evaluation", 5),
            ],
        )
        other_report = _pilot_report(other_collection)
        other_production = create_production_plan(
            plan_version=1,
            pilot_report=other_report,
            collection_plan=other_collection,
            parameters=_parameters(),
            evidence=_evidence(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection_path = root / "collection-plan.json"
            write_collection_plan(self.collection_plan, collection_path)
            loaded_collection = load_collection_plan(collection_path)
            production_path = write_production_plan(production_plan, root / "published")
            runtime = object()
            expected = {"status": "delegated"}
            output = root / "output"
            expected_context = {
                "production_plan_identity": production_plan.identity,
                "production_plan_version": production_plan.plan_version,
                "source_pilot_report_identity": self.report.identity,
                "parameters": production_plan.to_dict()["parameters"],
            }

            def delegated(*args, **kwargs):
                self.assertEqual(
                    (output / "production_parameter_plan.json").read_bytes(),
                    production_path.read_bytes(),
                )
                return expected

            with mock.patch("scripts.production_plan.execute_collection_plan", side_effect=delegated) as execute:
                for invalid in (None, root / "missing-production-plan.json", other_production):
                    with self.subTest(invalid=invalid):
                        with self.assertRaises(ValueError):
                            execute_production_plan(loaded_collection, invalid, runtime, output)
                execute.assert_not_called()
                self.assertIs(
                    execute_production_plan(loaded_collection, production_path, runtime, output),
                    expected,
                )
                execute.assert_called_once_with(
                    loaded_collection,
                    runtime,
                    output,
                    execution_context=expected_context,
                )
                self.assertEqual(
                    (output / "production_parameter_plan.json").read_bytes(),
                    production_path.read_bytes(),
                )

    def test_execution_rejects_valid_plan_bytes_at_non_authoritative_path(self) -> None:
        production_plan = create_production_plan(
            plan_version=1,
            pilot_report=self.report,
            collection_plan=self.collection_plan,
            parameters=_parameters(),
            evidence=_evidence(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection_path = root / "collection-source.json"
            write_collection_plan(self.collection_plan, collection_path)
            loaded_collection = load_collection_plan(collection_path)
            authoritative = write_production_plan(production_plan, root / "published")
            arbitrary = authoritative.parent / "arbitrary.json"
            arbitrary.write_bytes(authoritative.read_bytes())
            output = root / "output"

            with mock.patch("scripts.production_plan.execute_collection_plan") as execute:
                with self.assertRaisesRegex(ValueError, "authoritative publication path"):
                    execute_production_plan(loaded_collection, arbitrary, object(), output)
                execute.assert_not_called()
            self.assertFalse(output.exists())

    def test_execution_revalidates_parameters_and_preserves_published_output_plan(self) -> None:
        production_plan = create_production_plan(
            plan_version=1,
            pilot_report=self.report,
            collection_plan=self.collection_plan,
            parameters=_parameters(),
            evidence=_evidence(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection_path = root / "collection-plan.json"
            write_collection_plan(self.collection_plan, collection_path)
            loaded_collection = load_collection_plan(collection_path)
            production_path = write_production_plan(production_plan, root / "published")
            payload = json.loads(production_path.read_text(encoding="utf-8"))
            payload["parameters"]["bounded_negative_cap"] = 2
            invalid_plan = ProductionPlan.from_dict(payload)
            invalid_dir = root / "invalid-published"
            invalid_dir.mkdir()
            invalid_path = production_plan_path(invalid_dir, invalid_plan)
            invalid_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            output = root / "output"
            with mock.patch("scripts.production_plan.execute_collection_plan") as execute:
                with self.assertRaisesRegex(ValueError, "bounded_negative_cap"):
                    execute_production_plan(loaded_collection, invalid_path, object(), output)
                execute.assert_not_called()
                self.assertFalse(output.exists())

            output.mkdir()
            frozen_copy = output / "production_parameter_plan.json"
            frozen_copy.write_bytes(b"different")
            with mock.patch("scripts.production_plan.execute_collection_plan") as execute:
                with self.assertRaisesRegex(ValueError, "different"):
                    execute_production_plan(loaded_collection, production_path, object(), output)
                execute.assert_not_called()
                self.assertEqual(frozen_copy.read_bytes(), b"different")

    def test_execution_preflight_does_not_associate_plan_with_completed_output(self) -> None:
        production_plan = create_production_plan(
            plan_version=1,
            pilot_report=self.report,
            collection_plan=self.collection_plan,
            parameters=_parameters(),
            evidence=_evidence(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection_path = root / "collection-source.json"
            write_collection_plan(self.collection_plan, collection_path)
            loaded_collection = load_collection_plan(collection_path)
            production_path = write_production_plan(production_plan, root / "published")
            for state_name in ("collection_plan.json", "collection_plan_report.json"):
                with self.subTest(state_name=state_name):
                    output = root / f"output-{state_name}"
                    output.mkdir()
                    state = output / state_name
                    state.write_bytes(b"existing collection state")
                    with mock.patch("scripts.production_plan.execute_collection_plan") as execute:
                        with self.assertRaisesRegex(ValueError, "execution state"):
                            execute_production_plan(loaded_collection, production_path, object(), output)
                        execute.assert_not_called()
                    self.assertFalse((output / "production_parameter_plan.json").exists())
                    self.assertEqual(state.read_bytes(), b"existing collection state")

    def test_execution_detects_post_delegate_production_copy_tampering(self) -> None:
        production_plan = create_production_plan(
            plan_version=1,
            pilot_report=self.report,
            collection_plan=self.collection_plan,
            parameters=_parameters(),
            evidence=_evidence(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection_path = root / "collection-source.json"
            write_collection_plan(self.collection_plan, collection_path)
            loaded_collection = load_collection_plan(collection_path)
            production_path = write_production_plan(production_plan, root / "published")
            output = root / "output"

            def tamper(*args, **kwargs):
                (output / "production_parameter_plan.json").write_bytes(b"tampered")
                return {"status": "delegated"}

            with mock.patch("scripts.production_plan.execute_collection_plan", side_effect=tamper):
                with self.assertRaisesRegex(ValueError, "changed during execution"):
                    execute_production_plan(loaded_collection, production_path, object(), output)

    def test_collection_negative_cap_and_retry_policy_must_match(self) -> None:
        for mutation in (
            "negative-value",
            "inconsistent-caps",
            "retry-value",
            "missing-retry",
            "unknown-retry",
            "inconsistent-retries",
            "different-retry-codes",
        ):
            with self.subTest(mutation=mutation):
                collection_plan = self.collection_plan
                parameters = _parameters()
                if mutation == "negative-value":
                    parameters["bounded_negative_cap"] = 2
                elif mutation == "inconsistent-caps":
                    collection_plan = _collection_plan(final_cap=2)
                elif mutation == "retry-value":
                    parameters["transient_retry_counts"]["engine_start_timeout"] = 1
                elif mutation == "missing-retry":
                    parameters["transient_retry_counts"].pop("transport_unavailable")
                elif mutation == "unknown-retry":
                    parameters["transient_retry_counts"]["capture_temporarily_unavailable"] = 2
                elif mutation == "different-retry-codes":
                    collection_plan = _collection_plan(
                        final_codes=("engine_start_timeout", "capture_temporarily_unavailable"),
                    )
                else:
                    collection_plan = _collection_plan(final_max_attempts=2)
                report = self.report if collection_plan is self.collection_plan else _pilot_report(collection_plan)
                with self.assertRaises(ValueError):
                    create_production_plan(
                        plan_version=1,
                        pilot_report=report,
                        collection_plan=collection_plan,
                        parameters=parameters,
                        evidence=_evidence(),
                    )

    def test_quota_keys_must_be_targeted_in_every_collection_scenario(self) -> None:
        for quota_key in ("support change", "support_change", "not-a-stratum"):
            with self.subTest(quota_key=quota_key):
                parameters = _parameters()
                parameters["prospective_quotas"] = {quota_key: 1}
                with self.assertRaisesRegex(ValueError, "quota"):
                    create_production_plan(
                        plan_version=1,
                        pilot_report=self.report,
                        collection_plan=self.collection_plan,
                        parameters=parameters,
                        evidence=_evidence(),
                    )


if __name__ == "__main__":
    unittest.main()
