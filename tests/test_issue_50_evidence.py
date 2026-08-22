from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_issue_50_evidence import (
    PROBE_CASES,
    build_issue_50_evidence,
    validate_issue_50_evidence,
)
from scripts.build_issue_50_probe_plan import build_issue_50_probe_plan
from scripts.cohort_v2_physical_violations import (
    EXCESS_PENETRATION,
    UNSUPPORTED_STATIONARY,
    derive_capture_physical_violations,
)
from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_json
from scripts.collection_plan import load_collection_plan
from scripts.physics_capture_v2 import PhysicsCaptureV2Error, parse_physics_capture_v2


ROOT = Path(__file__).resolve().parents[1]
ISSUE_44 = ROOT / "data/runtime_evidence/issue-44"


def _record(case: str) -> dict:
    return json.loads((ISSUE_44 / f"captures/{case}.json").read_text(encoding="utf-8"))


def _derive(record: dict, reference: str = "test-capture.json"):
    capture = parse_physics_capture_v2(record)
    return capture, derive_capture_physical_violations(
        capture,
        source_reference=reference,
        source_capture_bundle_identity="test-capture-bundle-v1",
    )


def _label(derivation, fixed_step: int, predicate: str, entity_id: str | None = None):
    record = next(item for item in derivation["labels"] if item["fixed_step"] == fixed_step)
    value = record["predicates"][predicate]
    if predicate == EXCESS_PENETRATION:
        return value
    return next(item for item in value if item["entity_id"] == entity_id)


def _floating_record(source_case: str, case: str) -> dict:
    record = _record(source_case)
    record["capture_id"] = f"capture-v2:test-{case}"
    record["shot_id"] = f"shot-v2:test-{case}"
    for sample in record["fixed_step_samples"]:
        bird = next(entity for entity in sample["entities"] if entity["entity_id"] == "runtime:bird:0000")
        block = next(entity for entity in sample["entities"] if entity["scenario_object_id"] == "block:0000")
        bird["scenario_object_id"] = "block:0000"
        block["scenario_object_id"] = "block:source-negative"
    for sample in record["fixed_step_samples"][:2]:
        bird = next(entity for entity in sample["entities"] if entity["entity_id"] == "runtime:bird:0000")
        bird["body"].update(
            {
                "gravity_scale": 0.48,
                "gravity_applicable": True,
                "velocity": [0, 0],
                "angular_velocity_degrees_per_second": 0,
            }
        )
    return record


def _probe_root(path: Path) -> Path:
    authority_root = path
    plan = build_issue_50_probe_plan(authority_root)
    records = {}
    probes = []
    for case in PROBE_CASES:
        source_case = "collision" if case.startswith("floating-a") else "support-change"
        record = _floating_record(source_case, case)
        capture = parse_physics_capture_v2(record)
        records[case] = record
        probes.append(
            {
                "case": case,
                "capture_id": capture.capture_id,
                "scenario_lineage_id": capture.source_bindings["scenario_lineage_id"],
                "level_instance_id": capture.source_bindings["level_instance_id"],
                "scenario_template_id": capture.source_bindings["scenario_template_id"],
            }
        )
        write_immutable_cohort_v2_json(record, authority_root / f"captures/{case}.json")
    runtime_identity = "issue-50-test-runtime-bundle-v1"
    runtime = {
        "schema": "issue_50_physical_violation_probe_runtime_bundle_v1",
        "identity": runtime_identity,
        "source_snapshot_commit": "test-source-snapshot",
        "collection_plan_identity": plan["plan_identity"],
        "evidence_source": "unity_runtime_non_fixture",
        "final_evaluation": False,
        "probe_environment": "NOVPHY_ISSUE_50_CAPABILITY_PROBE=unsupported-stationary-v1",
        "probes": probes,
    }
    bundle = {
        "schema": "issue_50_physical_violation_probe_capture_bundle_v1",
        "identity": "issue-50-test-capture-bundle-v1",
        "runtime_bundle_identity": runtime_identity,
        "captures": {
            case: {
                "capture_id": record["capture_id"],
                "path": f"captures/{case}.json",
            }
            for case, record in records.items()
        },
    }
    write_immutable_cohort_v2_json(runtime, authority_root / "runtime-bundle-manifest.json")
    write_immutable_cohort_v2_json(bundle, authority_root / "capture-bundle-manifest.json")
    return authority_root


class CohortV2PhysicalViolationDerivationTests(unittest.TestCase):
    def test_penetration_uses_strict_frozen_tolerance_and_exact_contact_citations(self) -> None:
        capture, derivation = _derive(_record("collision"))
        collision_step = next(
            event["fixed_step"]
            for event in capture.record["events"]
            if event["event_type"] == "collision"
        )
        positive = _label(derivation, collision_step, EXCESS_PENETRATION)
        negative = _label(derivation, collision_step - 1, EXCESS_PENETRATION)

        self.assertIs(positive["value"], True)
        self.assertIs(negative["value"], False)
        self.assertEqual(positive["evidence"]["penetration_tolerance_unity_units"], 0.006)
        self.assertEqual(positive["source_records"]["fixed_steps"], [collision_step])
        self.assertTrue(positive["source_records"]["contact_ids"])

    def test_stationary_unsupported_is_unavailable_then_true_without_a_regime_label(self) -> None:
        capture, derivation = _derive(_floating_record("collision", "floating"))
        entity_id = "runtime:bird:0000"
        first_step = capture.record["fixed_step_samples"][0]["fixed_step"]
        second_step = first_step + 1
        first = _label(derivation, first_step, UNSUPPORTED_STATIONARY, entity_id)
        second = _label(derivation, second_step, UNSUPPORTED_STATIONARY, entity_id)

        self.assertIsNone(first["value"])
        self.assertEqual(first["availability"], "unavailable_incomplete_stability_window")
        self.assertIs(second["value"], True)
        self.assertTrue(second["evidence"]["gravity_applicable_all_steps"])
        self.assertTrue(second["evidence"]["stationary"])
        self.assertFalse(second["evidence"]["supported"])
        self.assertNotIn("physical_regime", json.dumps(derivation))

        first_aggregate = derivation["labels"][0]["aggregate"]
        second_aggregate = derivation["labels"][1]["aggregate"]
        self.assertIsNone(first_aggregate["value"])
        self.assertEqual(first_aggregate["availability"], "unavailable_component")
        self.assertIs(second_aggregate["value"], True)

    def test_missing_geometry_is_whole_rollout_invalidation_not_false(self) -> None:
        record = _record("collision")
        del record["fixed_step_samples"][0]["colliders"][0]["shape"]
        with self.assertRaisesRegex(PhysicsCaptureV2Error, "missing fields"):
            parse_physics_capture_v2(record)

    def test_aggregate_can_be_false_without_treating_initial_unknown_as_zero(self) -> None:
        _, derivation = _derive(_record("no-contact"))
        self.assertIsNone(derivation["labels"][0]["aggregate"]["value"])
        self.assertEqual(
            derivation["labels"][0]["aggregate"]["availability"],
            "unavailable_component",
        )
        self.assertIs(derivation["labels"][-1]["aggregate"]["value"], False)
        self.assertEqual(derivation["labels"][-1]["aggregate"]["availability"], "available")


class Issue50ProbePlanTests(unittest.TestCase):
    def test_freezes_two_source_bound_nonfinal_lineages_and_four_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = build_issue_50_probe_plan(Path(temporary))
            plan = load_collection_plan(Path(result["plan_path"])).plan
            interventions = [
                intervention
                for scenario in plan.scenarios
                for intervention in scenario.interventions
            ]
            self.assertEqual(len(plan.scenarios), 2)
            self.assertEqual(len(interventions), 4)
            self.assertEqual({scenario.exposure_role for scenario in plan.scenarios}, {"calibration"})
            self.assertEqual(
                len({item["scenario_lineage_id"] for item in result["scenarios"].values()}),
                2,
            )
            self.assertEqual(
                len({item["scenario_template_id"] for item in result["scenarios"].values()}),
                2,
            )
            for scenario in result["scenarios"].values():
                self.assertIn(
                    'physicsViolationProbe="unsupported_stationary_v1"',
                    Path(scenario["xml_path"]).read_text(encoding="utf-8"),
                )


class Issue50EvidenceBundleTests(unittest.TestCase):
    def test_dry_run_and_publication_pass_exact_rederivation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probe_root = _probe_root(root / "probes")
            dry_output = root / "dry-output"
            dry = build_issue_50_evidence(
                dry_output,
                probe_root=probe_root,
                dry_run=True,
            )
            self.assertTrue(dry["passed"])
            self.assertFalse(dry_output.exists())

            output = root / "issue-50"
            result = build_issue_50_evidence(output, probe_root=probe_root)
            self.assertTrue(result["passed"])
            self.assertEqual(result, validate_issue_50_evidence(output, probe_root=probe_root))


if __name__ == "__main__":
    unittest.main()
