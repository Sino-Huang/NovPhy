from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts.build_issue_51_evidence import (
    FAILED_RUNTIME_DETERMINATION_IDENTITY,
    Issue51EvidenceError,
    PRIOR_DETERMINATION_IDENTITY,
    _derived_artifacts,
    _load_issue_44,
    _micro_audit,
    _supplementary_sources,
    build_issue_51_evidence,
    source_bound_quarantine_audit,
)
from scripts.build_issue_51_pilot_plan import (
    TEMPLATE_REFERENCE,
    build_issue_51_supplementary_plan,
)
from scripts.cohort_v2_scenarios import load_cohort_v2_scenario_manifest
from scripts.collection_plan import RuntimeInput, load_collection_plan
from scripts.collect_rollouts import (
    classify_physics_capture_v2_coverage,
    validate_cohort_v2_constraints_authority,
)
from scripts.physics_capture_v2_persistence import source_bindings_from_collection
from scripts.physics_capture_v2 import parse_physics_capture_v2


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _supplementary(root: Path, *, level_clear: bool = True) -> Path:
    supplement = root / "supplementary"
    with redirect_stdout(io.StringIO()):
        authority = build_issue_51_supplementary_plan(supplement)
    shutil.copyfile(ROOT / TEMPLATE_REFERENCE, supplement / "template.xml")
    scenario = json.loads((supplement / "scenario-manifest.json").read_text(encoding="utf-8"))
    manifest = scenario["scenario_manifest"]
    plan = load_collection_plan(supplement / "collection-plan.json").plan
    interventions = {
        intervention.id: intervention.identity
        for collection_scenario in plan.scenarios
        for intervention in collection_scenario.interventions
    }
    source = json.loads(
        (ROOT / "data/runtime_evidence/issue-44/captures/stable-terminal.json").read_text(
            encoding="utf-8"
        )
    )
    attempts = []
    for index, (intervention_id, intervention_identity) in enumerate(
        sorted(interventions.items())
    ):
        record = deepcopy(source)
        record["capture_id"] = f"capture-v2:issue51-test-{index}"
        record["shot_id"] = f"shot-v2:issue51-test-{index}"
        record["source_bindings"] = {
            "scenario_template_id": scenario["template_record"]["identity"],
            "level_instance_id": manifest["level_instance"]["identity"],
            "scenario_lineage_id": manifest["scenario_lineage"]["identity"],
            "rollout_id": f"issue51-test-rollout-{index}",
            "intervention_id": intervention_identity,
        }
        terminal_reason = "stable_entered"
        if level_clear and intervention_id == "level-clear-targeted":
            terminal_reason = "level_clear"
            terminal_id = record["terminal_evidence"]["event_id"]
            terminal_event = next(
                event for event in record["events"] if event["event_id"] == terminal_id
            )
            terminal_event["event_type"] = "level_clear"
            terminal_event["payload"] = {"score": 1000}
            record["terminal_evidence"]["reason"] = "level_clear"
        parse_physics_capture_v2(record)
        realized = list(
            classify_physics_capture_v2_coverage(parse_physics_capture_v2(record))
        )
        capture_path = f"captures/{intervention_id}.json"
        _write_json(supplement / capture_path, record)
        attempts.append({
            "attempt_identity": f"issue51-test-attempt-{index}",
            "intervention_id": intervention_id,
            "intervention_identity": intervention_identity,
            "status": "accepted",
            "capture_id": record["capture_id"],
            "capture_path": capture_path,
            "terminal_reason": terminal_reason,
            "realized_coverage_strata": realized,
        })
    source_commit = "issue51-test-implementation"
    runtime = {
        "schema": "issue_51_supplementary_runtime_authority_v1",
        "identity": (
            f"issue-51-supplementary-runtime-v1:{authority['plan_identity']}:"
            f"{source_commit}"
        ),
        "evidence_source": "unity_runtime_non_fixture",
        "collection_plan_identity": authority["plan_identity"],
        "scenario_manifest_identity": authority["scenario_manifest_identity"],
        "source_snapshot_commit": source_commit,
        "unity_version": "2019.4.41f2",
        "physics_protocol_version": 1,
        "configured_fixed_step_capture_stride": 1,
        "attempts": attempts,
    }
    _write_json(supplement / "runtime-authority.json", runtime)
    (supplement / "prior-captures").mkdir()
    shutil.copyfile(
        supplement / "collection-plan.json",
        supplement / "prior-captures/collection-plan.json",
    )
    prior_attempts = []
    for index, (intervention_id, intervention_identity) in enumerate(
        sorted(interventions.items())
    ):
        record = deepcopy(source)
        record["capture_id"] = f"capture-v2:issue51-prior-test-{index}"
        record["shot_id"] = f"shot-v2:issue51-prior-test-{index}"
        record["source_bindings"] = {
            "scenario_template_id": scenario["template_record"]["identity"],
            "level_instance_id": manifest["level_instance"]["identity"],
            "scenario_lineage_id": manifest["scenario_lineage"]["identity"],
            "rollout_id": f"issue51-prior-test-rollout-{index}",
            "intervention_id": intervention_identity,
        }
        capture = parse_physics_capture_v2(record)
        capture_path = f"prior-captures/{intervention_id}.json"
        _write_json(supplement / capture_path, record)
        prior_attempts.append({
            "attempt_identity": f"issue51-prior-test-attempt-{index}",
            "intervention_id": intervention_id,
            "intervention_identity": intervention_identity,
            "status": "accepted",
            "capture_id": capture.capture_id,
            "capture_path": capture_path,
            "terminal_reason": capture.record["terminal_evidence"]["reason"],
            "realized_coverage_strata": list(
                classify_physics_capture_v2_coverage(capture)
            ),
        })
    prior = {
        "schema": "issue_51_prior_failed_pilot_determination_v1",
        "identity": PRIOR_DETERMINATION_IDENTITY,
        "disposition": "failed",
        "failure_reason": "unmet_level_clear",
        "collection_plan_identity": authority["plan_identity"],
        "collection_plan_path": "prior-captures/collection-plan.json",
        "counts": {
            "accepted": 2,
            "rejected": 0,
            "failed": 0,
            "quarantined": 0,
            "retried": 0,
        },
        "realized_coverage_shortfalls": [{
            "scenario_id": authority["scenario_id"],
            "intervention_id": "level-clear-targeted",
            "intended_coverage_stratum": "level clear",
            "realized_coverage_strata": next(
                attempt["realized_coverage_strata"]
                for attempt in prior_attempts
                if attempt["intervention_id"] == "level-clear-targeted"
            ),
        }],
        "attempts": prior_attempts,
    }
    _write_json(supplement / "prior-determination.json", prior)
    (supplement / "prior-failures").mkdir()
    shutil.copyfile(
        supplement / "collection-plan.json",
        supplement / "prior-failures/collection-plan.json",
    )
    failed_attempts = []
    failed_ledger = []
    unmet_slots = []
    failure_reason = (
        "physics capture v2 failed (14): physics capture v2 event is outside "
        "recorded fixed-step coverage"
    )
    for index, (intervention_id, intervention_identity) in enumerate(
        sorted(interventions.items())
    ):
        attempt_identity = f"issue51-failed-test-attempt-{index}"
        failure_path = f"prior-failures/{intervention_id}-failure.json"
        failure = {
            "schema": "collection_attempt_failure_v1",
            "attempt_id": attempt_identity,
            "attempt_number": 1,
            "status": "failed",
            "reason": failure_reason,
            "failure_code": "transport_unavailable",
            "failure_class": "permanent",
            "retryable": False,
            "retry_decision": "stop",
            "quarantine_path": f"prior-failures/{intervention_id}",
            "exception_type": "PhysicsCaptureV2Failure",
        }
        _write_json(supplement / failure_path, failure)
        failed_attempts.append({
            "attempt_identity": attempt_identity,
            "intervention_id": intervention_id,
            "intervention_identity": intervention_identity,
            "status": "failed",
            "failure_code": "transport_unavailable",
            "failure_class": "permanent",
            "reason": failure_reason,
            "failure_manifest_path": failure_path,
            "disposition": "quarantined",
            "eligible": False,
        })
        failed_ledger.append({
            "attempt_id": attempt_identity,
            "intervention_id": intervention_id,
            "intervention_identity": intervention_identity,
            "status": "failed",
            "disposition": "quarantine",
            "failure_code": "transport_unavailable",
            "failure_class": "permanent",
            "reason": failure_reason,
        })
        intended = next(
            intervention.intended_coverage_stratum
            for collection_scenario in plan.scenarios
            for intervention in collection_scenario.interventions
            if intervention.id == intervention_id
        )
        unmet_slots.append({
            "disposition": "failed",
            "intended_coverage_stratum": intended,
            "intervention_id": intervention_id,
            "scenario_id": authority["scenario_id"],
        })
    failed_report = {
        "plan_identity": authority["plan_identity"],
        "accepted_count": 0,
        "rejected_count": 0,
        "failed_count": 2,
        "quarantined_count": 2,
        "unmet_slots": unmet_slots,
        "realized_coverage_shortfalls": [],
        "attempt_ledger": failed_ledger,
    }
    _write_json(
        supplement / "prior-failures/collection-plan-report.json", failed_report
    )
    failed = {
        "schema": "issue_51_prior_failed_pilot_determination_v1",
        "identity": FAILED_RUNTIME_DETERMINATION_IDENTITY,
        "disposition": "failed",
        "failure_reason": "fixed_step_capture_gap",
        "collection_plan_identity": authority["plan_identity"],
        "collection_plan_path": "prior-failures/collection-plan.json",
        "collection_report_path": "prior-failures/collection-plan-report.json",
        "counts": {
            "accepted": 0,
            "rejected": 0,
            "failed": 2,
            "quarantined": 2,
            "retried": 0,
        },
        "unmet_slots": unmet_slots,
        "attempts": failed_attempts,
    }
    _write_json(supplement / "failed-determination.json", failed)
    return root


class Issue51PilotPlanTests(unittest.TestCase):
    def test_supplementary_level_preserves_exact_legacy_static_geometry(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            value = build_issue_51_supplementary_plan(Path(temporary))
            scenario = load_cohort_v2_scenario_manifest(
                Path(value["manifest_path"]),
                xml_path=Path(value["xml_path"]),
                template_source_path=Path(value["template_path"]),
            )
            self.assertEqual(scenario.scenario_manifest.generation.mode, "legacy_static")
            self.assertIsNone(scenario.template_record.generation_constraints)
            self.assertEqual(
                Path(value["xml_path"]).read_bytes(),
                Path(value["template_path"]).read_bytes(),
            )

    def test_legacy_static_authority_binds_without_generator_constraints(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            value = build_issue_51_supplementary_plan(Path(temporary))
            scenario = load_cohort_v2_scenario_manifest(
                Path(value["manifest_path"]),
                xml_path=Path(value["xml_path"]),
                template_source_path=Path(value["template_path"]),
            )
            loaded = load_collection_plan(Path(value["plan_path"]))
            collection_scenario = loaded.plan.scenarios[0]
            intervention = collection_scenario.interventions[0]
            request = RuntimeInput(
                plan_identity=loaded.plan.identity,
                plan_version=loaded.plan.plan_version,
                scenario_id=collection_scenario.scenario_id,
                scenario_identity=collection_scenario.identity,
                intervention_id=intervention.id,
                intervention_identity=intervention.identity,
                attempt_id="issue51-legacy-static-test-attempt",
                attempt_number=1,
                expected_initial_engine_state_identity=(
                    collection_scenario.expected_initial_engine_state_identity
                ),
                interface_action=intervention.interface_action,
                engine_relative_action=intervention.engine_relative_action,
                mapping_version=intervention.mapping_version,
                slingshot_reference=intervention.slingshot_reference,
            )
            with patch(
                "scripts.collect_rollouts.validate_scenario_template_constraints_workbook"
            ) as validate_workbook:
                validate_cohort_v2_constraints_authority(
                    scenario, Path(value["workbook_path"])
                )
            bindings = source_bindings_from_collection(
                scenario,
                collection_scenario,
                request,
                rollout_identity=request.attempt_id,
            )
        validate_workbook.assert_not_called()
        self.assertEqual(bindings["scenario_template_id"], scenario.template_record.identity)
        self.assertEqual(bindings["scenario_lineage_id"], value["scenario_lineage_id"])

    def test_supplementary_plan_prospectively_targets_level_clear(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            value = build_issue_51_supplementary_plan(Path(temporary))
            plan = load_collection_plan(Path(value["plan_path"])).plan
        interventions = [
            intervention
            for scenario in plan.scenarios
            for intervention in scenario.interventions
        ]
        self.assertEqual(len(interventions), 2)
        self.assertEqual({item.source for item in interventions}, {"targeted_rare", "geometry_stratified"})
        self.assertEqual(
            {item.intended_coverage_stratum for item in interventions},
            {"collision", "level clear"},
        )
        self.assertTrue(all(item.ordinal in {1, 2} for item in interventions))


class Issue51EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = _supplementary(Path(cls.temporary.name))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_micro_predicates_meet_the_declared_floor(self):
        captures = _load_issue_44(ROOT)["captures"]
        for predicate in ("contact", "supports"):
            audit = _micro_audit(captures, predicate)
            self.assertTrue(audit["passed"])
            self.assertGreaterEqual(audit["coverage"]["positive_witness_count"], 2)
            self.assertGreaterEqual(audit["coverage"]["negative_witness_count"], 2)
            self.assertGreaterEqual(audit["coverage"]["boundary_window_count"], 2)
            self.assertTrue(audit["unavailable_or_invalidation_check"]["passed"])

    def test_v2_realized_coverage_comes_from_authoritative_capture(self):
        captures = _load_issue_44(ROOT)["captures"]
        self.assertIn(
            "no-contact/miss",
            classify_physics_capture_v2_coverage(captures["no-contact"]),
        )
        collision = classify_physics_capture_v2_coverage(captures["collision"])
        self.assertIn("collision", collision)
        self.assertIn("destruction", collision)
        self.assertIn(
            "persistent support",
            classify_physics_capture_v2_coverage(captures["support"]),
        )
        self.assertIn(
            "support change",
            classify_physics_capture_v2_coverage(captures["support-change"]),
        )

    def test_source_bound_invalid_capture_is_atomically_quarantined(self):
        capture = json.loads(
            (
                ROOT / "data/runtime_evidence/issue-44/captures/collision.json"
            ).read_text(encoding="utf-8")
        )
        audit = source_bound_quarantine_audit(capture)
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["whole_attempt_quarantined"])
        self.assertTrue(audit["accepted_namespace_untouched"])
        self.assertFalse(audit["eligible_for_capability_evidence"])
        self.assertFalse(audit["retryable"])
        self.assertEqual(audit["status"], "failed")
        self.assertEqual(audit["disposition"], "quarantined")

    def test_integrated_report_is_representative_and_outcome_complete(self):
        artifacts = _derived_artifacts(
            self.root,
            ROOT,
            "issue51-test-implementation",
            "issue51-test-execution",
        )
        report = artifacts["representative-cohort-v2-pilot-report.json"]
        accounting = artifacts["attempt-accounting.json"]
        self.assertEqual(report["disposition"], "accepted")
        self.assertTrue(report["representative_audit"])
        self.assertTrue(report["capability_audit"]["passed"])
        self.assertTrue(all(
            value["passed"]
            for value in report["capability_audit"]["supported_terminations"].values()
        ))
        self.assertEqual(report["final_evaluation"]["access"], "sealed")
        self.assertFalse(report["final_evaluation"]["consumed"])
        self.assertEqual(
            set(accounting["counts"]),
            {"planned", "accepted", "rejected", "failed", "quarantined", "retried"},
        )
        self.assertEqual(accounting["counts"]["rejected"], 0)
        self.assertEqual(accounting["counts"]["quarantined"], 3)
        self.assertEqual(accounting["counts"]["failed"], 3)
        self.assertEqual(accounting["counts"]["retried"], 0)
        self.assertEqual(accounting["unavailable"], [])
        self.assertEqual(accounting["unmet"], [])
        self.assertEqual(
            accounting["prior_determinations"][0]["identity"],
            PRIOR_DETERMINATION_IDENTITY,
        )
        self.assertEqual(
            accounting["prior_determinations"][1]["identity"],
            FAILED_RUNTIME_DETERMINATION_IDENTITY,
        )
        self.assertEqual(
            sum(
                attempt["source"] == "issue-51-determination-1"
                for attempt in accounting["attempts"]
            ),
            2,
        )
        self.assertEqual(
            sum(
                attempt["source"] == "issue-51-determination-2"
                for attempt in accounting["attempts"]
            ),
            2,
        )

    def test_publication_round_trips_through_exact_revalidation(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "scripts.build_issue_51_evidence._require_clean_tracked_worktree"
            ),
            patch(
                "scripts.build_issue_51_evidence._implementation_revision",
                return_value="issue51-test-implementation",
            ),
            patch(
                "scripts.build_issue_51_evidence._execution_revision",
                return_value="issue51-test-execution",
            ),
        ):
            output = Path(temporary) / "issue-51"
            result = build_issue_51_evidence(
                output,
                self.root / "supplementary",
                repository_root=ROOT,
            )
            self.assertTrue(result["passed"])
            self.assertTrue(result["representative_audit"])
            self.assertTrue((output / "bundle-manifest.json").is_file())

    def test_supplementary_sources_fail_without_level_clear(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _supplementary(Path(temporary), level_clear=False)
            with self.assertRaisesRegex(Issue51EvidenceError, "realized coverage is incomplete"):
                _supplementary_sources(root)

    def test_supplementary_sources_reject_stale_intervention_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _supplementary(Path(temporary))
            runtime_path = root / "supplementary/runtime-authority.json"
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            runtime["attempts"][0]["intervention_identity"] = "stale-intervention"
            _write_json(runtime_path, runtime)
            with self.assertRaisesRegex(Issue51EvidenceError, "attempt record is stale"):
                _supplementary_sources(root)

    def test_supplementary_sources_reject_rewritten_failed_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _supplementary(Path(temporary))
            failure_path = (
                root
                / "supplementary/prior-failures/level-clear-targeted-failure.json"
            )
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            failure["reason"] = "rewritten failure"
            _write_json(failure_path, failure)
            with self.assertRaisesRegex(
                Issue51EvidenceError, "failed determination-2 attempt is stale"
            ):
                _supplementary_sources(root)


if __name__ == "__main__":
    unittest.main()
