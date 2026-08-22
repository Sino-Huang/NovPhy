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
    Issue51EvidenceError,
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
from scripts.collection_plan import load_collection_plan
from scripts.collect_rollouts import classify_physics_capture_v2_coverage
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
            "scenario_template_id": manifest["scenario_template"]["identity"],
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
    issue50_runtime = json.loads(
        (
            ROOT
            / "data/runtime_evidence/issue-50/source-probes/runtime-bundle-manifest.json"
        ).read_text(encoding="utf-8")
    )
    source_commit = issue50_runtime["source_snapshot_commit"]
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
    return root


class Issue51PilotPlanTests(unittest.TestCase):
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
        self.assertEqual(accounting["counts"]["quarantined"], 1)
        self.assertEqual(accounting["counts"]["failed"], 1)
        self.assertEqual(accounting["counts"]["retried"], 0)
        self.assertEqual(accounting["unavailable"], [])
        self.assertEqual(accounting["unmet"], [])

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


if __name__ == "__main__":
    unittest.main()
