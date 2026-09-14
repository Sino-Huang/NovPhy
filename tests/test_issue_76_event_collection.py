from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import validate_issue_76_event_collection as audit


class EventCollectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = dict(identity="source", base_cluster="base", exposure_role="training", generator_family="family")
        self.assignments = [dict(identity=f"case-{i}", source_member_identity="source", ordinal=i + 1,
            candidate_ordinal=i, original_action_reference=i == 0, action={"drag_x": -80}) for i in range(3)]
        self.plan = {"identity": "execution", "inventory": {"source_members": [self.source],
            "assignments": self.assignments, "limits": {"native_step_seconds": .0004}}, "comparability_limits": {}}
        self.segment = {"summary": {"first_fixed_step": 10, "last_fixed_step": 20,
                                   "observed_window_valid": True, "censored": False, "terminal_observed": True}}
        for row in self.assignments:
            audit.run.files.write(self.root / "results" / (row["identity"] + ".json"),
                                  {"complete": True, "failure": None, "segments": [self.segment]})
            audit.run.files.write(self.root / "supervision" / (row["identity"] + ".json"),
                                  {"member_identity": row["identity"], "attempt_artifact_bytes": 1})
        self.completed = {row["identity"]: 1 for row in self.assignments}
        self.case = {"decision_time": 1., "capture_time": 2., "decision_image_path": "decision.png"}

    def group(self, failures=(), comparable=True):
        def read_case(root, plan, member):
            if member["identity"] in failures:
                raise ValueError("invalid observation")
            return self.case
        with patch.object(audit.run.validation, "read_case", side_effect=read_case), patch.object(
                audit.run, "terminal_evidence", return_value={"reason": "level_clear", "fixed_step": 20}), patch.object(
                audit.run.previous.comparison, "compare", return_value={"comparable": comparable, "failures": []}):
            return audit.audit_group(self.root, self.plan, self.source, self.assignments, self.completed, lambda: None)

    def test_comparability_failure_does_not_invalidate_individual_targets(self):
        group = self.group(comparable=False)
        self.assertFalse(group["paired_ranking_admissible"])
        self.assertTrue(all(row["capture_contract_validated"] for row in group["rows"]))
        self.assertTrue(all(row["target"]["native_level_clear"] for row in group["rows"]))

    def test_invalid_reference_invalidates_whole_pair_group_not_other_targets(self):
        group = self.group(failures=("case-0",))
        self.assertFalse(group["paired_ranking_admissible"])
        self.assertTrue(all(not row["comparable"] for row in group["comparisons"]))
        self.assertIsNone(group["rows"][0]["target"])
        self.assertTrue(group["rows"][1]["capture_contract_validated"])
        self.assertIn("invalid observation", group["rows"][0]["validation_error"])

    def test_missing_capture_stays_in_inventory_without_negative_target(self):
        # Use an assigned identity with no files, without deleting any fixture.
        self.assignments[2]["identity"] = "unattempted"
        group = self.group()
        self.assertEqual(len(group["rows"]), 3)
        row = group["rows"][2]
        self.assertIsNone(row["target"])
        self.assertIsNone(row["recorded_complete"])
        self.assertFalse(row["capture_contract_validated"])
        self.assertFalse(group["paired_ranking_admissible"])

    def test_live_or_incomplete_collection_rejected_before_validation_starts(self):
        for budget in ({"running": True, "active_members": []},
                       {"running": False, "active_members": [], "stopped": False, "completed_attempt_bytes": {}}):
            audit.run.files.write(self.root / "capture-budget.json", budget)
            with patch.object(audit.run, "load_plan", return_value=self.plan):
                with self.assertRaises(ValueError):
                    audit.validate(self.root, self.root / "public")
            self.assertFalse((self.root / "collection-validation-start.json").exists())

    def test_budget_check_is_not_converted_to_an_unavailable_capture(self):
        def exhausted():
            raise RuntimeError("offline allowance")
        with self.assertRaisesRegex(RuntimeError, "offline allowance"):
            audit.audit_group(self.root, self.plan, self.source, self.assignments, self.completed, exhausted)

    def test_unsealed_or_mismatched_accounting_cannot_supply_targets(self):
        self.completed.pop("case-0")
        self.completed["case-1"] = 2
        group = self.group()
        self.assertEqual([row["capture_contract_validated"] for row in group["rows"]], [False, False, True])

    def test_full_report_preserves_resource_stop_and_refuses_an_implicit_rerun(self):
        self.plan["inventory"]["limits"].update(offline_validation_wall_seconds=100,
            collection_wall_seconds=100, aggregate_cpu_rss_mib=100, artifact_bytes=100)
        budget = {"running": False, "active_members": [], "stopped": True, "active_seconds": 100,
                  "smoke_active_seconds": 1, "peak_cpu_rss_mib": 1, "artifact_bytes": 3,
                  "completed_attempt_bytes": self.completed}
        audit.run.files.write(self.root / "capture-budget.json", budget)
        audit.run.files.write(self.root / "smoke-validation.json",
            {"validated": True, "execution_plan_identity": "execution", "validation_wall_seconds": 2})
        group = self.group()
        with patch.object(audit.run, "load_plan", return_value=self.plan), patch.object(
                audit, "audit_group", return_value=group), patch.object(audit, "data_readiness",
                return_value={"data_ready": True, "supports_separate_training_protocol_preparation": True}):
            report = audit.validate(self.root, self.root / "public")
            self.assertFalse(report["data_readiness"]["data_ready"])
            self.assertEqual(report["collection_resource_failure"], "collection_wall_limit")
            self.assertEqual(report["prior_smoke_validation_seconds"], 2)
            self.assertFalse(report["model_training_authorized"])
            self.assertTrue((self.root / "public/collection-validation.json").exists())
            with self.assertRaisesRegex(ValueError, "already attempted"):
                audit.validate(self.root, self.root / "public")
