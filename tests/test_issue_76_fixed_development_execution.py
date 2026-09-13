from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import run_issue_76_fixed_development as run
from tests.test_issue_76_fixed_development_scoring import selection_fixture


class FixedDevelopmentExecutionTests(unittest.TestCase):
    def test_cost_ledger_counts_canonical_jobs_not_mirrors_or_reservations(self):
        with TemporaryDirectory() as directory:
            project = Path(directory)
            first = project / ".local-artifacts/issue-76-native-refit-v2"
            second = project / ".local-artifacts/issue-76-test-v1"
            public = project / "data/issue-76-test"
            run.fit.files.write(first / "budgets/common-1.json", {"active_seconds": 10., "stopped": False})
            # Failed/nonselected work remains charged.
            receipt = {"active_seconds": 3., "stopped": True}
            run.fit.files.write(second / "budgets/validation.json", receipt)
            run.fit.files.write(public / "validation-budget.json", receipt)
            run.fit.files.write(first / "engineering-cost-reservation.json", {"already_debited": 2.})
            ledger = run.prior_cost_ledger(project)
            self.assertEqual(len(ledger["receipts"]), 2)
            self.assertEqual(ledger["recorded_active_seconds_sum"], 13.)
            self.assertEqual(len(ledger["public_mirrors_not_added"]), 1)
            run.fit.files.write(public / "validation-budget.json", {"active_seconds": 7.})
            with self.assertRaises(ValueError):
                run.prior_cost_ledger(project)

    def test_fresh_roles_and_model_selection_without_freeze_are_rejected(self):
        with self.assertRaises(ValueError):
            run.phase_path("fresh", "anything.json")
        plan = {"roles": list(run.ROLES)}
        with self.assertRaises(ValueError):
            run.check_role_access(plan, "fresh")
        with patch.object(run.fit.files, "read", side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):
                run.check_role_access(plan, "model_selection")
        run.check_role_access(plan, "calibration")

    def test_model_selection_requires_reproducible_common_calibration_choice(self):
        inventory, results = selection_fixture()
        plan = {"identity": "execution", "roles": list(run.ROLES), "inventory": inventory}
        frozen = {"execution_plan_identity": "execution", "choice": run.scoring.calibration_choice(inventory, results)}
        with patch.object(run, "load_results", return_value=results), patch.object(run.fit.files, "read", return_value=frozen):
            run.check_role_access(plan, "model_selection")
            frozen["choice"]["selected"]["hybrid"]["recipe"] = "different"
            with self.assertRaises(ValueError):
                run.check_role_access(plan, "model_selection")

    def test_prepared_role_or_member_cannot_be_reassigned(self):
        inventory, _ = selection_fixture()
        plan = {"identity": "execution", "inventory": inventory}
        prepared = {"execution_plan_identity": "execution", "seed": 1, "role": "calibration",
                    "members": [{"entry": entry} for entry in inventory["entries"]]}
        with patch.object(run.torch, "load", return_value=prepared):
            self.assertEqual(len(run.load_prepared(plan, "calibration", 1)), 3)
            with self.assertRaises(ValueError):
                run.load_prepared(plan, "model_selection", 1)
            with self.assertRaises(ValueError):
                run.load_prepared(plan, "calibration", 2)
            prepared["members"].reverse()
            with self.assertRaises(ValueError):
                run.load_prepared(plan, "calibration", 1)

    def test_selection_freeze_requires_validation_and_precedes_model_selection(self):
        inventory, results = selection_fixture()
        plan = {"identity": "execution", "inventory": inventory}
        with TemporaryDirectory() as directory, patch.object(run, "ROOT", Path(directory)), \
                patch.object(run, "OUTPUT", Path(directory) / "public"), patch.object(run, "load_results", return_value=results):
            path = run.phase_path("calibration", "validation.json")
            validation = {"validated": False, "execution_plan_identity": "execution",
                          "cells_validated": 54, "all_assigned_members_checked": True}
            run.fit.files.write(path, validation)
            with self.assertRaises(ValueError):
                run.freeze_choice(plan)
            validation["validated"] = True
            run.fit.files.write(path, validation)
            run.freeze_choice(plan)
            first = run.fit.files.read(run.ROOT / "calibration-choice.json")
            run.freeze_choice(plan)
            self.assertEqual(first, run.fit.files.read(run.ROOT / "calibration-choice.json"))
            self.assertFalse(first["model_selection_started"])

    def test_source_freeze_rejects_changed_scoring_source(self):
        with TemporaryDirectory() as directory:
            project = Path(directory)
            # A source file fixture is produced by the test, not a repository edit.
            source = project / "source.py"
            source.write_text("original\n")
            plan = dict(score_execution_authorized=True, fresh_access=False, optimization_performed=False,
                        inventory={}, source_text={"source.py": "different\n"})
            with patch.object(run.fit.files, "ROOT", project), patch.object(run.fit.files, "read", side_effect=[plan, {}]):
                with self.assertRaises(ValueError):
                    run.load_plan()


if __name__ == "__main__":
    unittest.main()
