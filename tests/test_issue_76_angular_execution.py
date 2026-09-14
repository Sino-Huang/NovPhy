from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import run_issue_76_angular_replay as angular


class AngularExecutionTests(unittest.TestCase):
    def test_failed_comparison_is_preserved_and_does_not_authorize(self):
        members = [{"identity": "reference"}, {"identity": "candidate"}]
        plan = {"identity": "angular", "comparability_limits": {}}
        written = {}
        with patch.object(angular, "load_plan", return_value=plan), \
             patch.object(angular.supervisor.files, "read", return_value={"running": False, "stopped": False}), \
             patch.object(angular.supervisor, "selected_members", return_value=members), \
             patch.object(angular.validation, "read_case", return_value={"decision_image_path": "image"}), \
             patch.object(angular.validation.comparison, "compare", return_value={"comparable": False, "failures": ["bird_motion"]}), \
             patch.object(angular.validation, "immutable", side_effect=lambda path, value: written.update({str(path): value})):
            angular.validate_smoke(Path("private"), Path("public"))
        report = written["private/smoke-validation.json"]
        self.assertFalse(report["validated"])
        self.assertEqual(report["comparability"]["failures"], ["bird_motion"])
        self.assertFalse(report["advancement_authorized"])
        self.assertEqual(report, written["public/smoke-validation.json"])

    def test_full_run_requires_angular_smoke_not_original_smoke(self):
        plan = {"identity": "angular", "inventory": {"members": [], "smoke_member_identities": ["new-ref", "new-action"]}}
        with TemporaryDirectory() as directory, patch.object(angular.supervisor.files, "read", return_value={
                "validated": True, "execution_plan_identity": "original", "member_identities": ["old-ref", "old-action"]}):
            with self.assertRaisesRegex(ValueError, "exact two-replay smoke"):
                angular.supervisor.selected_members(plan, Path(directory), False)
