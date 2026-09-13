from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_issue_76_automated_refit as execution
from scripts import issue_76_fit_budget as costs


class AutomatedExecutionTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory) / "refit"
        collection = Path(directory) / "collection"
        plan = {"identity": "fixture", "seeds": [1], "collection_root": str(collection),
                "members": [{"identity": "member"}]}
        costs.files.write(collection / "results/member.json", {"complete": True})
        costs.files.write(collection / "attempts/member/trace.json", {"original": True})
        attempt_bytes = sum(p.stat().st_size for p in (collection / "attempts").rglob("*") if p.is_file())
        costs.files.write(collection / "budget.json", {"artifact_bytes": attempt_bytes, "active_seconds": 40000})
        costs.files.write(root / "engineering-cost-reservation.json", {"unchanged": True})
        for key in execution.budget_keys(plan):
            costs.files.write(root / "budgets" / (key + ".json"), {
                "active_seconds": 7., "engineering_seconds": 2., "limit_seconds": 30.,
                "peak_cpu_rss_mib": 0., "peak_cuda_allocated_mib": 0.,
                "peak_cuda_reserved_mib": 0., "running": False, "stopped": False})
        return root, plan

    def prepare(self, root, plan):
        with patch.object(execution.engineering, "reservation", return_value={"unchanged": True}):
            return execution.prepare_amendment(root, plan)

    def test_amendment_retains_prior_cost_and_extends_both_arms_identically(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan = self.fixture(directory)
            amendment = self.prepare(root, plan)
            self.assertEqual(execution.load_amendment(root, plan), amendment)
            for key in execution.budget_keys(plan):
                value = costs.files.read(root / "budgets" / (key + ".json"))
                self.assertEqual(value["active_seconds"], 7)
                self.assertEqual(value["engineering_seconds"], 2)
                self.assertEqual(value["limit_seconds"], 86400)
                self.assertEqual(amendment["before_budgets"][key]["limit_seconds"], 30)
            with self.assertRaisesRegex(ValueError, "already exists"):
                self.prepare(root, plan)

    def test_running_or_stopped_budget_cannot_be_silently_amended(self):
        for flag in ("running", "stopped"):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as directory:
                root, plan = self.fixture(directory)
                path = root / "budgets/data-preparation.json"
                previous = costs.files.read(path)
                costs.files.write(path, {**previous, flag: True})
                with self.assertRaisesRegex(ValueError, "audit"):
                    self.prepare(root, plan)
                self.assertFalse((root / execution.AMENDMENT).exists())

    def test_exact_size_without_revisiting_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan = self.fixture(directory)
            amendment = self.prepare(root, plan)
            original_size = execution.fit.working_bytes
            expected = original_size(root, plan)
            original_glob = Path.rglob
            def reject_attempt_scan(path, pattern):
                if path == Path(plan["collection_root"]):
                    self.fail("the execution policy rescanned the captured corpus")
                return original_glob(path, pattern)
            with execution.execution_policy(root, amendment), patch.object(Path, "rglob", reject_attempt_scan):
                self.assertEqual(execution.fit.working_bytes(root, plan), expected)
            self.assertIs(execution.fit.working_bytes, original_size)

    def test_original_budget_context_resumes_consumed_time_under_declared_limit(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(costs, "process_rss", return_value=10):
            root, plan = self.fixture(directory)
            amendment = self.prepare(root, plan)
            original = execution.fit.fitting_budget
            with execution.execution_policy(root, amendment):
                with execution.fit.fitting_budget(root, "data-preparation", 30, "cpu") as budget:
                    self.assertEqual(budget.seconds, 86400)
                    self.assertEqual(budget.initial, 7)
                    budget.check()
            self.assertIs(execution.fit.fitting_budget, original)
            self.assertGreaterEqual(costs.files.read(root / "budgets/data-preparation.json")["active_seconds"], 7)

    def test_changed_source_or_reset_cost_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan = self.fixture(directory)
            self.prepare(root, plan)
            path = root / "budgets/data-preparation.json"
            costs.files.write(path, {**costs.files.read(path), "active_seconds": 0})
            with self.assertRaisesRegex(ValueError, "prior cost"):
                execution.load_amendment(root, plan)


if __name__ == "__main__":
    unittest.main()
