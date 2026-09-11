from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts import issue_76_fit_budget as costs
from scripts import run_issue_76_native_refit as fit


class NativeRefitControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_paired_update_horizons_and_all_hybrid_pairs(self):
        hybrid = [fit.pair_for_update(False, i) for i in range(18)]
        pure = [fit.pair_for_update(True, i) for i in range(18)]
        self.assertEqual([p.delta for p in hybrid], [p.delta for p in pure])
        self.assertEqual(set(hybrid), set(fit.PAIRS))

    def test_nontraining_shard_cannot_enter_an_optimizer(self):
        with self.assertRaisesRegex(ValueError, "cannot enter an optimizer"):
            fit.load_shard(Path("unused"), {"exposure_role": "model_selection"}, {}, fitting=True)

    def test_budget_resume_keeps_consumed_time_and_resource_stop_is_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clock = SimpleNamespace(now=0.)
            with patch.object(costs.time, "monotonic", side_effect=lambda: clock.now), patch.object(costs, "process_rss", return_value=10):
                with costs.fitting_budget(root, "job", 5, "cpu") as budget:
                    clock.now = 2
                    budget.check()
                clock.now = 100
                with costs.fitting_budget(root, "job", 5, "cpu") as budget:
                    clock.now = 101
                    budget.check()
                self.assertEqual(costs.files.read(root / "budgets/job.json")["active_seconds"], 3)
                with self.assertRaises(costs.FitBudgetExceeded):
                    with costs.fitting_budget(root, "job", 5, "cpu") as budget:
                        clock.now = 104
                        budget.check()
                with self.assertRaises(costs.FitBudgetExceeded):
                    costs.FitBudget(root, "job", 5, "cpu")

    def test_clean_pause_resumes_exact_optimizer_counter_without_replaying_update(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(costs, "process_rss", return_value=10):
            root = Path(temporary)
            torch.manual_seed(76)
            reference = torch.nn.Linear(2, 1)
            initial = {k: v.clone() for k, v in reference.state_dict().items()}
            def run(model, destination, key, pause):
                with costs.fitting_budget(root, key, 30, "cpu") as budget:
                    def loss(step):
                        if pause and step == 1:
                            budget.pause_requested = True
                        return model(torch.tensor([[step + 1., 1.]])).square().mean()
                    return costs.fit_updates(destination, model, plan_identity="fixture", updates=3,
                                             lr=.001, make_loss=loss, budget=budget, device="cpu")
            run(reference, root / "reference.pt", "reference", False)
            interrupted = torch.nn.Linear(2, 1)
            interrupted.load_state_dict(initial)
            with self.assertRaises(costs.FitPaused):
                run(interrupted, root / "paused.pt", "paused", True)
            partial = torch.load(root / "paused.pt", weights_only=False)
            self.assertEqual(partial["updates_completed"], 2)
            self.assertFalse(partial["complete"])
            resumed = torch.nn.Linear(2, 1)
            complete = run(resumed, root / "paused.pt", "paused", False)
            self.assertEqual(complete["updates_applied"], 3)
            for key, value in reference.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)

    def test_unclean_job_cannot_reset_its_allowance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            costs.files.write(root / "budgets/job.json", {"active_seconds": 1, "limit_seconds": 30, "stopped": False, "running": True})
            with self.assertRaisesRegex(costs.FitBudgetExceeded, "unclean interrupted"):
                costs.FitBudget(root, "job", 30, "cpu")

    def test_corrupt_preparation_is_retained_without_replacement_or_budget_reset(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(costs, "process_rss", return_value=10):
            root = Path(temporary) / "fit"
            capture = Path(temporary) / "capture"
            member = {"identity": "assigned", "base_cluster": "assigned", "exposure_role": "training",
                      "generator_family": "fixture"}
            plan = {"identity": "fixture", "members": [member], "collection_root": str(capture),
                    "limits": {"working_bytes": 2**30}, "minimum_usable_fraction_per_role_family": .9}
            costs.files.write(capture / "budget.json", {"stopped": False, "active_seconds": 100})
            raw_result = {"complete": True, "member_identity": "assigned", "gameplay_success": False}
            costs.files.write(fit.collection.c2.result_path(capture, member), raw_result)
            with patch.object(fit, "prepare_episode", side_effect=ValueError("fixture corrupt trace")) as derive:
                fit.prepare_data(root, plan)
                index = costs.files.read(root / "data-index.json")
                self.assertFalse(index["fit_data_gate_passed"])
                self.assertFalse(index["entries"][0]["usable"])
                self.assertIn("fixture corrupt", index["entries"][0]["preparation_failure"])
                budget = (root / "budgets/data-preparation.json").read_bytes()
                fit.prepare_data(root, plan)
                self.assertEqual(derive.call_count, 1)
                self.assertEqual((root / "budgets/data-preparation.json").read_bytes(), budget)
                self.assertEqual(costs.files.read(fit.collection.c2.result_path(capture, member)), raw_result)


if __name__ == "__main__":
    unittest.main()
