from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from scripts import run_issue_76_lower_rate_dynamics as run


class LowerRateTests(unittest.TestCase):
    def test_loaded_rate_changes_but_moments_and_source_are_retained(self):
        torch.set_num_threads(1)
        torch.manual_seed(76)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
        x = torch.ones(2, 2)
        for _ in range(2):
            optimizer.zero_grad()
            model(x).square().mean().backward()
            optimizer.step()
        old = dict(plan_identity="source", complete=True, failure=None, updates_completed=1800,
                   updates_applied=1786, updates_skipped=14, model=deepcopy(model.state_dict()),
                   optimizer=deepcopy(optimizer.state_dict()))
        plan = dict(identity="next", source_plan_identity="source", updates={"predictor": 2}, learning_rate=.00003)
        initial = run.initial_checkpoint(old, plan)
        self.assertEqual(old["optimizer"]["param_groups"][0]["lr"], .0003)
        self.assertEqual(initial["optimizer"]["param_groups"][0]["lr"], .00003)
        self.assertEqual(initial["updates_completed"], 0)
        self.assertFalse(initial["complete"])
        for key, state in old["optimizer"]["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(initial["optimizer"]["state"][key][name], value, rtol=0, atol=0)
        optimizer.load_state_dict(deepcopy(old["optimizer"]))
        for group in optimizer.param_groups:
            group["lr"] = .00003
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(x).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictor.pt"
            run.fit.atomic_torch(path, initial)
            actual = torch.nn.Linear(2, 1)
            with run.fit.fitting_budget(Path(folder), "fixture", 30, "cpu") as budget:
                result = run.fit.fit_updates(path, actual, plan_identity="next", updates=2, lr=.0003,
                    make_loss=lambda _: actual(x).square().mean(), budget=budget, device="cpu")
            for key, value in model.state_dict().items():
                torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)
            self.assertEqual(result["optimizer"]["param_groups"][0]["lr"], .00003)
            self.assertTrue(all(float(s["step"]) == 4 for s in result["optimizer"]["state"].values()))

    def test_newest_pure_reference_cannot_be_weakened(self):
        plan = dict(seeds=[1], evaluation=list(range(5)), unchanged_mse_ratio=.8,
                    required_unchanged_wins=4, retention_ratio=1.1)
        rows = []
        for pure in (False, True):
            for member in range(5):
                row = dict(seed=1, pure=pure, member_identity=member)
                for kind in ("lower_rate", "full_duration", "boundary", "reference"):
                    mse = .06 if kind == "lower_rate" else .04 if kind == "full_duration" else .1
                    row[kind] = {run.fit.policy_name(p): [dict(elapsed=t, available=True, recursive_mse=mse,
                        unchanged_mse=1.) for t in (750, 11250)] for p in (run.fit.CONTINUOUS_PAIRS if pure else run.fit.PAIRS)}
                rows.append(row)
        hybrid, pure = run.summarize(plan, rows)
        self.assertTrue(hybrid["qualification_supported"])
        self.assertTrue(pure["endpoint_accuracy_retained"])
        self.assertFalse(pure["full_duration_pure_accuracy_retained"])
        self.assertFalse(pure["qualification_supported"])
        self.assertIn("lower_rate", hybrid["anchor_metrics"])
        self.assertEqual(rows[0]["full_duration"]["fixed-50-continuous"][0]["recursive_mse"], .04)


if __name__ == "__main__":
    unittest.main()
