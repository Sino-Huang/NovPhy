from copy import deepcopy
import unittest

import torch

from scripts.validate_issue_76_lower_rate_dynamics import check_checkpoint


class LowerRateValidationTests(unittest.TestCase):
    def test_wrong_loaded_rate_or_reset_steps_are_rejected(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        before = dict(plan_identity="source", complete=True, failure=None, updates_completed=1800,
                      updates_applied=1786, updates_skipped=14, optimizer=deepcopy(optimizer.state_dict()))
        plan = dict(identity="new", source_plan_identity="source", learning_rate=.00003,
                    updates={"predictor": 6000}, start_update=7800,
                    training=[{"usable": i not in (1, 2)} for i in range(250)])
        saved = dict(plan_identity="new", complete=True, failure=None, updates_requested=6000,
                     updates_completed=6000, updates_applied=5952, updates_skipped=48,
                     optimizer=deepcopy(before["optimizer"]))
        saved["optimizer"]["param_groups"][0]["lr"] = .00003
        for state in saved["optimizer"]["state"].values():
            state["step"] += 5952
        check_checkpoint(saved, before, model, plan, True)
        saved["optimizer"]["param_groups"][0]["lr"] = .0003
        with self.assertRaisesRegex(ValueError, "settings differ"):
            check_checkpoint(saved, before, model, plan, True)
        saved["optimizer"]["param_groups"][0]["lr"] = .00003
        for state in saved["optimizer"]["state"].values():
            state["step"] -= 1
        with self.assertRaisesRegex(ValueError, "optimizer history differs"):
            check_checkpoint(saved, before, model, plan, True)


if __name__ == "__main__":
    unittest.main()
