from copy import deepcopy
import unittest

import torch

from scripts import run_issue_76_endpoint_anchor as run
from scripts.validate_issue_76_endpoint_anchor import check_checkpoint


class EndpointAnchorStageTests(unittest.TestCase):
    def test_stronger_local_anchor_references_are_retained(self):
        plan = dict(seeds=[1], evaluation=list(range(5)), unchanged_mse_ratio=.8,
                    required_unchanged_wins=4, retention_ratio=1.1)
        rows = []
        for pure in (False, True):
            for member in range(5):
                row = dict(seed=1, pure=pure, member_identity=member)
                for kind in ("endpoint_anchor", "local_anchor", "boundary_focus", "lower_rate", "full_duration", "boundary", "reference"):
                    mse = .06 if kind == "endpoint_anchor" else .04 if kind == "local_anchor" else .1
                    row[kind] = {run.fit.policy_name(p): [dict(elapsed=t, available=True, recursive_mse=mse,
                        unchanged_mse=1.) for t in (750, 11250)] for p in (run.fit.CONTINUOUS_PAIRS if pure else run.fit.PAIRS)}
                rows.append(row)
        for summary in run.summarize(plan, rows):
            self.assertTrue(summary["lower_rate_anchor_accuracy_retained"])
            self.assertTrue(summary["boundary_focus_anchor_accuracy_retained"])
            self.assertFalse(summary["local_anchor_accuracy_retained"])
            self.assertFalse(summary["qualification_supported"])
        for row in rows:
            row["endpoint_anchor"] = deepcopy(row["local_anchor"])
        self.assertTrue(all(s["qualification_supported"] for s in run.summarize(plan, rows)))

    def test_validator_rejects_optimizer_reset_or_changed_learning_rate(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.00003)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        before = dict(plan_identity="source", complete=True, failure=None, updates_requested=1800,
                      updates_completed=1800, updates_applied=1785, updates_skipped=15,
                      model=model.state_dict(), optimizer=deepcopy(optimizer.state_dict()))
        plan = dict(identity="new", source_plan_identity="source", learning_rate=.00003,
                    updates={"predictor": 900}, start_update=15600,
                    training=[{"usable": i not in (56, 132)} for i in range(250)])
        saved = {**run.initial_checkpoint(before, plan), "complete": True, "updates_completed": 900,
                 "updates_applied": 893, "updates_skipped": 7, "optimizer": deepcopy(before["optimizer"])}
        for state in saved["optimizer"]["state"].values():
            state["step"] += 893
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
