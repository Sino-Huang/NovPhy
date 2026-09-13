from collections import Counter, defaultdict
from copy import deepcopy
import unittest

import torch

from scripts import run_issue_76_boundary_focus as run
from scripts.validate_issue_76_boundary_focus import check_checkpoint


class BoundaryFocusTests(unittest.TestCase):
    def test_all_boundaries_and_paired_exposure_with_unusable_assignments(self):
        starts = [[0, 12, 24] if i not in (56, 132) else [] for i in range(250)]
        totals = []
        for pure in (False, True):
            schedule = run.schedule(starts, 760930001, pure, 13800, 1800)
            self.assertEqual(len(schedule), 1785)
            exposures = defaultdict(Counter)
            for update, rows in schedule.items():
                self.assertTrue(starts[update % 250])
                self.assertEqual(tuple(rows.shape), (32, 2))
                self.assertTrue(all(start in starts[entry] for entry, start in rows.tolist()))
                exposures[run.fit.pair_for_update(pure, update).delta].update(map(tuple, rows.tolist()))
            totals.append(exposures)
        self.assertEqual(totals[0], totals[1])

    def test_initial_state_and_validator_retain_optimizer_history(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.00003)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        before = dict(plan_identity="source", complete=True, failure=None, updates_requested=6000,
                      updates_completed=6000, updates_applied=5952, updates_skipped=48,
                      model=model.state_dict(), optimizer=deepcopy(optimizer.state_dict()))
        plan = dict(identity="new", source_plan_identity="source", learning_rate=.00003,
                    updates={"predictor": 1800}, start_update=13800,
                    training=[{"usable": i not in (56, 132)} for i in range(250)])
        initial = run.initial_checkpoint(before, plan)
        self.assertIs(initial["model"], before["model"])
        self.assertIs(initial["optimizer"], before["optimizer"])
        self.assertEqual(initial["updates_completed"], 0)
        self.assertEqual(before["updates_completed"], 6000)
        saved = {**initial, "complete": True, "updates_completed": 1800, "updates_applied": 1785,
                 "updates_skipped": 15, "optimizer": deepcopy(before["optimizer"])}
        for state in saved["optimizer"]["state"].values():
            state["step"] += 1785
        check_checkpoint(saved, before, model, plan, True)
        saved["optimizer"]["param_groups"][0]["lr"] = .0003
        with self.assertRaisesRegex(ValueError, "settings differ"):
            check_checkpoint(saved, before, model, plan, True)
        saved["optimizer"]["param_groups"][0]["lr"] = .00003
        for state in saved["optimizer"]["state"].values():
            state["step"] -= 1
        with self.assertRaisesRegex(ValueError, "optimizer history differs"):
            check_checkpoint(saved, before, model, plan, True)

    def test_stronger_lower_rate_anchors_are_not_discarded(self):
        plan = dict(seeds=[1], evaluation=list(range(5)), unchanged_mse_ratio=.8,
                    required_unchanged_wins=4, retention_ratio=1.1)
        rows = []
        for pure in (False, True):
            for member in range(5):
                row = dict(seed=1, pure=pure, member_identity=member)
                for kind in ("boundary_focus", "lower_rate", "full_duration", "boundary", "reference"):
                    mse = .06 if kind == "boundary_focus" else .04 if kind == "lower_rate" else .1
                    row[kind] = {run.fit.policy_name(p): [dict(elapsed=t, available=True, recursive_mse=mse,
                        unchanged_mse=1.) for t in (750, 11250)] for p in (run.fit.CONTINUOUS_PAIRS if pure else run.fit.PAIRS)}
                rows.append(row)
        for summary in run.summarize(plan, rows):
            self.assertTrue(summary["initial_accuracy_retained"])
            self.assertTrue(summary["endpoint_accuracy_retained"])
            self.assertFalse(summary["lower_rate_anchor_accuracy_retained"])
            self.assertFalse(summary["qualification_supported"])
        self.assertEqual(rows[0]["lower_rate"]["fixed-750-micro"][0]["recursive_mse"], .04)


if __name__ == "__main__":
    unittest.main()
