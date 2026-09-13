from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from scripts import run_issue_76_full_duration_dynamics as run


class FullDurationStageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(76)

    def test_stage_initialization_retains_optimizer_history_and_starts_new_counter(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
        x, target = torch.ones(2, 2), torch.zeros(2, 1)
        for _ in range(2):
            optimizer.zero_grad()
            (model(x) - target).square().mean().backward()
            optimizer.step()
        old = dict(plan_identity="parent", complete=True, failure=None, updates_completed=6000,
                   updates_applied=5952, updates_skipped=48, model=deepcopy(model.state_dict()),
                   optimizer=deepcopy(optimizer.state_dict()))
        plan = dict(identity="new-stage", source_plan_identity="parent", updates={"predictor": 2})
        initial = run.initial_checkpoint(old, plan)
        self.assertEqual(initial["updates_completed"], 0)
        self.assertFalse(initial["complete"])
        self.assertEqual(old["updates_completed"], 6000)
        for key in old["optimizer"]["state"]:
            self.assertTrue(torch.equal(initial["optimizer"]["state"][key]["exp_avg"], old["optimizer"]["state"][key]["exp_avg"]))
        expected = torch.nn.Linear(2, 1)
        expected.load_state_dict(old["model"])
        expected_optimizer = torch.optim.AdamW(expected.parameters(), lr=.0003)
        expected_optimizer.load_state_dict(deepcopy(old["optimizer"]))
        for _ in range(2):
            expected_optimizer.zero_grad(set_to_none=True)
            (expected(x) - target).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.)
            expected_optimizer.step()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "progress.pt"
            run.fit.atomic_torch(path, initial)
            actual = torch.nn.Linear(2, 1)
            seen = []

            def loss(update):
                seen.append(update)
                return (actual(x) - target).square().mean()

            with run.fit.fitting_budget(Path(folder), "fixture", 30, "cpu") as budget:
                result = run.fit.fit_updates(path, actual, plan_identity=plan["identity"], updates=2,
                                             lr=.0003, make_loss=loss, budget=budget, device="cpu")
            self.assertEqual(seen, [0, 1])
            self.assertEqual(result["updates_applied"], 2)
            for key, value in actual.state_dict().items():
                self.assertTrue(torch.equal(value, expected.state_dict()[key]))
            self.assertTrue(all(float(s["step"]) == 4 for s in result["optimizer"]["state"].values()))

    def test_absolute_schedule_and_paired_horizon_exposure_are_preserved(self):
        lengths, starts = [40, 0, 53, 61, 80], [[0, 20], [], [0, 30], [0], [0, 40]]
        rng = torch.random.get_rng_state().clone()
        totals = []
        original = run.source.previous.sampled_schedule(lengths, 761, False, 6120)
        original = {u: rows for u, rows in original.items() if u >= 6000}
        changed = run.source.boundary_schedule(original, starts, 761)
        for pure in (False, True):
            schedule = run.continuation_schedule(lengths, starts, 761, pure, 6000, 120)
            self.assertEqual(set(schedule), set(original))
            before, after, horizons = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
            for update, rows in schedule.items():
                pair = run.fit.pair_for_update(pure, update)
                before[pair].update(map(tuple, changed[update].tolist()))
                after[pair].update(map(tuple, rows.tolist()))
                horizons[pair.delta].update(map(tuple, rows.tolist()))
            self.assertEqual(before, after)
            totals.append(horizons)
        self.assertEqual(*totals)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_qualification_covers_short_horizons_and_stronger_older_pure(self):
        plan = dict(seeds=[1], evaluation=list(range(5)), unchanged_mse_ratio=.8,
                    required_unchanged_wins=4, retention_ratio=1.1)
        rows = []
        for pure in (False, True):
            pairs = run.fit.CONTINUOUS_PAIRS if pure else run.fit.PAIRS
            for member in range(5):
                row = dict(seed=1, pure=pure, member_identity=member)
                for kind in ("full_duration", "boundary", "reference"):
                    row[kind] = {run.fit.policy_name(pair): [dict(elapsed=t, available=True,
                        recursive_mse=(.05 if kind == "full_duration" else .4 if kind == "boundary" else .1),
                        unchanged_mse=1.) for t in (750, 11250)] for pair in pairs}
                rows.append(row)
        self.assertTrue(all(s["qualification_supported"] for s in run.summarize(plan, rows)))
        for row in rows[:5]:
            row["full_duration"]["fixed-50-micro"][-1]["recursive_mse"] = .9
        self.assertFalse(run.summarize(plan, rows)[0]["qualification_supported"])
        for row in rows[5:]:
            row["full_duration"]["fixed-750-continuous"][-1]["recursive_mse"] = .12
        pure = run.summarize(plan, rows)[1]
        self.assertFalse(pure["endpoint_accuracy_retained"])
        self.assertTrue(all(p["qualification_supported"] for p in pure["policies"].values()))


if __name__ == "__main__":
    unittest.main()
