from collections import Counter, defaultdict
import unittest

import torch

from scripts.run_issue_76_boundary_dynamics import boundary_schedule, summarize
from scripts.issue_76_matched_batches import mixed_schedule, sampled_schedule
from scripts.run_issue_76_native_refit import pair_for_update


class BoundaryDynamicsTests(unittest.TestCase):
    def test_half_original_half_same_lineage_boundaries_without_rng_mutation(self):
        starts = [[0, 20], [], [0, 18, 37], [0], [0, 41]]
        original = sampled_schedule([40, 0, 53, 61, 80], 761, False, 120)
        before = {u: rows.clone() for u, rows in original.items()}
        rng = torch.random.get_rng_state().clone()
        changed = boundary_schedule(original, starts, 761)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(set(changed), set(original))
        for update, rows in changed.items():
            self.assertTrue(torch.equal(original[update], before[update]))
            self.assertTrue(torch.equal(rows[1::2], original[update][1::2]))
            self.assertTrue(torch.equal(rows[:, 0], original[update][:, 0]))
            self.assertEqual(len(rows[::2]), 16)
            self.assertTrue(all(int(s) in starts[int(i)] for i, s in rows[::2]))
        again = boundary_schedule(original, starts, 761)
        self.assertTrue(all(torch.equal(rows, again[u]) for u, rows in changed.items()))

    def test_mixing_preserves_paired_horizon_exposure_and_failed_assignments(self):
        original = sampled_schedule([40, 0, 53, 61, 80], 761, False, 120)
        changed = boundary_schedule(original, [[0, 20], [], [0, 30], [0], [0, 40]], 761)
        exposures = []
        for pure in (False, True):
            schedule = mixed_schedule(changed, 761, pure)
            self.assertEqual(set(schedule), set(original))
            by_horizon, before_pair, after_pair = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
            for update, rows in changed.items():
                before_pair[pair_for_update(pure, update)].update(map(tuple, rows.tolist()))
            for update, rows in schedule.items():
                pair = pair_for_update(pure, update)
                self.assertFalse(bool((rows[:, 0] == 1).any()))
                after_pair[pair].update(map(tuple, rows.tolist()))
                by_horizon[pair.delta].update(map(tuple, rows.tolist()))
            self.assertEqual(before_pair, after_pair)
            exposures.append(by_horizon)
        self.assertEqual(*exposures)

    def test_qualification_requires_first_step_endpoint_and_each_seed(self):
        plan = dict(seeds=[1, 2, 3], evaluation=list(range(5)), first_step_ratio=.5,
                    hybrid_endpoint_ratio=.8, pure_endpoint_ratio=1.1, required_unchanged_wins=4)
        rows = []
        for seed in plan["seeds"]:
            for pure in (False, True):
                name = "fixed-750-continuous" if pure else "fixed-750-micro"
                for member in range(5):
                    rows.append(dict(seed=seed, pure=pure, member_identity=member,
                        reference={name: [dict(elapsed=t, available=True, recursive_mse=1., unchanged_mse=.6)
                                          for t in (750, 11250)]},
                        boundary={name: [dict(elapsed=t, available=True, recursive_mse=.4, unchanged_mse=.6)
                                         for t in (750, 11250)]}))
        self.assertTrue(all(s["qualification_supported"] for s in summarize(plan, rows)))
        for row in rows[:5]:
            row["boundary"]["fixed-750-micro"][0]["recursive_mse"] = .6
        result = summarize(plan, rows)
        self.assertFalse(result[0]["qualification_supported"])
        self.assertTrue(result[0]["endpoint_contrast_supported"])
        rows[-1]["boundary"]["fixed-750-continuous"][-1]["available"] = False
        self.assertFalse(summarize(plan, rows)[-1]["qualification_supported"])


if __name__ == "__main__":
    unittest.main()
