import unittest

import torch

from scripts.run_issue_76_matched_dynamics import recursive_probe, summarize
from tests.test_native_history_fit import synthetic_shard
from world_model.training.native_history_model import NativeHistoryDynamics, STATE_DIM


class MatchedDynamicsTests(unittest.TestCase):
    def test_qualification_requires_every_seed_both_arms_and_unchanged_contrast(self):
        plan = dict(seeds=[1, 2, 3], evaluation=list(range(5)), hybrid_parent_mse_ratio=.8,
                    pure_parent_mse_ratio=1.1, hybrid_unchanged_wins_per_seed=4)
        rows = []
        for seed in plan["seeds"]:
            for pure in (False, True):
                policy = "fixed-750-continuous" if pure else "fixed-750-micro"
                for member in range(5):
                    rows.append(dict(seed=seed, pure=pure, member_identity=member,
                        parent=dict(available=True, policies={policy: 1.}),
                        mixed=dict(available=True, policies={policy: .5}, unchanged_carrier_mse=.6)))
        self.assertTrue(all(s["qualification_supported"] for s in summarize(plan, rows)))
        rows[0]["mixed"]["unchanged_carrier_mse"] = .4
        rows[1]["mixed"]["unchanged_carrier_mse"] = .4
        self.assertFalse(summarize(plan, rows)[0]["qualification_supported"])
        rows[-1]["mixed"]["available"] = False
        self.assertFalse(summarize(plan, rows)[-1]["qualification_supported"])
        self.assertTrue(summarize(plan, rows)[2]["qualification_supported"])

    def test_recursive_probe_does_not_invent_an_endpoint(self):
        shard = synthetic_shard(40)
        carrier = torch.zeros(40, STATE_DIM)
        model = NativeHistoryDynamics(pure=True, width=16)
        self.assertEqual(recursive_probe(model, shard, carrier), {"available": False})


if __name__ == "__main__":
    unittest.main()
