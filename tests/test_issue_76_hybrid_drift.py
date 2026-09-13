import unittest

import torch

from scripts.diagnose_issue_76_hybrid_drift import alignment, drift_curves, gradient_probe
from tests.test_native_history_fit import synthetic_shard
from world_model.training.native_history_model import NativeHistoryDynamics, STATE_DIM


class HybridDriftTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(76)

    def test_alignment_and_probe_do_not_update_weights_or_grad_buffers(self):
        self.assertAlmostEqual(alignment(torch.tensor([1., 0.]), torch.tensor([-1., 0.]))["cosine"], -1.)
        model = NativeHistoryDynamics(pure=False, width=16)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rows = gradient_probe(model, synthetic_shard(80), torch.randn(80, STATE_DIM), 76)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(len(row["shared_mode_gradients"]) == 3 for row in rows))
        self.assertTrue(all(set(row["head_gradients"]) == {"micro", "macro"} for row in rows))
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_curves_preserve_missing_endpoint_and_first_step_equality(self):
        model = NativeHistoryDynamics(pure=True, width=16)
        curves = drift_curves(model, synthetic_shard(40), torch.randn(40, STATE_DIM))
        for rows in curves.values():
            self.assertEqual(rows[0]["local_mse"], rows[0]["recursive_mse"])
            self.assertFalse(rows[-1]["available"])
            self.assertEqual(rows[-1]["elapsed"], 11250)


if __name__ == "__main__":
    unittest.main()
