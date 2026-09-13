import unittest

import torch

from scripts.issue_76_full_duration_loss import trajectory_batch, full_continuous_loss, full_hybrid_loss
from scripts.issue_76_matched_batches import indexed_batch
from tests.test_native_history_fit import synthetic_shard
from world_model.training.native_history_fit import continuous_loss, hybrid_loss
from world_model.training.native_history_model import NativeHistoryDynamics, STATE_DIM


class FullDurationLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(76)

    def test_four_step_values_and_gradients_match_original_losses(self):
        shard = synthetic_shard(80)
        carriers = [torch.randn(80, STATE_DIM)]
        rows = torch.tensor([[0, 0], [0, 5], [0, 75]])
        for pure in (False, True):
            model = NativeHistoryDynamics(pure=pure, width=16)
            for pair in model.pairs:
                batch = trajectory_batch([shard], carriers, rows, pair.delta,
                                         symbolic=not pure, endpoint=4 * pair.delta)
                if pure:
                    args = (model, batch["z"], batch["available"], batch["action"], pair)
                    old, new = continuous_loss(*args), full_continuous_loss(*args)
                else:
                    old, new = hybrid_loss(model, batch, pair), full_hybrid_loss(model, batch, pair)
                torch.testing.assert_close(old, new)
                old_grads = torch.autograd.grad(old, tuple(model.parameters()), allow_unused=True)
                new_grads = torch.autograd.grad(new, tuple(model.parameters()), allow_unused=True)
                for a, b in zip(old_grads, new_grads):
                    if a is None:
                        self.assertIsNone(b)
                    else:
                        torch.testing.assert_close(a, b)

    def test_targets_match_prefix_and_do_not_cross_shot_boundaries(self):
        shard = synthetic_shard(40)
        shard["segment_ranges"] = [{"start": 0, "stop": 20, "action": torch.ones(5)},
                                   {"start": 20, "stop": 40, "action": -torch.ones(5)}]
        shard["tensors"]["fixed_steps"][20:] -= 1000
        carriers = [torch.randn(40, STATE_DIM)]
        rows = torch.tensor([[0, 0], [0, 19], [0, 20]])
        for symbolic in (False, True):
            short = indexed_batch([shard], carriers, rows, 50, symbolic=symbolic)
            full = trajectory_batch([shard], carriers, rows, 50, symbolic=symbolic)
            for key in short:
                actual = full[key][:, :5] if key in ("z", "available") else full[key]
                self.assertTrue(torch.equal(short[key], actual), key)
            self.assertEqual(int(full["available"][0].sum()), 20)
            self.assertEqual(int(full["available"][1].sum()), 1)
            self.assertEqual(int(full["available"][2].sum()), 20)
            self.assertEqual(full["z"].shape[1], 226)
            self.assertFalse(bool(full["available"][:, -1].any()))

    def test_recursive_chain_is_not_reset_and_missing_tail_does_not_supervise(self):
        class Increment(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.delta = torch.nn.Parameter(torch.tensor(1.))

            def carrier(self, z, action, pair):
                return z + self.delta

        model = Increment()
        z = torch.zeros(2, 9, 1)
        mask = torch.ones(2, 9, dtype=torch.bool)
        action = torch.zeros(2, 5)
        loss = full_continuous_loss(model, z, mask, action, None)
        expected = 1 + sum(i * i + .01 * max(i - 2, 0) ** 2 for i in range(1, 9)) / 8
        self.assertAlmostEqual(float(loss.detach()), expected, places=5)
        loss.backward()
        self.assertGreater(float(model.delta.grad), 50.)
        mask[:, 7:] = False
        z[:, 7:] = 9999.
        loss = full_continuous_loss(model, z, mask, action, None)
        expected = 1 + sum(i * i + .01 * max(i - 2, 0) ** 2 for i in range(1, 7)) / 6
        self.assertAlmostEqual(float(loss.detach()), expected, places=5)


if __name__ == "__main__":
    unittest.main()
