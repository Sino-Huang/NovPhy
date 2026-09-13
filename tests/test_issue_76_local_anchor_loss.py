import unittest

import torch

from scripts.diagnose_issue_76_gradient_tradeoff import terms
from scripts.issue_76_full_duration_loss import trajectory_batch
from scripts.issue_76_local_anchor_loss import local_anchor_loss
from tests.test_native_history_fit import synthetic_shard
from world_model.training.native_history_model import NativeHistoryDynamics, STATE_DIM


class LocalAnchorLossTests(unittest.TestCase):
    def test_exact_values_and_gradients_for_all_pairs_and_masked_tails(self):
        torch.set_num_threads(1)
        torch.manual_seed(76)
        shard = synthetic_shard(240)
        carriers = [torch.randn(240, STATE_DIM)]
        rows = torch.tensor([[0, 0], [0, 30], [0, 239]])
        for pure in (False, True):
            model = NativeHistoryDynamics(pure=pure, width=16)
            for pair in model.pairs:
                batch = trajectory_batch([shard], carriers, rows, pair.delta, symbolic=not pure)
                local, recursive, symbolic, _ = terms(model, batch, pair)
                expected = (10 if pair.delta == 750 else 1) * local + recursive
                expected = expected + (0 if symbolic is None else symbolic)
                actual = local_anchor_loss(model, batch, pair, pure=pure)
                torch.testing.assert_close(actual, expected)
                expected_grad = torch.autograd.grad(expected, tuple(model.parameters()), allow_unused=True)
                actual_grad = torch.autograd.grad(actual, tuple(model.parameters()), allow_unused=True)
                for left, right in zip(expected_grad, actual_grad):
                    if left is None:
                        self.assertIsNone(right)
                    else:
                        torch.testing.assert_close(left, right)

    def test_missing_first_target_stays_unusable(self):
        model = NativeHistoryDynamics(pure=True, width=16)
        pair = next(p for p in model.pairs if p.delta == 750)
        batch = dict(z=torch.zeros(2, 16, STATE_DIM), available=torch.zeros(2, 16, dtype=torch.bool),
                     action=torch.zeros(2, 5))
        batch["available"][:, 0] = True
        self.assertIsNone(local_anchor_loss(model, batch, pair, pure=True))


if __name__ == "__main__":
    unittest.main()
