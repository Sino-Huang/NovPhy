import unittest

import torch

from scripts.diagnose_issue_76_joint_interventions import endpoint_loss
from scripts.issue_76_endpoint_anchor_loss import endpoint_anchor_loss
from scripts.issue_76_full_duration_loss import trajectory_batch
from scripts.issue_76_local_anchor_loss import local_anchor_loss
from tests.test_native_history_fit import synthetic_shard
from world_model.training.native_history_model import NativeHistoryDynamics, STATE_DIM


class EndpointAnchorLossTests(unittest.TestCase):
    def test_values_and_gradients_match_frozen_probe_for_all_pairs(self):
        torch.set_num_threads(1)
        torch.manual_seed(76)
        shard = synthetic_shard(240)
        carriers = [torch.randn(240, STATE_DIM)]
        rows = torch.tensor([[0, 0], [0, 30], [0, 239]])
        for pure in (False, True):
            model = NativeHistoryDynamics(pure=pure, width=16)
            for pair in model.pairs:
                batch = trajectory_batch([shard], carriers, rows, pair.delta, symbolic=not pure)
                expected = local_anchor_loss(model, batch, pair, pure=pure)
                if pair.delta == 750:
                    expected = expected + endpoint_loss(model, batch, pair)
                actual = endpoint_anchor_loss(model, batch, pair, pure=pure)
                torch.testing.assert_close(actual, expected)
                left = torch.autograd.grad(expected, tuple(model.parameters()), allow_unused=True)
                right = torch.autograd.grad(actual, tuple(model.parameters()), allow_unused=True)
                for a, b in zip(left, right):
                    if a is None:
                        self.assertIsNone(b)
                    else:
                        torch.testing.assert_close(a, b)

    def test_missing_endpoint_retains_available_local_and_recursive_supervision(self):
        torch.manual_seed(76)
        model = NativeHistoryDynamics(pure=True, width=16)
        pair = next(p for p in model.pairs if p.delta == 750)
        batch = dict(z=torch.randn(2, 16, STATE_DIM), available=torch.ones(2, 16, dtype=torch.bool),
                     action=torch.zeros(2, 5))
        batch["available"][:, 8:] = False
        batch["z"][:, 8:] = 9999
        expected = local_anchor_loss(model, batch, pair, pure=True)
        actual = endpoint_anchor_loss(model, batch, pair, pure=True)
        self.assertIsNotNone(actual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        batch["available"][:, 1:] = False
        self.assertIsNone(endpoint_anchor_loss(model, batch, pair, pure=True))


if __name__ == "__main__":
    unittest.main()
