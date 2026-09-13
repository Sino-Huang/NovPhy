import unittest

import torch

from scripts.diagnose_issue_76_controller_utility import local_costs, solve
from world_model.training.cnn_hybrid import linear_macs
from world_model.training.native_history_fit import REFERENCE_TRANSITION_MACS, controller_targets
from world_model.training.native_history_model import NativeHistoryDynamics, STATE_DIM


class ControllerUtilityTests(unittest.TestCase):
    def test_diagnostic_reproduces_both_native_teachers(self):
        torch.set_num_threads(1)
        for pure in (False, True):
            torch.manual_seed(3)
            model = NativeHistoryDynamics(pure=pure, width=8).eval()
            carrier, action = torch.rand(36, STATE_DIM), torch.zeros(5)
            steps = list(range(0, 1800, 50))
            original = controller_targets(model, carrier, steps, action)
            _, kept, raw = local_costs(model, carrier, steps, action)
            scales = raw.new_tensor([p.delta * .0004 for p in model.pairs])
            penalties = raw.new_tensor([.01 * linear_macs(model, p) / REFERENCE_TRANSITION_MACS
                                        for p in model.pairs])
            labels, values, q = solve(raw * scales + penalties, model.pairs, kept)
            self.assertTrue(torch.equal(labels, original["labels"]))
            self.assertTrue(torch.equal(values, q.min(-1).values))

    def test_missing_path_is_not_padded(self):
        model = NativeHistoryDynamics(pure=True, width=8)
        with self.assertRaisesRegex(ValueError, "exact finite endpoint"):
            solve(torch.full((2, 3), torch.inf), model.pairs, [0, 75])


if __name__ == "__main__":
    unittest.main()
