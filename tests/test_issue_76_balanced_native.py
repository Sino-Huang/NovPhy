import unittest

import torch

from scripts.run_issue_76_balanced_native import controller_targets, training_phase
from world_model.training.native_history_model import NativeHistoryDynamics, STATE_DIM


class BalancedNativeTests(unittest.TestCase):
    def test_training_phase_is_fixed_at_six_thousand_updates(self):
        self.assertEqual(training_phase(0), "local")
        self.assertEqual(training_phase(5999), "local")
        self.assertEqual(training_phase(6000), "full_duration")
        self.assertEqual(training_phase(11999), "full_duration")
        with self.assertRaises(ValueError):
            training_phase(12000)

    def test_low_cost_controller_teacher_has_exact_paths(self):
        torch.manual_seed(3)
        model = NativeHistoryDynamics(pure=False, width=16).eval()
        carrier = torch.randn(5, STATE_DIM)
        value = controller_targets(model, carrier, [0, 50, 100, 150, 200],
                                   torch.zeros(5), compute_weight=0.0001)
        self.assertEqual(value["z"].shape, (4, STATE_DIM))
        self.assertEqual(value["labels"].shape, (4,))
        self.assertTrue(bool(torch.isfinite(value["remaining"]).all()))


if __name__ == "__main__":
    unittest.main()
