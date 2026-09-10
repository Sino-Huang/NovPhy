import unittest

import torch

from world_model.model import Abstraction, PredictionPair
from world_model.training.cnn_hybrid import PAIRS
from world_model.training.history_dynamics import HistoryDynamics, STATE_DIM, capacity_contract


class HistoryDynamicsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7601)
        self.z, self.a = torch.randn(2, STATE_DIM), torch.randn(2, 5)

    def test_pure_model_has_no_symbolic_or_mode_parameters(self):
        model = HistoryDynamics(pure=True, width=32)
        names = [name for name, _ in model.named_parameters()]
        self.assertFalse(any("micro" in n or "macro" in n or "abstraction_embedding" in n for n in names))
        for pair in model.pairs:
            self.assertEqual(model.carrier(self.z, self.a, pair).shape, self.z.shape)
        with self.assertRaisesRegex(ValueError, "permitted pair"):
            model.carrier(self.z, self.a, PredictionPair(1, Abstraction.MICRO))

    def test_same_observed_memory_influences_both_arms(self):
        changed = self.z.clone()
        changed[:, 236:] += 1
        pair = PredictionPair(1, Abstraction.CONTINUOUS)
        for pure in (False, True):
            model = HistoryDynamics(pure=pure, width=32)
            first = model.carrier(self.z, self.a, pair)[:, :236]
            second = model.carrier(changed, self.a, pair)[:, :236]
            self.assertFalse(torch.allclose(first, second))

    def test_only_selected_symbolic_head_executes(self):
        model = HistoryDynamics(pure=False, width=32)
        calls = []
        hooks = [model.micro_head.register_forward_hook(lambda *args: calls.append("micro")),
                 model.macro_head.register_forward_hook(lambda *args: calls.append("macro"))]
        for mode, expected in ((Abstraction.CONTINUOUS, []), (Abstraction.MICRO, ["micro"]),
                               (Abstraction.MACRO, ["macro"])):
            calls.clear()
            model.carrier(self.z, self.a, PredictionPair(1, mode))
            self.assertEqual(calls, expected)
        for hook in hooks:
            hook.remove()

    def test_every_hybrid_parameter_is_active_across_permitted_pairs(self):
        model = HistoryDynamics(pure=False, width=32)
        sum(model.carrier(self.z, self.a, p).square().mean() for p in PAIRS).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_count_only_capacity_match_with_no_padding(self):
        contract = capacity_contract()
        self.assertEqual(contract["state_dim"], 300)
        self.assertLessEqual(contract["relative_parameter_difference"], .02)
        self.assertEqual(contract["dead_padding_parameters"], 0)
        self.assertFalse(contract["fit_executed"])


if __name__ == "__main__":
    unittest.main()
