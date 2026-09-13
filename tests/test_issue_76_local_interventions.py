import unittest

import torch

from scripts.diagnose_issue_76_gradient_tradeoff import adamw_delta
from scripts.diagnose_issue_76_local_interventions import reset_history


class ResetHistoryTest(unittest.TestCase):
    def test_reset_copy_matches_fresh_adamw_without_mutating_source(self):
        parameter = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
        optimizer = torch.optim.AdamW([parameter], lr=.00003, weight_decay=.0001)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        saved = optimizer.state_dict()
        original = {key: value.clone() for key, value in saved["state"][0].items()}
        reset = reset_history(saved)
        self.assertEqual(reset["param_groups"], saved["param_groups"])
        for key, value in saved["state"][0].items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
            self.assertEqual(int(torch.count_nonzero(reset["state"][0][key])), 0)
        gradient = torch.tensor([30., -40.], dtype=torch.float64)
        expected = adamw_delta((parameter,), (gradient,), reset)
        before = parameter.detach().clone()
        fresh = torch.optim.AdamW([parameter], lr=.00003, weight_decay=.0001)
        parameter.grad = gradient.clone()
        torch.nn.utils.clip_grad_norm_([parameter], 1.)
        fresh.step()
        torch.testing.assert_close(parameter.detach() - before, expected, rtol=1e-9, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
