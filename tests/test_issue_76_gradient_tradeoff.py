import unittest

import torch

from scripts.diagnose_issue_76_gradient_tradeoff import adamw_delta, flatten


class AdamWDirectionTest(unittest.TestCase):
    def test_saved_moments_clip_and_unused_parameter_match_real_step(self):
        parameters = [torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64)),
                      torch.nn.Parameter(torch.tensor([3.], dtype=torch.float64))]
        optimizer = torch.optim.AdamW(parameters, lr=.0003, weight_decay=.0001)
        for p in parameters:
            p.grad = torch.ones_like(p)
        optimizer.step()
        before = [p.detach().clone() for p in parameters]
        state = optimizer.state_dict()
        moments = [{k: v.clone() if torch.is_tensor(v) else v for k, v in s.items()}
                   for s in state["state"].values()]
        gradients = (torch.tensor([30., -40.], dtype=torch.float64), None)
        predicted = adamw_delta(tuple(parameters), gradients, state)
        for p, old in zip(parameters, before):
            torch.testing.assert_close(p, old, rtol=0, atol=0)
        for actual, old in zip(state["state"].values(), moments):
            for key in old:
                torch.testing.assert_close(actual[key], old[key], rtol=0, atol=0)
        for p, g in zip(parameters, gradients):
            p.grad = None if g is None else g.clone()
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        actual = flatten([p.detach() - old for p, old in zip(parameters, before)], parameters)
        torch.testing.assert_close(predicted, actual, rtol=1e-9, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
