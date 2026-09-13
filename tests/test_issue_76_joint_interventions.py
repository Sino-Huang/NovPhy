import unittest

import torch

from scripts.diagnose_issue_76_joint_interventions import endpoint_loss


class EndpointLossTest(unittest.TestCase):
    def test_full_recursive_gradient_and_unavailable_endpoint_mask(self):
        class Increment(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.delta = torch.nn.Parameter(torch.tensor(1.))

            def carrier(self, z, action, pair):
                return z + self.delta

        model = Increment()
        z = torch.zeros(2, 4, 1)
        z[1, -1] = 9999
        available = torch.ones(2, 4, dtype=torch.bool)
        available[1, -1] = False
        batch = dict(z=z, available=available, action=torch.zeros(2, 5))
        loss = endpoint_loss(model, batch, None)
        self.assertEqual(float(loss.detach()), 9.)
        loss.backward()
        self.assertEqual(float(model.delta.grad), 18.)


if __name__ == "__main__":
    unittest.main()
