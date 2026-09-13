import unittest

import torch
from torch.nn import functional as F

from scripts.run_issue_76_balanced_pig import balanced_loss, fit


class BalancedPigTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.batch = dict(images=torch.zeros(4, 3, 64, 96),
            presence=torch.randint(2, (4, 22)).float(), centers=torch.rand(4, 22, 2))
        self.batch["presence"][:, 10] = torch.tensor([0., 1., 1., 1.])
        self.model = fit.NativeVisualParser()

    def test_unit_weights_preserve_original_loss_and_all_parameter_gradients(self):
        original = fit.perception_loss(self.model, self.batch)
        original.backward()
        gradients = [p.grad.clone() for p in self.model.parameters()]
        self.model.zero_grad()
        actual = balanced_loss(self.model, self.batch, dict(positive=1., negative=1.))
        actual.backward()
        torch.testing.assert_close(actual, original)
        for before, parameter in zip(gradients, self.model.parameters()):
            torch.testing.assert_close(parameter.grad, before)

    def test_weighted_correction_changes_only_pig_terms_with_one_forward(self):
        output = self.model(self.batch["images"])
        calls = []
        def model(_):
            calls.append(True)
            return output
        actual = balanced_loss(model, self.batch, dict(positive=2 / 3, negative=2.))
        self.assertEqual(len(calls), 1)
        target = self.batch["presence"][:, 10]
        logits = output["presence_logits"][:, 10]
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        mse = (logits.sigmoid() - target).square()
        weights = torch.tensor([2., 2 / 3, 2 / 3, 2 / 3])
        expected = fit.perception_loss(lambda _: output, self.batch) + ((weights - 1) * (4 / 22 * bce + mse)).mean()
        torch.testing.assert_close(actual, expected)
        self.assertEqual(float(weights[0]), float(weights[1:].sum()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
