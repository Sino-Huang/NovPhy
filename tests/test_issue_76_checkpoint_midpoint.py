import unittest

import torch

from scripts.diagnose_issue_76_checkpoint_midpoint import midpoint, error_comparison


class MidpointTests(unittest.TestCase):
    def test_midpoint_preserves_inputs_and_uses_equal_weights(self):
        left, right = {"weight": torch.tensor([1., 3.])}, {"weight": torch.tensor([3., 7.])}
        result = midpoint(left, right)
        torch.testing.assert_close(result["weight"], torch.tensor([2., 5.]), rtol=0, atol=0)
        torch.testing.assert_close(left["weight"], torch.tensor([1., 3.]), rtol=0, atol=0)
        torch.testing.assert_close(right["weight"], torch.tensor([3., 7.]), rtol=0, atol=0)

    def test_prediction_error_identity_distinguishes_shared_and_cancelling_errors(self):
        target = torch.zeros(2)
        for right in (torch.tensor([1., 2.]), torch.tensor([-1., -2.])):
            result = error_comparison(torch.tensor([1., 2.]), right, target)
            expected = (result["local_anchor_mse"] + result["endpoint_anchor_mse"] + 2 * result["cross_error_mean"]) / 4
            self.assertEqual(result["prediction_midpoint_mse"], expected)
        self.assertEqual(result["prediction_midpoint_mse"], 0.)


if __name__ == "__main__":
    unittest.main()
