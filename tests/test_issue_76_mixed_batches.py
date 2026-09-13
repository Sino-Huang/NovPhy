import unittest

import torch

from scripts.run_issue_76_mixed_batches import mix_batches


class MixedBatchTest(unittest.TestCase):
    def test_regrouping_preserves_duplicate_samples_and_input_target_alignment(self):
        samples = []
        for episode in range(4):
            values = episode * 100 + torch.arange(32) % 7
            samples.append({"images": values[:, None, None], "presence": (values + 10000)[:, None]})
        mixed = mix_batches(samples)
        self.assertEqual(mixed["images"].shape, (4, 32, 1, 1))
        original = torch.cat([s["images"] for s in samples]).flatten()
        self.assertTrue(torch.equal(mixed["images"].flatten().sort().values, original.sort().values))
        self.assertTrue(torch.equal(mixed["presence"].flatten(), mixed["images"].flatten() + 10000))
        self.assertEqual(mixed["images"][0, :4].flatten().tolist(), [0, 100, 200, 300])
