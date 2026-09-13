import unittest

import torch

from scripts.audit_issue_76_pig_exposure import count_samples, fit


class PigExposureTest(unittest.TestCase):
    def test_exact_original_draws_include_duplicates_and_use_pig_slot(self):
        presence = torch.ones(5, 22)
        presence[0:2, 10] = 0
        seed, updates = 760930001, [7, 257, 507]
        expected = torch.cat([presence[torch.randint(5, (32,), generator=fit.generator(seed, u)), 10]
                              for u in updates])
        result = count_samples(presence, seed, updates)
        self.assertEqual(result, dict(positive=int(expected.sum()), negative=int((expected == 0).sum())))
        self.assertEqual(sum(result.values()), 96)
        self.assertGreater(result["negative"], 2)


if __name__ == "__main__":
    unittest.main()
