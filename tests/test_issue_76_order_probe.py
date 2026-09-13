import unittest

import torch

from scripts.run_issue_76_order_probe import interleaved_order, original_update, evaluation_rows, training_sample
from scripts.run_issue_76_native_refit import generator


class OrderProbeTest(unittest.TestCase):
    def test_interleave_preserves_exact_update_and_frame_exposure(self):
        unusable = {51, 121}
        entries = [{"family": str(i // 50), "frames": 601 + i, "usable": i not in unusable} for i in range(250)]
        order = interleaved_order(entries)
        self.assertEqual(order[:10], [0, 50, 100, 150, 200, 1, 51, 101, 151, 201])
        mapped = [original_update(i, order) for i in range(1500)]
        self.assertEqual(sorted(mapped), list(range(1500)))
        for update, source in enumerate(mapped):
            entry, rows = training_sample(entries, order, 760930001, update)
            self.assertIs(entry, entries[source % 250])
            if source % 250 in unusable:
                self.assertIsNone(rows)
            else:
                old_rows = torch.randint(601 + source % 250, (32,), generator=generator(760930001, source))
                self.assertTrue(torch.equal(rows, old_rows))
        self.assertEqual(sum(i % 250 in unusable for i in mapped), 12)

    def test_evaluation_rows_include_real_endpoints_without_replacement(self):
        self.assertEqual(evaluation_rows(0), [])
        self.assertEqual(evaluation_rows(2), [0, 1])
        rows = evaluation_rows(601)
        self.assertEqual((len(rows), rows[0], rows[-1]), (32, 0, 600))
        self.assertEqual(len(set(rows)), 32)
