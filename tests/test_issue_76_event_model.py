import unittest

import torch

from world_model.training.event_ranking import (
    CLEAR_INDEX,
    EventReadout,
    event_loss,
    pair_teacher,
    second_fraction,
)


class EventModelTests(unittest.TestCase):
    def test_event_readout_and_ranking_loss_have_declared_shapes(self):
        model = EventReadout(7, hidden_dim=11)
        initial = torch.randn(4, 7)
        successor = torch.randn(4, 7)
        action = torch.randn(4, 5)
        logits = model(initial, successor, action)
        labels = torch.tensor([CLEAR_INDEX, 1, 2, 3])
        loss = event_loss(logits, labels, torch.ones(4), paired_ranking=True)
        self.assertEqual(logits.shape, (4, 4))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_pair_teacher_uses_each_records_observed_class(self):
        logits = torch.zeros(2, 3, 4)
        logits[0, 2, 0] = 4
        logits[1, 1, 3] = 4
        self.assertEqual(pair_teacher(logits, torch.tensor([0, 3])).tolist(), [2, 1])

    def test_second_fraction_reports_non_degenerate_choice_use(self):
        self.assertEqual(second_fraction({"a": 7, "b": 2, "c": 1}), 0.2)
        self.assertEqual(second_fraction({"a": 10}), 0.0)


if __name__ == "__main__":
    unittest.main()
