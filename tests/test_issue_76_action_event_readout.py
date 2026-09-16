import unittest

import torch
from torch.nn import functional as F

from world_model.training.action_event_readout import (
    ActionConditionedEventReadout,
    balanced_event_loss,
)


class ActionEventReadoutTests(unittest.TestCase):
    def test_state_action_interaction_and_gradient_paths_are_present(self):
        torch.manual_seed(76)
        model = ActionConditionedEventReadout(3, hidden_dim=8)
        state = torch.randn(2, 3)
        action = torch.zeros(2, 5)
        action[1, 1] = .1
        first = model(state, state, action)
        second = model(state + 1, state + 1, action)
        self.assertEqual(first.shape, (2, 4))
        zero = model(state, state, torch.zeros_like(action))
        changed_zero = model(state + 1, state + 1, torch.zeros_like(action))
        self.assertFalse(torch.allclose(first - zero, second - changed_zero))
        first.square().mean().backward()
        self.assertTrue(all(parameter.grad is not None
                            and bool(torch.isfinite(parameter.grad).all())
                            for parameter in model.parameters()))

    def test_global_class_weights_do_not_cancel_in_homogeneous_groups(self):
        logits = torch.zeros(4, 4)
        labels = torch.tensor([1, 1, 2, 2])
        weights = torch.tensor([1., .2, .4, 1.])
        groups = [torch.tensor([0, 1]), torch.tensor([2, 3])]
        expected = F.cross_entropy(logits, labels, weight=weights, reduction="none").mean()
        actual = balanced_event_loss(logits, labels, weights, groups, [True, True])
        torch.testing.assert_close(actual, expected)
        self.assertLess(float(actual), float(F.cross_entropy(logits, labels, weight=weights)))

    def test_ranking_remains_within_each_training_lineage(self):
        logits = torch.zeros(4, 4)
        labels = torch.tensor([0, 1, 0, 1])
        groups = [torch.tensor([0, 1]), torch.tensor([2, 3])]
        classification = F.cross_entropy(logits, labels)
        actual = balanced_event_loss(logits, labels, torch.ones(4), groups, [True, False])
        torch.testing.assert_close(actual, classification + torch.log(torch.tensor(2.)) / 2)


if __name__ == "__main__":
    unittest.main()
