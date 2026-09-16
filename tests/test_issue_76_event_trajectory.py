import unittest

import torch

from world_model.model import Abstraction, PredictionPair
from world_model.training.event_trajectory_ranking import (
    adaptive_trajectory,
    fixed_trajectory,
    trajectory_pair_teacher,
)


class TrajectoryTests(unittest.TestCase):
    def test_fixed_snapshots_and_adaptive_reselection_reach_exact_endpoint(self):
        class Model:
            pairs = (PredictionPair(50, Abstraction.CONTINUOUS),
                     PredictionPair(250, Abstraction.MICRO),
                     PredictionPair(750, Abstraction.MACRO))

            def carrier(self, state, action, pair):
                return state + pair.delta

        class Selector:
            def __call__(self, state, action, remaining):
                logits = torch.zeros(len(state), 3)
                logits[:, 2] = 10
                logits[:, 1] = 5
                return logits

        model = Model()
        initial, action = torch.zeros(2, 1), torch.zeros(2, 5)
        endpoint, snapshots = fixed_trajectory(model, initial, action, model.pairs[1],
            endpoint=1000, checkpoints=(0, 250, 750))
        torch.testing.assert_close(endpoint, torch.full((2, 1), 1000.))
        torch.testing.assert_close(snapshots[0, :, 0], torch.tensor([0., 250., 750.]))
        endpoint, counts = adaptive_trajectory(model, Selector(), initial, action, endpoint=1000)
        torch.testing.assert_close(endpoint, torch.full((2, 1), 1000.))
        torch.testing.assert_close(counts, torch.tensor([[0, 1, 1], [0, 1, 1]]))
        torch.testing.assert_close(initial, torch.zeros_like(initial))

    def test_teacher_uses_training_class_loss_and_declared_compute_cost(self):
        logits = torch.tensor([[[2., 0.], [0., 2.]], [[0., 2.], [2., 0.]]])
        labels = torch.tensor([0, 0])
        costs = torch.tensor([10., 0.])
        torch.testing.assert_close(trajectory_pair_teacher(logits, labels, costs, 0.),
                                   torch.tensor([0, 1]))
        torch.testing.assert_close(trajectory_pair_teacher(logits, labels, costs, 1.),
                                   torch.tensor([1, 1]))


if __name__ == "__main__":
    unittest.main()
