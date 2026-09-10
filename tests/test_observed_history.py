import unittest

import torch

from world_model.training.observed_history import DIM, ObservedHistoryEncoder


class ObservedHistoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7601)
        self.model = ObservedHistoryEncoder()
        self.z = torch.randn(2, 5, DIM)
        self.a = torch.randn(2, 5, 5)
        self.t = torch.arange(5, dtype=torch.float64)[None].expand(2, -1)
        self.observed = torch.tensor([[True, False, True, False, True]]).expand(2, -1)
        self.acted = torch.tensor([[False, True, False, False, False]]).expand(2, -1)

    def run_sequence(self, z=None, a=None):
        return self.model(self.z if z is None else z, self.a if a is None else a,
                          self.t, self.observed, self.acted)

    def test_streaming_matches_whole_prefix_and_does_not_mutate_state(self):
        full, _ = self.run_sequence()
        _, first = self.model(self.z[:, :2], self.a[:, :2], self.t[:, :2],
                              self.observed[:, :2], self.acted[:, :2])
        saved = first.memory.clone()
        rest, _ = self.model(self.z[:, 2:], self.a[:, 2:], self.t[:, 2:],
                            self.observed[:, 2:], self.acted[:, 2:], first)
        torch.testing.assert_close(full[:, 2:], rest)
        torch.testing.assert_close(first.memory, saved)

    def test_future_observation_cannot_change_earlier_memory(self):
        full, _ = self.run_sequence()
        changed = self.z.clone()
        changed[:, 4] += 100
        other, _ = self.run_sequence(z=changed)
        torch.testing.assert_close(full[:, :4], other[:, :4])
        self.assertFalse(torch.allclose(full[:, 4], other[:, 4]))

    def test_missing_fields_do_not_leak_and_padding_holds_memory(self):
        full, _ = self.run_sequence()
        z, a = self.z.clone(), self.a.clone()
        z[~self.observed] = float("nan")
        a[~self.acted] = float("nan")
        other, _ = self.run_sequence(z, a)
        torch.testing.assert_close(full, other)
        torch.testing.assert_close(full[:, 2], full[:, 3])

    def test_executed_action_updates_memory_without_an_observation(self):
        full, _ = self.run_sequence()
        changed = self.a.clone()
        changed[:, 1] += 1
        other, _ = self.run_sequence(a=changed)
        self.assertFalse(torch.allclose(full[:, 1], other[:, 1]))
        self.assertFalse(torch.allclose(full[:, 4], other[:, 4]))

    def test_previous_shot_memory_is_not_equivalent_to_episode_reset(self):
        full, _ = self.run_sequence()
        reset, _ = self.model(self.z[:, 4:], self.a[:, 4:], self.t[:, 4:],
                             self.observed[:, 4:], self.acted[:, 4:])
        self.assertFalse(torch.allclose(full[:, 4:], reset))

    def test_nonfinite_observed_carrier_and_backwards_clock_fail(self):
        broken = self.z.clone()
        broken[:, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.run_sequence(z=broken)
        with self.assertRaisesRegex(ValueError, "backwards"):
            self.model(self.z, self.a, -self.t, self.observed, self.acted)

    def test_all_recurrent_parameters_participate_in_training(self):
        sequence, _ = self.run_sequence()
        sequence.square().mean().backward()
        self.assertTrue(all(p.grad is not None and bool(p.grad.abs().sum() > 0)
                            for p in self.model.parameters()))


if __name__ == "__main__":
    unittest.main()
