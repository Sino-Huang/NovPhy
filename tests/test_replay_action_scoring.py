from types import SimpleNamespace
import unittest

import torch

from world_model.planning.replay_action_scoring import score_actions
from world_model.training.native_history_model import STATE_DIM


class ReplayScoringTests(unittest.TestCase):
    def test_common_anchor_recursive_feedback_and_failure_retention(self):
        pair = SimpleNamespace(delta=50)
        calls = []

        class Model:
            pairs = (pair,)

            def carrier(self, current, action, selected):
                calls.append((current.clone(), action.clone()))
                result = current + .1
                result[1, 0] = float("nan")
                return result

        initial = torch.zeros(STATE_DIM)
        actions = [{"drag_x": x, "drag_y": 0, "release_time_ms": 600, "tap_time_ms": 0}
                   for x in (-10, -60)]
        result = score_actions(Model(), initial, actions, pair, endpoint=100)
        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.equal(calls[0][0][0], calls[0][0][1]))
        self.assertTrue(torch.allclose(calls[1][0][0], torch.full_like(initial, .1)))
        self.assertTrue(torch.equal(initial, torch.zeros_like(initial)))
        self.assertFalse(torch.equal(calls[0][1][0], calls[0][1][1]))
        self.assertEqual(result["candidate_transitions"], 4)
        self.assertFalse(result["candidates"][0]["prediction_failed"])
        self.assertTrue(result["candidates"][1]["prediction_failed"])
        self.assertIsNone(result["candidates"][1]["predicted_cost"])
