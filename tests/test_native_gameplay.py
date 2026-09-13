from dataclasses import asdict
from io import BytesIO
import unittest

import numpy as np
from PIL import Image
import torch

from world_model.planning.native_gameplay import (
    NativeObservationHistory, agent_image_tensor, score_native_actions,
)
from world_model.planning.gameplay import SlingshotAction, SlingshotActionBounds
from world_model.training.native_history_data import VISUAL_DIM, action_context
from world_model.training.native_history_fit import (
    NativeVisualParser, NativeHistoryController, encode_visual, encode_history,
)
from world_model.training.native_history_model import NativeHistoryDynamics
from world_model.training.observed_history import ObservedHistoryEncoder


class NativeGameplayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7613)
        self.parser = NativeVisualParser().eval()
        self.history = ObservedHistoryEncoder(carrier_dim=VISUAL_DIM).eval()
        self.images = torch.randint(0, 256, (5, 3, 64, 96), dtype=torch.uint8)
        self.times = torch.tensor([0., .02, .04, 3., 3.02], dtype=torch.float64)
        self.action = {"drag_x": -80, "drag_y": 10, "release_time_ms": 600, "tap_time_ms": 0}

    def test_streaming_observations_actions_and_intershot_gap_match_training_exactly(self):
        stream = NativeObservationHistory(self.parser, self.history)
        segments = [{"start": i, "action": action_context(self.action)} for i in (0, 3)]
        visual = encode_visual(self.parser, {"images": self.images, "timestamps": self.times})
        expected = encode_history(self.history, visual, self.times, segments)
        actual = []
        for i, (image, timestamp) in enumerate(zip(self.images, self.times)):
            actual.append(stream.observe(image, float(timestamp)))
            if i in (0, 3):
                stream.record_executed_action(self.action, float(timestamp))
        torch.testing.assert_close(torch.stack(actual), expected, atol=2e-5, rtol=2e-5)
        self.assertEqual(stream.executed_actions, 2)
        self.assertEqual(stream.observations, 5)

    def test_new_episode_reset_reproduces_initial_memory(self):
        stream = NativeObservationHistory(self.parser, self.history)
        first = stream.observe(self.images[0], 0.)
        stream.record_executed_action(self.action, 0.)
        stream.observe(self.images[1], .02)
        stream.reset()
        torch.testing.assert_close(stream.observe(self.images[0], 0.), first, atol=0, rtol=0)

    def test_prediction_branches_cannot_mutate_real_observation_or_action_memory(self):
        stream = NativeObservationHistory(self.parser, self.history)
        initial = stream.observe(self.images[0], 0.)
        memory = stream.state.memory.clone()
        actions = [SlingshotAction(-80, 10, 0), SlingshotAction(-60, 45, 0)]
        bounds = SlingshotActionBounds((-160, -40), (-80, 80), (0, 1000))
        for pure in (False, True):
            model = NativeHistoryDynamics(pure=pure, width=16).eval()
            controller = NativeHistoryController(pure, width=16).eval()
            record = score_native_actions(model, controller, initial, actions, bounds, endpoint=750)
            self.assertEqual(record["action"], asdict(actions[record["selected_ordinal"]]))
            self.assertEqual(len(record["candidates"]), 2)
            for candidate in record["candidates"]:
                self.assertEqual(sum(w["horizon_native_steps"] for w in candidate["work"]), 750)
                if pure:
                    self.assertTrue(all(w["symbol_decoder_calls"] == 0 for w in candidate["work"]))
            torch.testing.assert_close(stream.state.memory, memory, atol=0, rtol=0)
            torch.testing.assert_close(stream.current, initial, atol=0, rtol=0)
            self.assertEqual(stream.executed_actions, 0)

    def test_illegal_action_and_exhausted_decision_are_not_silent_fallbacks(self):
        model = NativeHistoryDynamics(pure=True, width=16).eval()
        control = NativeHistoryController(True, width=16).eval()
        bounds = SlingshotActionBounds((-160, -40), (-80, 80), (0, 1000))
        carrier = torch.zeros(VISUAL_DIM + 64)
        with self.assertRaisesRegex(ValueError, "legal"):
            score_native_actions(model, control, carrier, [SlingshotAction(100, 0, 0)], bounds)
        with self.assertRaisesRegex(ValueError, "linear-MAC"):
            score_native_actions(model, control, carrier, [SlingshotAction(-80, 10, 0)], bounds,
                                 endpoint=50, maximum_linear_macs=0)

    def test_png_resize_matches_fitting_transform(self):
        source = Image.fromarray(np.arange(90 * 120 * 3, dtype=np.uint8).reshape(90, 120, 3))
        encoded = BytesIO()
        source.save(encoded, format="PNG")
        expected = torch.from_numpy(np.asarray(source.resize((96, 64), Image.Resampling.BILINEAR)).copy()).permute(2, 0, 1)
        torch.testing.assert_close(agent_image_tensor(encoded.getvalue()), expected)


if __name__ == "__main__":
    unittest.main()
