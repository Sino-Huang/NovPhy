import unittest

import torch

from world_model.training.fixed_development import (
    CURVE_TIMES, FAILURE_MSE, field_metrics, first_shot_indices,
    observed_context_prediction, recursive_curves,
)
from world_model.training.native_history_data import VISUAL_DIM, VOCABULARY
from world_model.training.native_history_fit import rollout
from world_model.training.native_history_model import CONTINUOUS_PAIRS, NativeHistoryDynamics, STATE_DIM


class Increment:
    pairs = CONTINUOUS_PAIRS

    def carrier(self, current, action, pair):
        return current + action[:, :1] * pair.delta


class FixedDevelopmentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(76)

    def test_all_horizons_have_exact_uninterrupted_common_times(self):
        initial = torch.zeros(2, STATE_DIM)
        action = torch.ones(2, 5)
        action[1, 0] = 2
        for pair in CONTINUOUS_PAIRS:
            curves = recursive_curves(Increment(), initial, action, pair)
            self.assertEqual(tuple(curves), CURVE_TIMES)
            for elapsed, row in curves.items():
                self.assertTrue(torch.equal(row["predicted"][:, 0], torch.tensor([elapsed, 2 * elapsed])))
                self.assertEqual(row["transitions_per_member"], elapsed // pair.delta)
                self.assertFalse(row["failed"].any())
            local = observed_context_prediction(Increment(), initial + 7, action, pair)
            self.assertEqual(float(local["predicted"][0, 0]), 7 + pair.delta)
        self.assertFalse(initial.any())

    def test_exact_first_shot_indices_do_not_borrow_later_shot_endpoint(self):
        shard = {"segment_ranges": [{"start": 1, "stop": 5}, {"start": 5, "stop": 7}],
                 "tensors": {"fixed_steps": torch.tensor([0, 100, 600, 850, 1600, 100, 11350])}}
        indices = first_shot_indices(shard)
        self.assertEqual(indices["initial"], 1)
        self.assertEqual(indices["targets"], {750: 3, 1500: 4, 3000: None, 6000: None, 11250: None})
        self.assertEqual(indices["local_contexts"][750][750], 1)
        self.assertEqual(indices["local_contexts"][250][750], 2)
        self.assertIsNone(indices["local_contexts"][50][750])

    def test_real_models_match_existing_single_member_rollouts(self):
        for pure in (False, True):
            model = NativeHistoryDynamics(pure=pure, width=16).eval()
            before = {key: value.clone() for key, value in model.state_dict().items()}
            initial, action = torch.randn(2, STATE_DIM), torch.randn(2, 5)
            for pair in model.pairs:
                batch = recursive_curves(model, initial, action, pair, times=(750,))[750]
                for i in range(2):
                    expected, work = rollout(model, None, initial[i], action[i], endpoint=750, fixed_pair=pair)
                    torch.testing.assert_close(batch["predicted"][i], expected, rtol=1e-4, atol=1e-5)
                    self.assertEqual(batch["transitions_per_member"], len(work))
            self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in before.items()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_invalid_path_rejected_and_failure_cannot_recover_silently(self):
        pair = CONTINUOUS_PAIRS[0]
        initial, action = torch.zeros(2, STATE_DIM), torch.ones(2, 5)
        with self.assertRaises(ValueError):
            recursive_curves(Increment(), initial, action, pair, times=(751,))

        class Recovering(Increment):
            def carrier(self, current, action, pair):
                result = current.nan_to_num() + 1
                result[0, 0] = torch.nan if current[0, 0] == 0 else 1
                return result

        rows = recursive_curves(Recovering(), initial, action, pair, times=(50, 100))
        self.assertTrue(torch.isfinite(rows[100]["predicted"]).all())
        self.assertEqual(rows[100]["failed"].tolist(), [True, False])

    def test_engine_slots_masks_counts_and_failure_penalty(self):
        target = torch.zeros(3, STATE_DIM)
        predicted = target.clone()
        presence = torch.zeros(3, len(VOCABULARY))
        centers = torch.full((3, len(VOCABULARY), 2), torch.nan)
        presence[0, 10] = 1
        centers[0, 10] = torch.tensor([.25, .5])
        predicted[0, 2 + 13 * 10] = .75
        predicted[0, 7 + 13 * 10] = .75
        predicted[0, 8 + 13 * 10] = 1.
        predicted[0, VISUAL_DIM:] = 3
        predicted[2, 0] = torch.inf
        rows = field_metrics(predicted, target, presence, centers)
        self.assertEqual(rows[0]["center_mse_to_engine"], .25)
        self.assertEqual(rows[0]["pig_count_absolute_error"], .25)
        self.assertEqual(rows[0]["block_count_absolute_error"], 0)
        self.assertEqual(rows[0]["memory_mse"], 9)
        self.assertEqual(rows[0]["carrier_absolute_bound_excess"], 1)
        self.assertIsNone(rows[1]["center_mse_to_engine"])
        self.assertEqual(rows[2]["carrier_mse"], FAILURE_MSE)
        self.assertIsNotNone(rows[2]["failure"])
        target[0, 0] = torch.nan
        with self.assertRaises(ValueError):
            field_metrics(predicted, target, presence, centers)


if __name__ == "__main__":
    unittest.main()
