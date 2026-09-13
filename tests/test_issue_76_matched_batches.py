from collections import Counter, defaultdict
import unittest

import torch

from scripts.issue_76_matched_batches import indexed_batch, mixed_schedule, sampled_schedule
from scripts.run_issue_76_native_refit import generator, pair_for_update
from tests.test_native_history_fit import synthetic_shard
from world_model.training.native_history_fit import transition_batch
from world_model.training.native_history_model import STATE_DIM


class MatchedBatchTests(unittest.TestCase):
    def test_exact_pair_exposures_skips_and_rng_preserved(self):
        for pure in (False, True):
            original = sampled_schedule([40, 0, 53, 61, 80], 761, pure, 120)
            state = torch.random.get_rng_state().clone()
            mixed = mixed_schedule(original, 761, pure)
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            self.assertEqual(set(original), set(mixed))
            self.assertFalse(any(u % 5 == 1 for u in mixed))
            counts = []
            for schedule in (original, mixed):
                grouped = defaultdict(Counter)
                for update, rows in schedule.items():
                    grouped[pair_for_update(pure, update)].update(map(tuple, rows.tolist()))
                counts.append(grouped)
            self.assertEqual(*counts)
            self.assertTrue(all(torch.equal(v, mixed_schedule(original, 761, pure)[u])
                                for u, v in mixed.items()))
            self.assertTrue(any(len(rows[:, 0].unique()) > 1 for rows in mixed.values()))

    def test_explicit_rows_match_native_masks_actions_and_symbolic_targets(self):
        shards = [synthetic_shard(40), synthetic_shard(53)]
        # Repeated step coordinates across shot segments must not cross shots.
        for shard in shards:
            n = len(shard["tensors"]["fixed_steps"])
            shard["segment_ranges"] = [
                {"start": 0, "stop": 20, "action": torch.ones(5)},
                {"start": 20, "stop": n, "action": -torch.ones(5)}]
            shard["tensors"]["fixed_steps"][20:] -= 1000
            shard["tensors"]["relations"].normal_()
        carriers = [torch.randn(len(s["tensors"]["fixed_steps"]), STATE_DIM) for s in shards]
        for pure in (False, True):
            original = sampled_schedule([40, 53], 761, pure, 18)
            for update, rows in original.items():
                entry = update % 2
                horizon = pair_for_update(pure, update).delta
                expected = transition_batch(shards[entry], carriers[entry], horizon,
                                            generator(761, update), symbolic=not pure)
                actual = indexed_batch(shards, carriers, rows, horizon, symbolic=not pure)
                self.assertEqual(set(actual), set(expected))
                for key in expected:
                    self.assertTrue(torch.equal(actual[key], expected[key]), key)
            mixed = mixed_schedule(original, 761, pure)
            for update, rows in mixed.items():
                horizon = pair_for_update(pure, update).delta
                actual = indexed_batch(shards, carriers, rows, horizon, symbolic=not pure)
                for i, row in enumerate(rows):
                    single = indexed_batch(shards, carriers, row[None], horizon, symbolic=not pure)
                    for key in actual:
                        self.assertTrue(torch.equal(actual[key][i], single[key][0]), key)


if __name__ == "__main__":
    unittest.main()
