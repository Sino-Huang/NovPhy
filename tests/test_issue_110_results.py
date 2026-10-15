import unittest

import numpy as np
import torch

from world_model.data import novelty_contract as c
from world_model.data import novelty_labels_v2 as L
from world_model.training import novelty_v2 as N


def entity(slot, position, velocity=(0.0, 0.0), body_type="dynamic"):
    return {"entity_id": f"runtime:{slot}", "scenario_object_id": slot, "lifecycle": "active", "body_present": True,
            "body": {"body_type": body_type, "position": list(position), "velocity": list(velocity),
                     "gravity_scale": 1.0}, "contact_ids": []}


def circle(slot, center, radius=0.2, trigger=False):
    return {"collider_id": f"runtime:{slot}:collider:0000", "entity_id": f"runtime:{slot}", "enabled": True,
            "is_trigger": trigger, "shape": {"kind": "circle", "center": list(center), "radius": radius}}


def fan_sample(bird_at, block_at=(30.0, 30.0)):
    return {"entities": [entity("novelty:0000", (0.0, 0.0), body_type="kinematic"), entity("bird:0000", bird_at),
                         entity("block:0000", block_at), entity("platform:0000", (1.0, 0.0))],
            "colliders": [{"collider_id": "runtime:novelty:0000:collider:0001", "entity_id": "runtime:novelty:0000",
                           "enabled": True, "is_trigger": True,
                           "shape": {"kind": "box", "center": [0.0, 0.0], "size": [8, 1], "angle_degrees": 0}},
                          circle("bird:0000", bird_at), circle("block:0000", block_at), circle("platform:0000", (1.0, 0.0))]}


class LabelsV2(unittest.TestCase):
    def test_vocabulary_layout(self):
        self.assertEqual(L.SLOTS, 28)
        self.assertEqual(c.VOCABULARY_V2[L.PIG_SLOT], "pig:0000")
        self.assertEqual(N.PIG_PRESENCE, 6 + 13 * L.PIG_SLOT)
        self.assertEqual(N.DIM, 370)

    def test_pack_roundtrip(self):
        generator = torch.Generator().manual_seed(1)
        dense = torch.rand(5, 28, 28, 3, generator=generator) > 0.7
        self.assertTrue(torch.equal(L.unpack(L.pack(dense)), dense))

    def test_window_any_is_the_per_delta_rule(self):
        flags = torch.tensor([False, True, False, False, False, False, True])
        out = L.window_any(flags, 3)
        self.assertEqual(out.tolist(), [False, True, True, True, False, False, True])
        for t in range(len(flags)):
            self.assertEqual(bool(out[t]), bool(flags[max(0, t - 2):t + 1].any()))

    def test_relation_mask_needs_both_present_and_available(self):
        presence = torch.zeros(28, dtype=torch.bool)
        presence[[0, 13]] = True
        mask = L.relation_mask(presence, torch.tensor([True, False, True]))
        self.assertTrue(mask[0, 13, 0] and mask[13, 0, 2])
        self.assertFalse(mask[0, 13, 1] or mask[0, 0, 0] or mask[0, 1, 0])

    def test_repaired_fan_pushes_right_inside_the_8_by_1_box(self):
        edges = L.fan_edges(fan_sample((-3.9, 0.0)))
        self.assertEqual(edges, {"bird:0000": (50.0, 0.0)})
        self.assertEqual(L.fan_edges(fan_sample((-4.3, 0.0))), {})        # outside (with radius 0.2)
        self.assertEqual(L.fan_edges(fan_sample((3.0, 0.65))), {"bird:0000": (50.0, 0.0)})  # right half, edge overlap
        self.assertEqual(L.fan_edges(fan_sample((3.0, 0.8))), {})
        both = L.fan_edges(fan_sample((0.0, 0.0), block_at=(2.0, 0.0)))
        self.assertEqual(set(both), {"bird:0000", "block:0000"})          # every non-platform body, platforms excluded

    def test_force_edges_dispatches_the_repaired_fan(self):
        source = c.Source(slot="novelty:0000", type="Fan", position=(0.0, 0.0))
        self.assertEqual(L.force_edges(source, fan_sample((-1.0, 0.0)), {}, set(), False),
                         {"bird:0000": (50.0, 0.0)})
        self.assertEqual(L.force_edges(None, fan_sample((-1.0, 0.0)), {}, set(), False), {})

    def test_novelty_kinds(self):
        fan = c.Source(slot="novelty:0000", type="Fan", position=(0.0, 0.0))
        magnet = c.Source(slot="novelty:0000", type="Magnet", position=(0.0, 0.0))
        self.assertEqual(L.KINDS[int(L.slot_kinds(fan)[L.NOVELTY_SLOT])], "fan")
        self.assertEqual(L.KINDS[int(L.slot_kinds(magnet)[L.NOVELTY_SLOT])], "block")
        self.assertEqual(L.KINDS[int(L.slot_kinds(None)[L.PIG_SLOT])], "pig")


class ModelsV2(unittest.TestCase):
    def test_carrier_values_layout(self):
        T = 3
        presence = torch.zeros(T, 28)
        presence[:, L.PIG_SLOT] = 1.0
        centers = torch.zeros(T, 28, 2)
        centers[:, L.PIG_SLOT] = torch.tensor([[0.5, 0.5], [0.6, 0.5], [0.7, 0.5]])
        kinds = torch.zeros(T, 28, 9)
        kinds[..., 1] = 1.0
        carriers = N.carrier_values(presence, centers, kinds, torch.tensor([0.0, 0.02, 0.04], dtype=torch.float64))
        self.assertEqual(carriers.shape, (T, 370))
        self.assertEqual(carriers[0, :2].tolist(), [0.0, 0.0])
        self.assertAlmostEqual(float(carriers[1, 1]), 0.02, places=6)
        self.assertEqual(float(carriers[2, N.PIG_PRESENCE]), 1.0)
        self.assertAlmostEqual(float(carriers[2, N.PIG_CENTER[0]]), 0.7, places=6)
        self.assertAlmostEqual(float(carriers[2, N.PIG_PRESENCE + 8]), 5.0, places=4)  # motion x per second
        self.assertEqual(float(carriers[0, N.PIG_PRESENCE + 10]), 0.0)                 # no prior at position 0
        self.assertAlmostEqual(float(carriers[0, N.PIG_PRESENCE + 11]), 1.0)            # expected kind (pig)

    def test_masked_errors_ignore_absent_slots(self):
        target = torch.zeros(1, 370)
        prediction = target.clone()
        prediction[0, N.SLOT0 + 13 * 3 + 5] = 1.0      # an absent slot
        presence = torch.zeros(1, 28, dtype=torch.bool)
        presence[0, L.PIG_SLOT] = True
        masked, position = N.masked_errors(prediction, target, presence)
        self.assertEqual(float(masked[0]), 0.0)
        prediction[0, N.PIG_CENTER[0]] = 1.0
        masked, position = N.masked_errors(prediction, target, presence)
        self.assertAlmostEqual(float(masked[0]), 1.0 / 15)
        self.assertAlmostEqual(float(position[0]), 0.5)

    def test_predictor_shapes(self):
        from world_model.training.cnn_hybrid import PAIRS
        torch.manual_seed(0)
        model = N.HybridPredictorV2()
        z, a = torch.randn(2, 370), torch.randn(2, 5)
        for pair in PAIRS:
            self.assertEqual(model.carrier(z, a, pair).shape, (2, 370))
        relations, availability = model.symbols(z, PAIRS[1].abstraction)
        self.assertEqual(relations.shape, (2, 28, 28, 3))
        self.assertEqual(availability.shape, (2, 3))
        self.assertTrue(torch.allclose(relations[..., 0], relations[..., 0].transpose(1, 2)))
        from world_model.training.matched_dynamics import CONTINUOUS_PAIRS, pairs_for
        continuous = N.ContinuousDynamicsV2(64)
        self.assertEqual(pairs_for(continuous), CONTINUOUS_PAIRS)
        self.assertEqual(continuous.carrier(z, a, CONTINUOUS_PAIRS[0]).shape, (2, 370))

    def test_scene_inputs_clamp_at_the_shot_start(self):
        carriers = torch.arange(4 * 370, dtype=torch.float32).reshape(4, 370)
        x = N.scene_inputs(carriers, torch.tensor([3]), torch.tensor([2]), k=3)
        keep = torch.cat((carriers[:, :2], carriers[:, 6:]), -1)
        expected = torch.cat((keep[[2, 2, 3]].flatten(), (keep[[2, 3]] - keep[[2, 2]]).flatten()))
        self.assertTrue(torch.equal(x[0], expected))


class Statistics(unittest.TestCase):
    def test_auroc(self):
        from scripts import run_novelty_coverage as r
        self.assertEqual(r.auroc([0.1, 0.2, 0.9, 0.8], [0, 0, 1, 1]), 1.0)
        self.assertEqual(r.auroc([0.5, 0.5], [0, 1]), 0.5)
        self.assertIsNone(r.auroc([0.5, 0.6], [1, 1]))
        rng = np.random.default_rng(0)
        scores = np.r_[rng.normal(0, 1, 500), rng.normal(1, 1, 500)]
        truth = np.r_[np.zeros(500, bool), np.ones(500, bool)]
        low, high = r.auroc_bootstrap(scores, truth, [i % 25 for i in range(1000)])
        point = r.auroc(scores, truth)
        self.assertLess(low, point)
        self.assertGreater(high, point)

    def test_verdict_rule(self):
        from scripts import run_novelty_coverage as r
        good = {"positives": 50, "negatives": 50, "auroc": 0.8, "interval": [0.7, 0.9]}
        self.assertEqual(r.verdict(good, {"auroc": 0.95}), "identifiable")
        self.assertEqual(r.verdict({**good, "auroc": 0.6, "interval": [0.5, 0.7]}, {"auroc": 0.95}), "not_identifiable")
        self.assertEqual(r.verdict({**good, "auroc": 0.76, "interval": [0.6, 0.85]}, {"auroc": 0.95}), r.STOP_TOKEN)
        self.assertEqual(r.verdict({**good, "positives": 5}, {"auroc": 0.95}), r.STOP_TOKEN)
        self.assertEqual(r.verdict(good, {"auroc": 0.7}), "label_invalid")


if __name__ == "__main__":
    unittest.main()
