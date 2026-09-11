from types import SimpleNamespace
import unittest

import torch

from world_model.data.deployment_temporal import AgentObservation, TemporalObservationContext, TemporalVisualCarrierAdapter
from world_model.training.native_history_model import NativeHistoryDynamics as HistoryDynamics, PAIRS, STATE_DIM
from world_model.training.native_history_data import VOCABULARY, VISUAL_DIM, visual_carriers, history_events, native_labels, physical_center
from world_model.training.native_history_fit import (NativeVisualParser, CommonHistoryFit, perception_loss,
    encode_history, transition_batch, continuous_loss, hybrid_loss, NativeHistoryController, controller_targets, rollout)


def synthetic_shard(frames=40, seed=760940001):
    generator = torch.Generator().manual_seed(seed)
    return {"schema": "issue_76_native_history_shard_v1", "synthetic": True, "exposure_role": "training",
            "member_identity": "synthetic-not-research", "base_cluster": "synthetic-not-research",
            "segment_ranges": [{"start": 0, "stop": frames, "censored": False,
                                "action": torch.tensor([-.1, .1, .6, 0., 1.])}],
            "tensors": {"images": torch.randint(0, 256, (frames, 3, 64, 96), generator=generator, dtype=torch.uint8),
                "presence": torch.randint(0, 2, (frames, len(VOCABULARY)), generator=generator).float(),
                "centers": torch.rand(frames, len(VOCABULARY), 2, generator=generator),
                "relations": torch.zeros(frames, len(VOCABULARY), len(VOCABULARY), 2), "relation_mask": torch.ones(frames, len(VOCABULARY), len(VOCABULARY), 2, dtype=torch.bool),
                "macros": torch.zeros(frames, 2), "macro_mask": torch.ones(frames, 2, dtype=torch.bool),
                "timestamps": torch.arange(frames, dtype=torch.float64) * .02,
                "fixed_steps": torch.arange(frames, dtype=torch.int64) * 50}}


class NativeHistoryFitTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7609)

    def test_shared_visual_parser_has_no_symbolic_heads_and_exact_cost(self):
        model = NativeVisualParser()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 241418)
        self.assertFalse(any("relation" in n or "macro" in n for n, _ in model.named_parameters()))
        data = synthetic_shard(2)["tensors"]
        perception_loss(model, data).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        history = CommonHistoryFit()
        self.assertEqual(sum(p.numel() for p in history.parameters()), 69504 + 101952)

    def test_visual_carrier_matches_existing_temporal_interface(self):
        slots = len(VOCABULARY)
        output = {"presence_logits": torch.randn(3, slots), "centers": torch.rand(3, slots, 2),
                  "kind_logits": torch.randn(3, slots, 7)}
        times = torch.tensor([0., .02, .07], dtype=torch.float64)
        actual = visual_carriers(output, times)
        parser = SimpleNamespace(object_vocabulary=VOCABULARY)
        thresholds = {name: .5 for name in ("object_presence", "contact", "supports", "steady-state", "structure-unstable")}
        adapter = TemporalVisualCarrierAdapter(parser, parser_checkpoint_identity="fixture", temperatures={},
                                              thresholds=thresholds, latent_dim=VISUAL_DIM, max_entities=slots)
        parsed = [{"presence": output["presence_logits"][i].sigmoid(), "centers": output["centers"][i],
                   "kinds": output["kind_logits"][i].softmax(-1), "relations": torch.zeros(slots, slots, 2), "macros": torch.zeros(2)} for i in range(3)]
        observations = [AgentObservation(str(i), i, float(t), b"fixture", "agent") for i, t in enumerate(times)]
        for i in range(3):
            expected = adapter.build_from_parsed(TemporalObservationContext(observations[i - 1] if i else None, observations[i]),
                                                 parsed[i], parsed[i - 1] if i else None).tensor
            torch.testing.assert_close(actual[i], expected)

    def test_each_action_event_occurs_once_and_precedes_only_later_memories(self):
        visual = torch.randn(4, VISUAL_DIM)
        times = torch.tensor([0., .02, 2., 2.02], dtype=torch.float64)
        segments = [{"start": i, "action": torch.ones(5)} for i in (0, 2)]
        events, indices = history_events(visual, times, segments)
        self.assertEqual(events["action_mask"].sum(), 2)
        self.assertEqual(indices, [0, 2, 3, 5])
        model = CommonHistoryFit()
        first = encode_history(model.history, visual, times, segments)
        changed = [{"start": s["start"], "action": s["action"] * 2} for s in segments]
        other = encode_history(model.history, visual, times, changed)
        torch.testing.assert_close(first[0], other[0])
        self.assertFalse(torch.allclose(first[1, VISUAL_DIM:], other[1, VISUAL_DIM:]))
        model.loss(visual, times, segments).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_endpoint_masks_and_pure_optimizer_exclude_symbolic_targets(self):
        shard = synthetic_shard()
        carrier = torch.randn(40, STATE_DIM)
        batch = transition_batch(shard, carrier, 750, torch.Generator().manual_seed(1), symbolic=False)
        self.assertEqual(set(batch), {"z", "available", "action"})
        model = HistoryDynamics(pure=True, width=16)
        pair = model.pairs[-1]
        loss = continuous_loss(model, batch["z"], batch["available"], batch["action"], pair)
        changed = batch["z"].clone()
        changed[~batch["available"]] = 999.
        other = continuous_loss(model, changed, batch["available"], batch["action"], pair)
        torch.testing.assert_close(loss, other)
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        hybrid = HistoryDynamics(pure=False, width=16)
        for pair in PAIRS:
            batch = transition_batch(shard, carrier, pair.delta, torch.Generator().manual_seed(1), symbolic=True)
            self.assertTrue(torch.isfinite(hybrid_loss(hybrid, batch, pair)))

    def test_controller_capacity_and_deployment_have_exact_native_time(self):
        counts = [sum(p.numel() for p in NativeHistoryController(pure).parameters()) for pure in (False, True)]
        self.assertLess(abs(counts[0] - counts[1]) / counts[0], .02)
        for pure in (False, True):
            model = HistoryDynamics(pure=pure, width=16)
            controller = NativeHistoryController(pure, width=16)
            carrier, action = torch.randn(40, STATE_DIM), torch.zeros(5)
            labels = controller_targets(model, carrier, list(range(0, 2000, 50)), action)
            self.assertEqual(labels["z"].shape, (39, STATE_DIM))
            predicted, work = rollout(model, controller, carrier[0], action, endpoint=150)
            self.assertEqual(predicted.shape, (STATE_DIM,))
            self.assertEqual(sum(r["horizon_native_steps"] for r in work), 150)
            if pure:
                self.assertTrue(all(r["symbol_decoder_calls"] == 0 for r in work))

    def test_native_macro_uses_unrendered_predecessor_across_chunk_boundary(self):
        base = {"entities": [], "colliders": [], "contacts": []}
        samples = [{**base, "fixed_step": i, "supports": ([{"supporter_entity_id": "a", "supported_entity_id": "b"}] if i == 1 else [])} for i in range(3)]
        chunks = [{"fixed_step_samples": samples[:2], "events": [{"event_type": "stable_exited", "fixed_step": 1}]},
                  {"fixed_step_samples": samples[2:], "events": []}]
        trace = SimpleNamespace(trace=SimpleNamespace(chunks=lambda: iter(chunks)), summary={"last_fixed_step": 2})
        labels = list(native_labels(trace, {0: {}, 2: {}}))
        self.assertFalse(bool(labels[0]["macro_mask"].any()))
        self.assertEqual(labels[1]["macros"].tolist(), [0., 1.])
        self.assertTrue(bool(labels[1]["macro_mask"].all()))
        self.assertEqual(physical_center({"body": None}, [{"enabled": True, "is_trigger": False,
                         "shape": {"kind": "box", "center": [2., 3.]}}]), [2., 3.])


if __name__ == "__main__":
    unittest.main()
