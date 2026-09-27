import numpy as np
import pytest
import torch

from scripts import run_readout_fidelity_jepa as r100
from world_model.data.deployment_temporal import (AgentObservation, TemporalObservationContext,
                                                  TemporalVisualCarrierAdapter)
from world_model.model import Abstraction, PredictionPair
from world_model.training import jepa_slot_encoder as js
from world_model.training.cnn_hybrid import CNNHybridPredictor

VOCABULARY = tuple([f"bird:{i:04d}" for i in range(3)] + [f"block:{i:04d}" for i in range(5)] + ["pig:0000"]
                   + [f"platform:{i:04d}" for i in range(6)] + ["slingshot:0000", "world:landscape:0000",
                                                                 "world:landscape:0001"])


class _Vocabulary(torch.nn.Module):
    object_vocabulary = VOCABULARY


def _adapter():
    return TemporalVisualCarrierAdapter(
        _Vocabulary(), parser_checkpoint_identity="test",
        temperatures={"object_presence": 1.0, "contact": 1.0, "supports": 1.0, "steady-state": 1.0,
                      "structure-unstable": 1.0},
        thresholds={"object_presence": 0.5, "contact": 0.5, "supports": 0.5, "steady-state": 0.5,
                    "structure-unstable": 0.5}, latent_dim=236, max_entities=18)


def _outputs(generator, batch):
    count = len(VOCABULARY)
    return {"presence_logits": torch.randn(batch, count, generator=generator) * 3,
            "centers": torch.rand(batch, count, 2, generator=generator),
            "kind_logits": torch.randn(batch, count, 7, generator=generator),
            "relation_logits": torch.randn(batch, count, count, 2, generator=generator),
            "macro_logits": torch.randn(batch, 2, generator=generator)}


def _parsed(output, index):
    return {"presence": torch.sigmoid(output["presence_logits"][index]), "centers": output["centers"][index],
            "kinds": torch.softmax(output["kind_logits"][index], -1),
            "relations": torch.sigmoid(output["relation_logits"][index]),
            "macros": torch.sigmoid(output["macro_logits"][index])}


def test_differentiable_carrier_matches_the_adapter_with_and_without_prior():
    generator = torch.Generator().manual_seed(3)
    current, prior = _outputs(generator, 4), _outputs(generator, 4)
    has_prior = torch.tensor([0.0, 1.0, 1.0, 1.0])
    elapsed = torch.tensor([0.0, 0.0868, 0.0868, 0.0869])
    z = js.carrier_from_outputs(current, prior, has_prior, elapsed, js.expected_kind_indices(VOCABULARY))
    adapter = _adapter()
    for i in range(4):
        now = AgentObservation(f"f{i}", 30050, 13.0 + float(elapsed[i]), b"", "agent")
        before = AgentObservation(f"p{i}", 30000, 13.0, b"", "agent") if has_prior[i] else None
        reference = adapter.build_from_parsed(TemporalObservationContext(before, now), _parsed(current, i),
                                              _parsed(prior, i) if has_prior[i] else None).tensor
        assert torch.allclose(z[i], reference, atol=1e-4), (i, (z[i] - reference).abs().max())


def test_carrier_gradient_reaches_presence_and_centers_only_where_available():
    generator = torch.Generator().manual_seed(4)
    current, prior = _outputs(generator, 2), _outputs(generator, 2)
    current["centers"].requires_grad_(True)
    current["presence_logits"].requires_grad_(True)
    z = js.carrier_from_outputs(current, prior, torch.ones(2), torch.full((2,), 0.0868),
                                js.expected_kind_indices(VOCABULARY))
    z.sum().backward()
    visible = torch.sigmoid(current["presence_logits"].detach()) >= 0.5
    assert bool((current["centers"].grad.abs().sum(-1)[~visible] == 0).all())
    assert bool((current["centers"].grad.abs().sum(-1)[visible] > 0).all())
    assert bool((current["presence_logits"].grad != 0).all())


def test_ema_target_follows_the_schedule_and_takes_no_gradient():
    online = torch.nn.Linear(3, 2)
    target = js.new_target(online)
    assert not any(p.requires_grad for p in target.parameters())
    before = target.weight.detach().clone()
    with torch.no_grad():
        online.weight.add_(1.0)
    js.ema_update(target, online, 0.75)
    assert torch.allclose(target.weight, 0.75 * before + 0.25 * online.weight)
    assert js.momentum_at(0, 100) == pytest.approx(0.996)
    assert js.momentum_at(100, 100) == pytest.approx(1.0)


def test_jepa_loss_backpropagates_into_context_but_never_into_targets():
    torch.manual_seed(0)
    predictor = CNNHybridPredictor()
    batch, rows = 3, [0, 5, 10, 15, 20]
    context = [torch.randn(batch, 236, requires_grad=True) for _ in range(4)]
    targets = [torch.randn(batch, 236, requires_grad=True) for _ in range(4)]
    labels = {"relations": torch.zeros(batch, 61, 18, 18, 2, dtype=torch.bool),
              "relations_mask": torch.ones(batch, 61, 18, 18, 2, dtype=torch.bool),
              "macros": torch.zeros(batch, 61, 2, dtype=torch.bool),
              "macros_mask": torch.ones(batch, 61, 2, dtype=torch.bool)}
    loss = js.jepa_loss(predictor, context, targets, torch.zeros(batch, 5), torch.full((batch,), 60), labels,
                        PredictionPair(5, Abstraction.MICRO), rows)
    loss.backward()
    assert all(c.grad is not None and bool(c.grad.abs().sum() > 0) for c in context)
    assert all(t.grad is None for t in targets)


def test_short_windows_mask_transitions_past_their_length():
    torch.manual_seed(0)
    predictor = CNNHybridPredictor()
    labels = {"relations": torch.zeros(2, 61, 18, 18, 2, dtype=torch.bool),
              "relations_mask": torch.ones(2, 61, 18, 18, 2, dtype=torch.bool),
              "macros": torch.zeros(2, 61, 2, dtype=torch.bool), "macros_mask": torch.ones(2, 61, 2, dtype=torch.bool)}
    context = [torch.randn(2, 236) for _ in range(4)]
    targets = [torch.randn(2, 236) for _ in range(4)]
    with pytest.raises(ValueError):
        js.jepa_loss(predictor, context, targets, torch.zeros(2, 5), torch.tensor([10, 14]), labels,
                     PredictionPair(15, Abstraction.CONTINUOUS), [0, 15, 30, 45, 60])


def _phase_inputs(count, contact_at, steady):
    names = list(VOCABULARY)
    relations = torch.zeros(count, 18, 18, 2, dtype=torch.bool)
    for p in contact_at:
        relations[p, 0, names.index("pig:0000"), 0] = True
    relations[:, 0, names.index("slingshot:0000"), 0] = True  # resting in the slingshot is not contact onset
    macros = torch.tensor([[s, False] for s in steady], dtype=torch.bool)
    return relations, macros, torch.ones(count, 2, dtype=torch.bool), names


def test_phases_split_at_bird_contact_and_final_settling():
    steady = [True] * 3 + [False] * 4 + [True] * 2 + [False] + [True] * 4
    relations, macros, mask, names = _phase_inputs(len(steady), [4, 5], steady)
    phase, onset, settled = r100.phase_segmentation(relations, macros, mask, names)
    assert onset == 4
    assert settled == 10  # the steady run at 7-8 is interrupted at 9; settling counts only to the end
    assert phase.tolist() == [0] * 4 + [1] * 6 + [2] * 4


def test_phases_without_contact_are_all_pre_contact():
    relations, macros, mask, names = _phase_inputs(6, [], [True] * 6)
    phase, onset, settled = r100.phase_segmentation(relations, macros, mask, names)
    assert (onset, settled) == (None, None)
    assert phase.tolist() == [0] * 6


def test_shard_positions_are_the_77_window_rows():
    assert r100.shard_positions(600) == sorted(set(range(0, 61)) | set(range(180, 241)) | set(range(360, 421))
                                               | set(range(540, 601)))
    assert r100.shard_positions(40) == list(range(0, 41))


def test_clustered_f1_pools_counts_per_seed_and_averages_seeds():
    counts = np.zeros((3, 3, 1, 3))
    counts[:, 0, 0] = [2, 1, 1]      # seed 0: F1 = 2*6 / (12 + 3 + 3) = 2/3 after pooling
    counts[:, 1, 0] = [1, 0, 1]      # seed 1: F1 = 6 / (6 + 0 + 3) = 2/3
    counts[:, 2, 0] = [0, 0, 0]      # seed 2 undefined: ignored
    block = r100.clustered(counts, r100.first(r100.f1_of))
    assert block["mean"] == pytest.approx(2 / 3)
    assert block["interval"][0] == pytest.approx(2 / 3) and block["interval"][1] == pytest.approx(2 / 3)
    empty = r100.clustered(np.zeros((2, 3, 1, 3)), r100.first(r100.f1_of))
    assert empty["mean"] is None and empty["interval"] == [None, None]


def test_contact_counts_unordered_pairs_once_and_respects_the_mask():
    phase = torch.zeros(1, dtype=torch.long)
    prediction = torch.zeros(1, 18, 18, dtype=torch.bool)
    truth = torch.zeros(1, 18, 18, dtype=torch.bool)
    prediction[0, 0, 8] = prediction[0, 8, 0] = True
    truth[0, 0, 8] = truth[0, 8, 0] = True
    prediction[0, 3, 4] = prediction[0, 4, 3] = True          # false positive, masked out below
    mask = torch.ones(1, 18, 18, dtype=torch.bool) & torch.triu(torch.ones(18, 18, dtype=torch.bool), 1)
    mask[0, 3, 4] = False
    tp, fp, fn, _ = r100.confusion(prediction, truth, mask, phase)[0]
    assert (tp, fp, fn) == (1, 0, 0)
