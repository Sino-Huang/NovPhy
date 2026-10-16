"""Issue #113 item 3: the temporal macro head interface."""
import numpy as np
import torch

from world_model.training import temporal_macro_head as tmh


def test_head_interface_engine_and_rollout_histories_agree():
    torch.manual_seed(0)
    carriers = torch.randn(40, tmh.DIM)
    positions = torch.tensor([0, 7, 39])
    history = tmh.gather_history(carriers, torch.zeros(3, dtype=torch.long), positions, 5, 3)
    assert history.shape == (3, 3, tmh.DIM)
    assert torch.equal(history[0], carriers[[0, 0, 0]])
    assert torch.equal(history[1], carriers[[0, 2, 7]])
    context = carriers[[25, 30]][None]                 # observed frames 25, 30 (anchor 30, Delta 5)
    landings = carriers[[35]][None]                    # predicted landing at 35
    rolled = tmh.rollout_history(context, landings, 5, 3)
    assert torch.equal(rolled[0], carriers[[25, 30, 35]])
    head = tmh.TemporalMacroHead(3)
    assert head(history, 5).shape == (3, 2)
    single = tmh.TemporalMacroHead(1)
    assert single(history[:, -1:], 1).shape == (3, 2)


def test_features_are_landing_carrier_then_newest_differences():
    history = torch.arange(3 * tmh.DIM, dtype=torch.float32).reshape(1, 3, tmh.DIM)
    features = tmh.TemporalMacroHead.raw_features(history)
    assert torch.equal(features[0, :tmh.DIM], history[0, 2])
    assert torch.equal(features[0, tmh.DIM:2 * tmh.DIM], history[0, 2] - history[0, 1])
    assert torch.equal(features[0, 2 * tmh.DIM:], history[0, 1] - history[0, 0])


def test_balanced_loss_masks_unavailable_rows():
    logits = torch.zeros(4, 2)
    values = torch.tensor([[1, 0], [0, 0], [1, 1], [0, 1]], dtype=torch.bool)
    available = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=torch.bool)
    weight = tmh.pos_weights(values, available)
    assert weight[0] == 1.0
    loss = tmh.balanced_loss(logits, values, available, weight)
    assert np.isclose(float(loss), float(np.log(2)), atol=1e-6)
