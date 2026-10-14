"""Issue #105: the hierarchical predictor's interface, capacity matching and ensemble compatibility."""
import pytest
import torch

from world_model.training import decision_dynamics as dd
from world_model.training import hierarchical_dynamics as hd
from world_model.training.cnn_hybrid import PAIRS, CNNHybridPredictor


def test_width_is_the_parameter_matched_width():
    width, target, counts = hd.matched_width()
    assert width == hd.WIDTH
    assert target == hd.parameter_count(CNNHybridPredictor())
    assert abs(counts[width] - target) / target < 0.002


def test_levels_are_the_diagonal_requests():
    keys = [(p.delta, p.abstraction.value) for p in hd.LEVEL_PAIRS]
    assert keys == [(1, "continuous"), (5, "micro"), (15, "macro")]
    assert all(p in PAIRS for p in hd.LEVEL_PAIRS)


def test_carrier_shapes_and_off_diagonal_rejection():
    torch.manual_seed(0)
    model = hd.HierarchicalPredictor()
    z, action = torch.randn(3, 236), torch.randn(3, 5)
    for pair in PAIRS:
        if pair in hd.LEVEL_OF:
            out = model.carrier(z, action, pair)
            assert out.shape == (3, 236) and torch.isfinite(out).all()
        else:
            with pytest.raises(ValueError):
                model.carrier(z, action, pair)
    with pytest.raises(ValueError):
        model.carrier(torch.randn(3, 235), action, hd.LEVEL_PAIRS[0])


def test_top_down_coupling_reaches_every_coarser_level():
    torch.manual_seed(1)
    model = hd.HierarchicalPredictor()
    for block in model.levels[2].blocks:  # zero-initialised FiLM: open the paths so gradients are non-trivial
        torch.nn.init.normal_(block.modulation.weight, std=0.1)
    for block in model.levels[1].blocks:
        torch.nn.init.normal_(block.modulation.weight, std=0.1)
    for block in model.levels[0].blocks:
        torch.nn.init.normal_(block.modulation.weight, std=0.1)
    z, action = torch.randn(4, 236), torch.randn(4, 5)
    model.carrier(z, action, hd.LEVEL_PAIRS[0]).square().sum().backward()
    for level in range(3):
        assert model.levels[level].input_projection.weight.grad.abs().sum() > 0
    assert model.top_code.grad.abs().sum() > 0
    model.zero_grad()
    model.carrier(z, action, hd.LEVEL_PAIRS[2]).square().sum().backward()
    assert model.levels[0].input_projection.weight.grad is None
    assert model.levels[1].input_projection.weight.grad is None


def test_symbolic_heads_match_the_hybrid_readout():
    model = hd.HierarchicalPredictor()
    z = torch.rand(2, 236)
    relations, availability = model.symbols(z, hd.LEVEL_PAIRS[1].abstraction)
    assert relations.shape == (2, 18, 18, 2) and availability.shape == (2, 2)
    macros, macro_availability = model.symbols(z, hd.LEVEL_PAIRS[2].abstraction)
    assert macros.shape == (2, 2) and macro_availability.shape == (2, 2)


def test_linear_macs_include_coarser_levels():
    model = hd.HierarchicalPredictor()
    top = hd.linear_macs(model, hd.LEVEL_PAIRS[2])
    middle = hd.linear_macs(model, hd.LEVEL_PAIRS[1])
    bottom = hd.linear_macs(model, hd.LEVEL_PAIRS[0])
    assert 0 < top < middle < bottom


def test_trains_under_the_stacked_ensemble():
    models = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        models.append(hd.HierarchicalPredictor())
    ensemble = dd.Ensemble(models, learning_rate=1e-4, weight_decay=1e-4, grad_clip=1.0, compile=False)
    z, action = torch.randn(2, 3, 236), torch.randn(2, 3, 5)
    for pair in hd.LEVEL_PAIRS:
        out = ensemble.transition(pair, z, action)
        assert out.shape == (2, 3, 236)
        expected = models[1].carrier(z[1], action[1], pair)
        assert torch.allclose(out[1], expected, atol=1e-5)


def test_fixed_rollout_lands_on_the_endpoint():
    torch.manual_seed(2)
    model = hd.HierarchicalPredictor().eval()
    z0, actions = torch.rand(236), torch.rand(4, 5)
    for pair in hd.LEVEL_PAIRS:
        z, timing = hd.fixed_rollout(model, z0, actions, pair, hd.linear_macs(model, pair))
        assert z.shape == (4, 236)
        assert timing["transition_calls"] == 4 * (225 // pair.delta)
        assert timing["linear_macs"] == timing["transition_calls"] * hd.linear_macs(model, pair)
