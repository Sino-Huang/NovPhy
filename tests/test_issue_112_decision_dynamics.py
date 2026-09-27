import copy

import pytest
import torch
from torch.nn import functional as F

from world_model.model import Abstraction, PredictionPair
from world_model.training import decision_dynamics as dd
from world_model.training.cnn_hybrid import PAIRS, CNNHybridPredictor, training_loss
from world_model.training.matched_dynamics import ContinuousDynamics, continuous_loss

WIDTH = 64
LENGTH = 60


def _hybrid(seed):
    torch.manual_seed(seed)
    return CNNHybridPredictor()


def _continuous(seed):
    torch.manual_seed(seed)
    return ContinuousDynamics(WIDTH)


def _windows(generator, rows, lengths):
    z = torch.randn(rows, LENGTH + 1, 236, generator=generator) * 0.3
    return {"z": z, "action": torch.rand(rows, 5, generator=generator),
            "length": torch.tensor(lengths),
            "relations": torch.rand(rows, LENGTH + 1, 18, 18, 2, generator=generator) > 0.8,
            "relations_mask": torch.rand(rows, LENGTH + 1, 18, 18, 2, generator=generator) > 0.3,
            "macros": torch.rand(rows, LENGTH + 1, 2, generator=generator) > 0.5,
            "macros_mask": torch.rand(rows, LENGTH + 1, 2, generator=generator) > 0.4}


def _stacked(batches):
    return {k: torch.stack([b[k] for b in batches]) for k in batches[0]}


def _ensemble(models):
    return dd.Ensemble(models, learning_rate=1e-3, weight_decay=1e-4, grad_clip=1.0, compile=False)


@pytest.mark.parametrize("pair", [PredictionPair(1, Abstraction.MICRO), PredictionPair(5, Abstraction.MACRO),
                                  PredictionPair(15, Abstraction.CONTINUOUS), PredictionPair(15, Abstraction.MICRO)])
def test_masked_local_loss_equals_the_issue_77_hybrid_loss_per_member(pair):
    generator = torch.Generator().manual_seed(1)
    # short terminal windows: some rows lose later transitions (masked, not relabeled)
    batches = [_windows(generator, 6, [60, 3, 12, 60, 30, 44]), _windows(generator, 6, [2, 60, 60, 16, 1, 59])]
    models = [_hybrid(11), _hybrid(12)]
    ensemble = _ensemble(models)
    ours = dd.local_loss(ensemble, _stacked(batches), pair)
    for index, (model, batch) in enumerate(zip(models, batches)):
        reference = training_loss(model, batch, pair, True)
        assert torch.allclose(ours[index], reference, rtol=1e-5, atol=1e-6), (index, ours[index], reference)


@pytest.mark.parametrize("delta", [1, 5, 15])
def test_masked_local_loss_equals_the_issue_74_continuous_loss_per_member(delta):
    generator = torch.Generator().manual_seed(2)
    pair = PredictionPair(delta, Abstraction.CONTINUOUS)
    batches = [_windows(generator, 5, [60, 4, 20, 45, 7]), _windows(generator, 5, [60, 60, 1, 33, 15])]
    models = [_continuous(3), _continuous(4)]
    ours = dd.local_loss(_ensemble(models), _stacked([{k: b[k] for k in ("z", "action", "length")}
                                                       for b in batches]), pair)
    for index, (model, batch) in enumerate(zip(models, batches)):
        reference = continuous_loss(model, batch["z"], batch["action"], batch["length"], pair)
        assert torch.allclose(ours[index], reference, rtol=1e-5, atol=1e-6)


def test_ensemble_updates_equal_independent_adamw_with_per_member_clipping():
    generator = torch.Generator().manual_seed(5)
    models = [_hybrid(21), _hybrid(22), _hybrid(23)]
    singles = [copy.deepcopy(m) for m in models]
    ensemble = _ensemble(models)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4) for m in singles]
    for step in range(3):
        pair = PAIRS[(step * 4) % 9]
        batches = [_windows(generator, 4, [60, 60, 20, 7]) for _ in models]
        # members start this step from identical parameters: re-sync away Adam's float-noise sign flips
        with torch.no_grad():
            for index, model in enumerate(singles):
                for name, value in model.named_parameters():
                    value.copy_(ensemble.params[name][index])
        ensemble.update(dd.local_loss(ensemble, _stacked(batches), pair))
        for index, (model, optimizer, batch) in enumerate(zip(singles, optimizers, batches)):
            optimizer.zero_grad(set_to_none=True)
            training_loss(model, batch, pair, True).backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            assert norm > 1.0  # the clip is exercised
            for name, value in model.named_parameters():
                stacked = ensemble.params[name].grad
                assert (value.grad is None) == (stacked is None), name  # unused heads stay untouched
                if value.grad is not None:
                    assert torch.allclose(stacked[index], value.grad, rtol=1e-4, atol=1e-7), (step, index, name)
            optimizer.step()
    for index, model in enumerate(singles):
        state = ensemble.member_state(index)
        for name, value in model.state_dict().items():
            # Adam divides by sqrt(v): an update moves every element by at most ~lr
            assert torch.allclose(state[name], value, atol=3e-3), (index, name)


def test_a_nonfinite_member_is_retired_without_touching_the_others():
    generator = torch.Generator().manual_seed(6)
    models = [_continuous(31), _continuous(32)]
    reference = _ensemble([copy.deepcopy(models[1])])
    ensemble = _ensemble(models)
    pair = PredictionPair(5, Abstraction.CONTINUOUS)
    for step in range(2):
        batch = {k: v for k, v in _windows(generator, 4, [60, 60, 60, 60]).items() if k in ("z", "action", "length")}
        broken = {k: v.clone() for k, v in batch.items()}
        if step == 0:
            broken["z"][0, 0, 0] = float("nan")
        report = ensemble.update(dd.local_loss(ensemble, _stacked([broken, batch]), pair))
        ensemble.mark_retired_step(step, report["newly_retired"])
        reference.update(dd.local_loss(reference, _stacked([batch]), pair))
    assert ensemble.retired.tolist() == [True, False] and ensemble.retired_at == [0, None]
    for name, value in reference.member_state(0).items():
        assert torch.allclose(ensemble.member_state(1)[name], value, atol=1e-7)


def test_ranking_loss_is_the_mean_pairwise_logistic_inside_each_mixed_cell():
    cost = torch.tensor([[[0.2, 0.5, 0.9, 7.0], [0.1, 0.3, 0.0, 0.0]]])
    verdict = torch.tensor([[[1, 0, 0, -1], [0, 0, -1, -1]]])  # second cell has no success
    value = dd.pairwise_ranking(cost, verdict, temperature=0.1)
    expected = (F.softplus(torch.tensor(-3.0)) + F.softplus(torch.tensor(-7.0))) / 2
    assert torch.allclose(value[0], expected)
    cost = cost.clone().requires_grad_()
    dd.pairwise_ranking(cost, verdict, temperature=0.1).sum().backward()
    grad = cost.grad[0]
    assert grad[0, 0] > 0 and grad[0, 1] < 0 and grad[0, 2] < 0  # descent lowers success, raises failures
    assert float(grad[0, 3]) == 0.0 and float(grad[1].abs().sum()) == 0.0


def test_tie_free_cost_matches_the_scoring_harness_endpoint_cost():
    from world_model.planning.scoring_harness import EndpointCosts
    from world_model.planning.task_objective import TaskObjective
    vocabulary = tuple([f"bird:{i:04d}" for i in range(3)] + [f"block:{i:04d}" for i in range(5)] + ["pig:0000"]
                       + [f"platform:{i:04d}" for i in range(6)] + ["slingshot:0000", "world:landscape:0000",
                                                                    "world:landscape:0001"])
    costs = EndpointCosts(TaskObjective.from_vocabulary(vocabulary))
    generator = torch.Generator().manual_seed(7)
    z_end, z0 = torch.randn(3, 236, generator=generator), torch.randn(236, generator=generator)
    ours = dd.tie_free_cost(z_end, z0, costs.presence, costs.center)
    for row in range(3):
        assert abs(float(ours[row]) - costs.row(row, z_end[row], z0)["tie_free"]) < 1e-5


def test_fused_recursion_equals_separate_long_and_ranking_recursions():
    generator = torch.Generator().manual_seed(8)
    pair = PredictionPair(5, Abstraction.CONTINUOUS)
    ensemble = _ensemble([_continuous(41), _continuous(42)])
    horizon, endpoint = 30, 60
    long = {"start": torch.randn(2, 3, 236, generator=generator) * 0.2,
            "action": torch.rand(2, 3, 5, generator=generator),
            "targets": torch.randn(2, 3, horizon // 5, 236, generator=generator) * 0.2}
    rank = {"z0": torch.randn(2, 2, 236, generator=generator) * 0.2,
            "action": torch.rand(2, 2, 4, 5, generator=generator),
            "verdict": torch.tensor([[[1, 0, 0, -1], [0, 1, 1, 0]]] * 2)}
    kwargs = dict(horizon=horizon, endpoint=endpoint, presence=106, center=(111, 112), temperature=0.1)
    fused = dd.long_and_rank_loss(ensemble, pair, long=long, rank=rank, **kwargs)
    long_only = dd.long_and_rank_loss(ensemble, pair, long=long, rank=None, **kwargs)
    rank_only = dd.long_and_rank_loss(ensemble, pair, long=None, rank=rank, **kwargs)
    assert torch.allclose(fused[0], long_only[0], atol=1e-6) and torch.allclose(fused[1], rank_only[1], atol=1e-6)
    # the long term is the mean over every visited delta multiple of row-mean MSE + bound
    z, total = long["start"], 0
    for step in range(horizon // 5):
        z = ensemble.transition(pair, z, long["action"])
        total = total + ((z - long["targets"][:, :, step]).square().mean(-1)
                         + 0.01 * F.relu(z.abs() - 2).square().mean(-1)).mean(1)
    assert torch.allclose(long_only[0], total / (horizon // 5), atol=1e-6)


def test_curriculum_and_ramp_boundaries():
    horizons = [dd.horizon_at(s, first=60, last=225, increment=15, every=500) for s in (0, 499, 500, 5499, 5500, 8999)]
    assert horizons == [60, 60, 75, 210, 225, 225]
    assert [dd.ramp_at(s, start=1000, length=1000) for s in (0, 999, 1000, 1999, 5000)] == [0.0, 0.0, 0.001, 1.0, 1.0]
