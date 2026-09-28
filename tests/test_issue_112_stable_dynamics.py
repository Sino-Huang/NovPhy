import copy

import torch

from world_model.model import Abstraction, PredictionPair
from world_model.training import decision_dynamics as dd
from world_model.training import stable_decision_dynamics as sd
from world_model.training.matched_dynamics import ContinuousDynamics

WIDTH = 64
KWARGS = dict(learning_rate=1e-3, weight_decay=1e-2, grad_clip=1.0, compile=False)


def _model(seed):
    torch.manual_seed(seed)
    return ContinuousDynamics(WIDTH)


def _inputs(generator, horizon):
    long = {"start": torch.randn(2, 3, 236, generator=generator) * 0.2,
            "action": torch.rand(2, 3, 5, generator=generator),
            "targets": torch.randn(2, 3, horizon, 236, generator=generator) * 0.2}
    rank = {"z0": torch.randn(2, 2, 236, generator=generator) * 0.2,
            "action": torch.rand(2, 2, 4, 5, generator=generator),
            "verdict": torch.tensor([[[1, 0, 0, -1], [0, 1, 1, 0]]] * 2)}
    return long, rank


def _loss(ensemble, long, rank, truncation, endpoint=12):
    pair = PredictionPair(1, Abstraction.CONTINUOUS)
    args = dict(long=long, rank=rank, horizon=long["targets"].shape[2], endpoint=endpoint, presence=106,
                center=(111, 112), temperature=0.1)
    if truncation is None:
        return dd.long_and_rank_loss(ensemble, pair, **args)
    return sd.truncated_long_and_rank_loss(ensemble, pair, truncation=truncation, **args)


def _grads(ensemble, value):
    grads = torch.autograd.grad(value.sum(), list(ensemble.params.values()), allow_unused=True)
    return [torch.zeros(()) if g is None else g for g in grads]


def test_truncation_keeps_every_loss_value_and_matches_v1_when_it_covers_the_recursion():
    generator = torch.Generator().manual_seed(1)
    ensemble = dd.Ensemble([_model(1), _model(2)], **KWARGS)
    long, rank = _inputs(generator, 8)
    reference = _loss(ensemble, long, rank, None)
    for truncation in (3, 5, 12):
        ours = _loss(ensemble, long, rank, truncation)
        assert torch.allclose(ours[0], reference[0]) and torch.allclose(ours[1], reference[1])
    full = _loss(ensemble, long, rank, 12)
    for a, b in zip(_grads(ensemble, full[0] + full[1]), _grads(ensemble, reference[0] + reference[1])):
        assert torch.allclose(a, b, atol=1e-6)


def test_truncation_cuts_the_gradient_path_at_the_segment_boundary():
    """With truncation k the ranking gradient equals that of a rollout whose state is detached
    at every multiple of k, and differs from the untruncated gradient."""
    generator = torch.Generator().manual_seed(2)
    ensemble = dd.Ensemble([_model(3), _model(4)], **KWARGS)
    _, rank = _inputs(generator, 1)
    pair = PredictionPair(1, Abstraction.CONTINUOUS)
    z = rank["z0"][:, :, None].expand(-1, -1, 4, -1).reshape(2, 8, -1)
    action = rank["action"].reshape(2, 8, -1)
    for step in range(12):
        if step and step % 4 == 0:
            z = z.detach()
        z = ensemble.transition(pair, z, action)
    manual = dd.pairwise_ranking(dd.tie_free_cost(z.reshape(2, 2, 4, -1), rank["z0"][:, :, None], 106, (111, 112)),
                                 rank["verdict"], 0.1)
    ours = sd.truncated_long_and_rank_loss(ensemble, pair, long=None, rank=rank, horizon=0, endpoint=12,
                                           presence=106, center=(111, 112), temperature=0.1, truncation=4)[1]
    full = dd.long_and_rank_loss(ensemble, pair, long=None, rank=rank, horizon=0, endpoint=12, presence=106,
                                 center=(111, 112), temperature=0.1)[1]
    ours_g, manual_g, full_g = _grads(ensemble, ours), _grads(ensemble, manual), _grads(ensemble, full)
    assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(ours_g, manual_g))
    assert any(not torch.allclose(a, b, atol=1e-6) for a, b in zip(ours_g, full_g))


def test_a_skipped_update_leaves_the_member_untouched_and_the_others_as_without_skipping():
    generator = torch.Generator().manual_seed(3)
    models = [_model(5), _model(6)]
    skipping = sd.SkippingEnsemble(copy.deepcopy(models), **KWARGS)
    reference = dd.Ensemble([copy.deepcopy(models[1])], **KWARGS)
    for step in range(3):  # a normal update first, so AdamW moments exist
        long, _ = _inputs(generator, 4)
        loss = _loss(skipping, long, None, 2, endpoint=0)[0]
        if step == 1:
            loss = torch.stack((torch.tensor(float("nan")), loss[1]))
        if step == 1:
            before = {k: v[0].detach().clone() for k, v in skipping.params.items()}
            moments = {k: {n: m[0].clone() for n, m in skipping.optimizer.state[v].items() if m.ndim}
                       for k, v in skipping.params.items()}
        report = skipping.update(loss)
        reference.update(_loss(reference, {k: v[1:] for k, v in long.items()}, None, 2, endpoint=0)[0])
        if step == 1:
            assert report["skipped"].tolist() == [True, False] and not bool(report["newly_retired"].any())
            for k, v in skipping.params.items():
                assert torch.equal(v[0].detach(), before[k])
                for n, m in moments[k].items():
                    assert torch.equal(skipping.optimizer.state[v][n][0], m)
    assert skipping.skipped.tolist() == [1, 0] and not bool(skipping.retired.any())
    worst = max(float((skipping.member_state(1)[n] - v).abs().max()) for n, v in reference.member_state(0).items())
    # batched and single-member gradients differ in the last float bits; Adam's normalisation
    # turns that into parameter differences of a few 1e-6 at lr 1e-3 (a real coupling would be ~lr)
    assert worst < 1e-5


def test_a_member_is_retired_after_patience_consecutive_skips_only():
    generator = torch.Generator().manual_seed(4)
    ensemble = sd.SkippingEnsemble([_model(7), _model(8)], **KWARGS)
    retired_at = None
    for step in range(16):
        long, _ = _inputs(generator, 2)
        loss = _loss(ensemble, long, None, 2, endpoint=0)[0]
        bad = step != 3  # one finite update resets the run of skips
        loss = torch.stack((torch.tensor(float("nan")) if bad else loss[0], loss[1]))
        report = ensemble.update(loss)
        if bool(report["newly_retired"][0]):
            retired_at = step
    assert retired_at == 3 + ensemble.patience
    assert ensemble.skipped.tolist() == [retired_at, 0] and ensemble.retired.tolist() == [True, False]
