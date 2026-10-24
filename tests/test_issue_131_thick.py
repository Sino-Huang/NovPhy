"""#131 D1 THICK: CPU tests of the port's uncertain semantics and the NovPhy adapter plumbing.

Run: CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/test_issue_131_thick.py
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from world_model.baselines.thick import novphy as nv
from world_model.baselines.thick import port

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "world_model/baselines/thick/vendor"


class FixedNoise:
    """Returns a constant standard normal (scaled by the cell) and never a uniform."""

    def __init__(self, value=0.0):
        self.value = value
        self.calls = 0

    def normal(self, shape, device=None):
        self.calls += 1
        return torch.full(tuple(shape), float(self.value))

    def uniform(self, shape, device=None):
        raise AssertionError("no one-hot sample expected")


def _cell(bias, sd=0.1, always=True):
    torch.manual_seed(0)
    cell = port.GateL0RDCell(8, 4, sample_sd=sd, always_sample=always)
    with torch.no_grad():
        cell.g_layer.weight.zero_()
        cell.g_layer.bias.fill_(bias)
    return cell


def test_gatel0rd_closed_gate_keeps_the_context_exactly():
    cell = _cell(-3.0)
    x, c = torch.randn(5, 8), torch.rand(5, 4)
    nxt, gates = cell(x, c, sample=False, noise=FixedNoise(0.0))
    assert torch.equal(nxt, c)
    assert torch.equal(gates, torch.zeros_like(gates))


def test_gatel0rd_open_gate_counts_one_and_moves_the_context():
    cell = _cell(0.5)
    x, c = torch.randn(5, 8), torch.rand(5, 4)
    nxt, gates = cell(x, c, sample=False, noise=FixedNoise(0.0))
    assert torch.equal(gates, torch.ones_like(gates))
    update = math.tanh(0.5)
    cand = torch.tanh(cell.r_layer(torch.cat([x, c], -1)))
    assert torch.allclose(nxt, update * cand + (1 - update) * c, atol=1e-6)


def test_gatel0rd_gate_gradient_is_straight_through():
    cell = _cell(0.3)
    with torch.no_grad():
        cell.g_layer.weight.normal_(0, 0.1)
    x, c = torch.randn(3, 8), torch.rand(3, 4)
    _, gates = cell(x, c, sample=False, noise=FixedNoise(0.0))
    gates.sum().backward()
    grad_st = cell.g_layer.bias.grad.clone()
    cell.zero_grad()
    pre = cell.g_layer(torch.cat([x, c], -1))
    torch.relu(torch.tanh(pre)).sum().backward()
    assert torch.allclose(grad_st, cell.g_layer.bias.grad, atol=1e-6)


def test_gatel0rd_noise_applies_when_sampling_or_always_sample():
    x, c = torch.randn(2, 8), torch.rand(2, 4)
    # a positive noise draw of 10 (x sd 0.1 = +1) opens a gate whose pre-activation is -0.5
    for always, sample, opened in ((False, False, False), (False, True, True), (True, False, True)):
        cell = _cell(-0.5, sd=0.1, always=always)
        noise = FixedNoise(10.0)
        _, gates = cell(x, c, sample=sample, noise=noise)
        assert bool(gates.all()) is opened
        assert noise.calls == int(opened)


def _model(gate_bias=None):
    torch.manual_seed(1)
    model = nv.build_model()
    if gate_bias is not None:
        with torch.no_grad():
            model.rssm.ctxt_rnn.g_layer.weight.zero_()
            model.rssm.ctxt_rnn.g_layer.bias.fill_(gate_bias)
    return model


@pytest.fixture(scope="module")
def model():
    return _model()


def test_rollout_length_stride_and_endpoint(model):
    assert nv.POSITIONS == tuple(range(0, 226, 5)) and nv.ROLLOUT_STEPS == 45
    anchors, launch = torch.randn(3, 236), torch.tensor([[-0.2, 0.1, 1.0, 0.0, 1.0]]).repeat(3, 1)
    out = nv.rollout(model, anchors, launch)
    assert out["carriers"].shape == (3, 46, 236)
    assert out["gates"].shape == (3, 45, 16)
    assert torch.equal(out["endpoint"], out["carriers"][:, -1])
    assert out["coarse_endpoint"].shape == (3, 236)
    assert torch.isfinite(out["carriers"]).all()


def test_rollout_is_deterministic(model):
    anchors, launch = torch.randn(2, 236), torch.tensor([[-0.1, 0.05, 1.0, 0.0, 1.0]]).repeat(2, 1)
    a, b = nv.rollout(model, anchors, launch), nv.rollout(model, anchors, launch)
    assert torch.equal(a["endpoint"], b["endpoint"]) and a["commit_steps"] == b["commit_steps"]


def test_commit_points_follow_the_gates():
    anchors, launch = torch.randn(2, 236), torch.tensor([[-0.2, 0.1, 1.0, 0.0, 1.0]]).repeat(2, 1)
    closed = nv.rollout(_model(gate_bias=-5.0), anchors, launch)
    assert closed["commit_steps"] == [[], []]
    assert closed["update_counts"].tolist() == [0, 0]
    opened = nv.rollout(_model(gate_bias=5.0), anchors, launch)
    assert opened["commit_steps"] == [list(range(1, 46))] * 2
    assert opened["update_counts"].tolist() == [45, 45]
    # mixed: commit steps are exactly the steps with any open gate
    mixed = nv.rollout(_model(), anchors, launch)
    for row, gates in zip(mixed["commit_steps"], mixed["gates"]):
        assert row == [t + 1 for t in range(45) if float(gates[t].sum()) > 0]


def test_closed_gates_freeze_the_context_over_the_rollout():
    m = _model(gate_bias=-5.0)
    anchors, launch = torch.randn(2, 236), torch.tensor([[-0.2, 0.1, 1.0, 0.0, 1.0]]).repeat(2, 1)
    contexts = []
    original = m.rssm.img_step

    def spy(prev_state, prev_action, *args, **kwargs):
        out = original(prev_state, prev_action, *args, **kwargs)
        contexts.append(out["context"].clone())
        return out

    m.rssm.img_step = spy
    nv.rollout(m, anchors, launch)
    assert len(contexts) == 46  # the t0 observe step's prior, then the 45 rollout steps
    assert all(torch.equal(c, contexts[0]) for c in contexts)


def test_launch_enters_only_at_t0(model):
    seen = []
    original = model.rssm.img_step

    def spy(prev_state, prev_action, *args, **kwargs):
        seen.append(prev_action.clone())
        return original(prev_state, prev_action, *args, **kwargs)

    model.rssm.img_step = spy
    try:
        launch = torch.tensor([[-0.2, 0.1, 1.0, 0.0, 1.0], [-0.3, 0.15, 1.0, 0.0, 1.0]])
        nv.rollout(model, torch.randn(2, 236), launch)
    finally:
        model.rssm.img_step = original
    # call 0 is the t0 observe step (reset: zero action), then the 45 rollout steps
    assert len(seen) == 46
    assert torch.equal(seen[0], torch.zeros_like(launch))
    assert torch.equal(seen[1], launch)
    assert all(torch.equal(a, torch.zeros_like(a)) for a in seen[2:])


def test_launch_changes_the_endpoint(model):
    anchors = torch.randn(1, 236).repeat(2, 1)
    launch = torch.tensor([[-0.05, 0.02, 1.0, 0.0, 1.0], [-0.33, 0.16, 1.0, 0.0, 1.0]])
    out = nv.rollout(model, anchors, launch)
    assert not torch.allclose(out["endpoint"][0], out["endpoint"][1])


def test_training_episode_layout():
    launch = torch.tensor([[-0.2, 0.1, 1.0, 0.0, 1.0]])
    batch = nv.episode_batch(torch.randn(1, 46, 236), launch)
    assert torch.equal(batch["action"][0, 1], launch[0])
    assert float(batch["action"][0, 0].abs().sum()) == 0 and float(batch["action"][0, 2:].abs().sum()) == 0
    assert batch["is_first"][0].tolist() == [True] + [False] * 45
    assert batch["is_last"][0].tolist() == [False] * 45 + [True]
    assert not batch["is_terminal"].any() and float(batch["reward"].abs().sum()) == 0
    with pytest.raises(ValueError):
        nv.episode_batch(torch.randn(1, 45, 236), launch)
    with pytest.raises(ValueError):
        nv.episode_actions(torch.tensor([[1.5, 0.0, 1.0, 0.0, 1.0]]))


def test_episode_positions_repeat_the_end():
    assert nv.episode_positions(285) == list(range(0, 226, 5))
    positions = nv.episode_positions(138)
    assert len(positions) == 46 and positions[:28] == list(range(0, 136, 5)) and set(positions[28:]) == {138}


def test_training_step_runs_and_moves_both_levels():
    m = _model()
    model_opt, thick_opt = m.make_optimizers()
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    batch = nv.episode_batch(torch.randn(2, 46, 236), torch.tensor([[-0.2, 0.1, 1.0, 0.0, 1.0]]).repeat(2, 1))
    out = m.train_step(batch, model_opt, thick_opt, noise=port.TorchNoise(torch.Generator().manual_seed(0)))
    assert math.isfinite(float(out["model_loss"])) and math.isfinite(float(out["thick_loss"]))
    moved = {n.split(".")[0] for n, p in m.named_parameters() if not torch.equal(p, before[n])}
    assert {"encoder", "rssm", "heads", "ctxt_heads", "thick_net"} <= moved
    assert model_opt.iterations == 1 and thick_opt.iterations == 1


def test_gen_thick_data_targets_the_state_before_each_context_change():
    s, b = 6, 1
    states = torch.arange(s, dtype=torch.float32).reshape(s, 1, 1).repeat(1, b, 2)
    gates = torch.zeros(s, b, 3)
    gates[2, 0, 1] = 1.0  # context changes at t = 2 and t = 4
    gates[4, 0, 0] = 1.0
    inps, targets, masks, _ = port.gen_thick_data(
        inp_states=states, tar_states=states, tar_actions=torch.zeros(s, b, 1) + torch.arange(s).reshape(s, 1, 1),
        tar_latent=states, tar_rewards=torch.zeros(s, b, 1), tar_stoch=states, gs=gates, discount=0.99)
    assert inps[:, 0].tolist() == [0, 1, 2, 3, 4]
    assert targets["thick_tar_states"][:, 0].tolist() == [1, 1, 3, 3, 99]
    assert targets["thick_time"][:, 0].tolist() == [1, 0, 1, 0, 99 - 4]
    assert targets["thick_act"][:, 0].tolist() == [2, 2, 4, 4, 0]  # the action at the change step
    assert masks["time"][:, 0].tolist() == [1, 1, 1, 1, 0]


def test_trunc_normal_log_prob_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    loc = torch.tensor([0.3, -0.9, 0.0, 0.99])
    scale = torch.tensor([0.2, 0.5, 1.5, 0.05])
    x = torch.tensor([0.1, -1.0, 1.0, 0.98])
    mine = port.IndependentTruncNormal(loc, scale, ndims=0).log_prob(x)
    a, b = (-1 - loc) / scale, (1 - loc) / scale
    ref = stats.truncnorm.logpdf(x.numpy(), a.numpy(), b.numpy(), loc=loc.numpy(), scale=scale.numpy())
    assert np.allclose(mine.numpy(), ref, atol=1e-4)
    assert torch.isinf(port.IndependentTruncNormal(loc, scale, ndims=0).log_prob(torch.full((4,), 1.01))).all()


def test_tf_adam_places_epsilon_outside_the_bias_correction():
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
    opt = port.TFOptimizer([("p", p)], lr=0.1, eps=1e-1, clip=100, wd=0.0)
    g = torch.tensor([1e-3, -0.5, 0.0])
    opt.apply([g])
    m, v = 0.1 * g, 0.001 * g * g
    alpha = 0.1 * math.sqrt(1 - 0.999) / (1 - 0.9)
    expected = torch.tensor([1.0, -2.0, 0.5]) - alpha * m / (torch.sqrt(v) + 1e-1)
    assert torch.allclose(p.detach(), expected, atol=1e-7)


def test_global_norm_clip_and_weight_decay_order():
    p = torch.nn.Parameter(torch.tensor([2.0, 0.0]))
    opt = port.TFOptimizer([("p", p)], lr=0.0, eps=1e-5, clip=1.0, wd=0.5)
    norm = opt.apply([torch.tensor([3.0, 4.0])])
    assert float(norm) == pytest.approx(5.0)  # the pre-clip norm is reported
    assert torch.allclose(p.detach(), torch.tensor([1.0, 0.0]))  # wd applied, lr 0 moves nothing else
    assert torch.allclose(opt.m[0], 0.1 * torch.tensor([0.6, 0.8]))  # clipped to norm 1 before Adam


def test_config_reads_yaml12_floats_and_native_values():
    config = nv.build_config("sparsity-10")
    assert config.model_opt.lr == pytest.approx(1e-4) and isinstance(config.model_opt.lr, float)
    assert config.loss_scales.ctxt_sparsity == 10.0
    assert config.rssm.deter == 256 and config.context.context == 16 and config.thick_hl_act_dim == 5
    assert config.encoder.mlp_keys == "carrier" and config.encoder.cnn_keys == "$^"
    assert tuple(config.thick_segment_crit) == ("gates",)
    assert nv.parameter_count(nv.build_model(config)) == 21_852_337


def test_vendor_hashes_match():
    vendor = json.loads((VENDOR / "VENDOR.json").read_text())
    assert vendor["commit"] == port.UPSTREAM_COMMIT
    edited = {rel for rel, entry in vendor["files"].items() if entry["edits"]}
    assert edited == {"thick/common/nets.py"}  # the one listed edit (Decision P-route)
    for rel, entry in vendor["files"].items():
        digest = "sha256:" + hashlib.sha256((VENDOR / rel).read_bytes()).hexdigest()
        assert digest == entry["vendored_sha256"], rel
        if rel not in edited:
            assert entry["vendored_sha256"] == entry["upstream_sha256"], rel
    nets = (VENDOR / "thick/common/nets.py").read_text()
    assert "self._rnn_type = rnn_type" not in nets and nets.count("\n") > 800
