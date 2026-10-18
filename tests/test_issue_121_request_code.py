"""Issue #121: the request-code arms (fused control, additive rank-2 code, factored-FiLM composition)."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from world_model.model import PredictionPair
from world_model.model.predictor import PairConditioner
from world_model.training import request_code as rc
from world_model.training.cnn_hybrid import CONFIG, DIM, PAIRS, SLOTS, CNNHybridPredictor, linear_macs


def carriers(count, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.rand(count, DIM, generator=g) * 0.5
    z[:, 2::13] = (torch.rand(count, SLOTS, generator=g) > 0.3).float()
    return z


def actions(count, seed=1):
    return torch.rand(count, 5, generator=torch.Generator().manual_seed(seed))


def model_of(arm, seed=3):
    torch.manual_seed(seed)
    return rc.new_predictor(arm).eval()


def randomise_film(model, arm, seed=5):
    """Non-zero FiLM generators (they start at zero), so that the request path matters."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for block in model.blocks:
            layers = (block.modulation_delta, block.modulation_alpha) if arm == "factored" else (block.modulation,)
            for layer in layers:
                layer.weight.copy_(torch.randn(layer.weight.shape, generator=g) * 0.1)
                layer.bias.copy_(torch.randn(layer.bias.shape, generator=g) * 0.1)
    return model


# --- fused equals the original ----------------------------------------------------------------------------

def test_fused_arm_is_the_original_predictor():
    fused, original = model_of("fused"), None
    torch.manual_seed(3)
    original = CNNHybridPredictor().eval()
    assert type(fused) is CNNHybridPredictor
    assert all(torch.equal(v, original.state_dict()[k]) for k, v in fused.state_dict().items())
    z, a = carriers(6), actions(6)
    with torch.no_grad():
        for pair in PAIRS:
            assert torch.equal(fused.carrier(z, a, pair), original.carrier(z, a, pair))


def test_delta_features_equal_the_pair_conditioner():
    conditioner = PairConditioner(CONFIG)
    for delta in (1, 5, 15, 37):
        assert torch.equal(rc.delta_features(CONFIG, delta, "cpu"), conditioner.delta_features(delta, "cpu"))


def test_arms_share_the_fused_trunk_initialisation():
    reference = model_of("fused", seed=11).state_dict()
    fused_request = rc.request_path_names("fused")
    trunk = sorted(k for k in reference if not k.startswith(fused_request))
    for arm in ("additive", "factored"):
        state = model_of(arm, seed=11).state_dict()
        shared = sorted(k for k in state if not k.startswith(rc.request_path_names(arm)))
        assert shared == trunk
        assert all(torch.equal(state[k], reference[k]) for k in shared)


def test_parameter_difference_is_confined_to_the_request_path():
    splits = {arm: rc.parameter_split(model_of(arm), arm) for arm in rc.ARMS}
    assert len({s["trunk_heads_adapters"] for s in splits.values()}) == 1
    assert splits["fused"]["total"] == sum(p.numel() for p in CNNHybridPredictor().parameters())
    code, film = 128, 2 * CONFIG.hidden_dim
    assert splits["additive"]["request_path"] == 3 * code + (16 * code + code) + code * code + 3 * (code * film + film)
    assert splits["factored"]["request_path"] == 3 * code + 3 * ((16 * film + film) + (code * film + film))


# --- additive: rank-2 structure ---------------------------------------------------------------------------

def test_additive_codes_have_rank_two_structure():
    model = model_of("additive")
    with torch.no_grad():
        codes = {(p.delta, p.abstraction): model.conditioner.code(p, 1, "cpu")[0] for p in PAIRS}
    deltas, alphas = rc.DELTAS, rc.ALPHAS
    for a1 in alphas:
        for a2 in alphas:
            differences = [codes[(d, a1)] - codes[(d, a2)] for d in deltas]
            assert all(torch.allclose(x, differences[0], atol=1e-6) for x in differences)
    centred = torch.stack([torch.stack([codes[(d, a)] for a in alphas]) for d in deltas]).double()
    centred = centred - centred.mean((0, 1), keepdim=True)
    stacked = centred.reshape(-1, centred.shape[-1])
    # u(Delta) + v(alpha) over a 3 x 3 grid spans at most (3 - 1) + (3 - 1) = 4 centred directions
    assert torch.linalg.matrix_rank(stacked, atol=1e-5) <= 4


def test_additive_film_has_no_interaction_and_fused_does():
    additive = randomise_film(model_of("additive"), "additive")
    for block in rc.coupling(additive, "additive")["blocks"]:
        for part in ("gamma", "beta"):
            assert block[part]["interaction_share"] < 1e-12
    fused = randomise_film(model_of("fused"), "fused")
    shares = [b["gamma"]["interaction_share"] for b in rc.coupling(fused, "fused")["blocks"]]
    assert min(shares) > 1e-4


def test_two_way_decomposition():
    u = torch.randn(3, 1, 7, dtype=torch.float64)
    v = torch.randn(1, 3, 7, dtype=torch.float64)
    additive = rc.two_way(u + v)
    assert additive["ss_interaction"] < 1e-20
    assert abs(additive["ss_delta"] + additive["ss_alpha"] - additive["ss_total"]) < 1e-9
    interaction = torch.randn(3, 3, 7, dtype=torch.float64)
    full = rc.two_way(u + v + interaction)
    assert abs(full["ss_delta"] + full["ss_alpha"] + full["ss_interaction"] - full["ss_total"]) < 1e-9
    assert 0 < full["interaction_share"] < 1


# --- factored-FiLM composition ----------------------------------------------------------------------------

def test_factored_block_composes_per_axis_modulation():
    model = randomise_film(model_of("factored"), "factored")
    block = model.blocks[1]
    pair = PredictionPair(5, rc.ALPHAS[2])
    hidden = torch.randn(4, CONFIG.hidden_dim, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        code = model.conditioner.code(pair, 4, "cpu")
        phi, alpha = code
        assert torch.equal(phi[0], rc.delta_features(CONFIG, 5, "cpu"))
        gd, bd = block.modulation_delta(phi).chunk(2, -1)
        ga, ba = block.modulation_alpha(alpha).chunk(2, -1)
        expected = hidden + block.outer(F.silu(block.inner(block.norm(hidden) * (1 + gd) * (1 + ga) + bd + ba)))
        assert torch.allclose(block(hidden, code), expected, atol=1e-6)
        # with the alpha generator at zero the block is a FiLM block driven by Delta alone
        block.modulation_alpha.weight.zero_()
        block.modulation_alpha.bias.zero_()
        expected = hidden + block.outer(F.silu(block.inner(block.norm(hidden) * (1 + gd) + bd)))
        assert torch.allclose(block(hidden, code), expected, atol=1e-6)


def test_factored_starts_as_identity_modulation_and_has_scale_interaction():
    fresh = model_of("factored")
    table = rc.film_table(fresh, "factored")
    assert all(float(g.abs().max()) == 0 for g in table["gamma"])
    model = randomise_film(model_of("factored"), "factored")
    blocks = rc.coupling(model, "factored")["blocks"]
    assert all(b["beta"]["interaction_share"] < 1e-12 for b in blocks)
    assert all(b["gamma"]["interaction_share"] > 1e-6 for b in blocks)


def test_arms_run_every_pair_and_count_macs():
    z, a = carriers(5), actions(5)
    for arm in rc.ARMS:
        model = randomise_film(model_of(arm), arm)
        with torch.no_grad():
            for pair in PAIRS:
                out = model.carrier(z, a, pair)
                assert out.shape == z.shape and bool(torch.isfinite(out).all())
        continuous = linear_macs(model, PAIRS[0])
        fused = linear_macs(CNNHybridPredictor(), PAIRS[0])
        code, film, width = 128, 2 * CONFIG.hidden_dim, CONFIG.hidden_dim
        if arm == "fused":
            assert continuous == fused
        elif arm == "additive":
            assert continuous == fused - (144 * 256 + 256 * code) + (16 * code + code * code)
        else:
            assert continuous == fused - (144 * 256 + 256 * code) + 3 * 16 * film
        assert width == 384


def test_d2_publication_states_the_demonstrated_claim_beside_the_frozen_token():
    import json

    import pytest

    from scripts import run_request_code_factorisation as runner
    if not (runner.OUTPUT / "compute.json").is_file():
        pytest.skip("issue-121 publication not present")
    compute = json.loads((runner.OUTPUT / "compute.json").read_text())
    d2 = compute["dispositions"]["interaction_decision_value"]
    assert d2["claim"] == "axis interaction in the request code carries measurable decision value"  # frozen
    assert d2["token"] == "supported"
    assert {k: d2[k] for k in runner.D2_READING} == runner.D2_READING
    assert "parameterization" in d2["demonstrated_claim"] and "unidentified" in d2["qualification"]
    findings = (runner.OUTPUT / "findings.md").read_text()
    assert runner.D2_READING["demonstrated_claim"] in findings
