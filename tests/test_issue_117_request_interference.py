import math

import torch

from world_model.training import request_interference as ri
from world_model.training.cnn_hybrid import PAIRS, CNNHybridPredictor


def test_per_request_budget_replays_the_joint_updates_of_the_arm_and_full_budget_matches_the_joint_count():
    joint = ri.schedule(range(9), "full")
    assert joint == ri.schedule(range(9), "request") == [(k % 9, k) for k in range(ri.STEPS)]
    per_delta = ri.schedule([3, 4, 5], "request")
    assert per_delta == [(p, g) for p, g in joint if p in (3, 4, 5)] and len(per_delta) == 3000
    specialist = ri.schedule([7], "request")
    assert [g for _, g in specialist] == list(range(7, ri.STEPS, 9))
    full = ri.schedule([2, 0], "full")
    assert len(full) == ri.STEPS and full[:4] == [(0, 0), (2, 1), (0, 2), (2, 3)]


def test_trunk_is_exactly_the_parameters_every_request_reaches():
    torch.manual_seed(0)
    model = CNNHybridPredictor()
    with torch.no_grad():  # FiLM modulation is zero-initialized; a trained model's is not
        for block in model.blocks:
            block.modulation.weight.normal_(0, 0.02)
    z, action = torch.randn(4, 236) * 0.3, torch.randn(4, 5)
    reached = []
    for pair in PAIRS:
        model.zero_grad(set_to_none=True)
        model.carrier(z, action, pair).square().sum().backward()
        reached.append({n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0})
    names = [n for n, _ in model.named_parameters()]
    assert set(ri.trunk_names(names)) == set.intersection(*reached) - {"conditioner.abstraction_embedding.weight"}


def test_gram_matches_concatenated_dot_products_and_cosines_handle_zero_gradients():
    generator = torch.Generator().manual_seed(1)
    a = {"x": torch.randn(2, 3, 4, generator=generator), "y": torch.randn(2, 5, generator=generator)}
    b = {"x": torch.randn(2, 3, 4, generator=generator), "y": None}
    c = {"x": -a["x"], "y": -a["y"]}
    zero = {"x": torch.zeros(2, 3, 4), "y": torch.zeros(2, 5)}
    gram = ri.request_gram([a, b, c, zero], ["x", "y"])
    flat = [torch.cat([g["x"].flatten(1), (torch.zeros(2, 5) if g["y"] is None else g["y"])], 1).double()
            for g in (a, b, c, zero)]
    expected = torch.einsum("rmn,smn->mrs", torch.stack(flat), torch.stack(flat))
    assert torch.allclose(gram, expected)
    cos = ri.cosines(gram)
    assert torch.allclose(cos[:, 0, 2], torch.full((2,), -1.0, dtype=torch.float64))
    assert torch.allclose(torch.diagonal(cos[:, :3, :3], dim1=-2, dim2=-1), torch.ones(2, 3, dtype=torch.float64))
    assert all(math.isnan(v) for v in cos[:, 3, 0].tolist())
