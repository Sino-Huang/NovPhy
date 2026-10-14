"""Issue-117 confirmatory ranking: the vectorized wrong-anchor statistics, the delta rule and the guard
propagation (data-free)."""
import numpy as np
import pytest
import torch

from scripts import run_decision_dynamics_controls as c2
from scripts import run_request_interference_ranking as rk

COST = rk.PRIMARY_COST


def rows(values, excluded=()):
    return [{"ordinal": o, "excluded": o in excluded, COST: v} for o, v in values.items()]


def loop_summary(entry, right, wrong_rows):
    wrong_auc = [a for a in (c2.auc(entry, w) for w in wrong_rows) if a is not None]
    rhos = [r for r in (c2.spearman_rows(right, w) for w in wrong_rows) if r is not None]
    return {"right": c2.auc(entry, right), "wrong": float(np.mean(wrong_auc)) if wrong_auc else None,
            "stability": float(np.mean(rhos)) if rhos else None}


@pytest.mark.parametrize("seed", range(6))
def test_vectorized_anchor_summary_equals_the_gate_b_loop(seed):
    generator = np.random.default_rng(seed)
    ordinals = list(range(16))
    # coarse values force ties within and across candidates; one anchor is constant (undefined Spearman)
    wrong = {f"a{k:03d}": {"rows": rows({o: float(generator.integers(0, 4)) / 4 for o in ordinals})}
             for k in range(40)}
    wrong["a999"] = {"rows": rows({o: 0.5 for o in ordinals})}
    subset = sorted(generator.choice(ordinals, size=int(generator.integers(4, 16)), replace=False).tolist())
    verdicts = {o: bool(i % 2) for i, o in enumerate(subset)}
    entry = {"verdicts": verdicts}
    right = rows({o: float(generator.normal()) for o in subset})
    matrix = rk.wrong_matrix(wrong)
    ours = rk.anchor_summary(entry, right, matrix)
    reference = loop_summary(entry, right, matrix["rows"])
    assert ours["right"] == reference["right"]
    assert ours["wrong"] == reference["wrong"]
    assert abs(ours["stability"] - reference["stability"]) <= rk.STABILITY_TOLERANCE


def test_excluded_or_nonfinite_rows_fall_back_to_the_loop():
    wrong = {"a": {"rows": rows({0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}, excluded=(2,))},
             "b": {"rows": rows({0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1})}}
    matrix = rk.wrong_matrix(wrong)
    assert matrix["costs"] is None
    entry = {"verdicts": {0: True, 1: False, 2: True, 3: False}}
    right = rows({0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0})
    assert rk.anchor_summary(entry, right, matrix) == loop_summary(entry, right, matrix["rows"])


def block(low, high, clusters=100):
    return {"mean": (low + high) / 2, "interval": [low, high], "clusters": clusters}


def test_noninferiority_delta_rule():
    d = rk.DELTA
    assert rk.noninferiority(block(-d + 1e-4, 0.05)) == "supported"
    assert rk.noninferiority(block(-0.10, -d - 1e-4)) == "not_supported_by_this_experiment"
    assert rk.noninferiority(block(-0.05, 0.01)) == rk.STOP_TOKEN
    assert rk.noninferiority(block(0.0, 0.1, clusters=2)) == rk.STOP_TOKEN
    assert rk.noninferiority(None) == rk.STOP_TOKEN


def test_a_failed_guard_turns_every_token_insufficient():
    result = {"disposition": {"token": "supported", "contrast_tokens": {"request_mean": "supported"}},
              "inventories": {"grid": {"x": [{"token": "not_supported_by_this_experiment", "estimate": 1.0}]}}}
    out = rk.invalidate(result)
    assert out["disposition"]["token"] == rk.STOP_TOKEN
    assert out["disposition"]["contrast_tokens"] == {"request_mean": rk.STOP_TOKEN}
    assert out["inventories"]["grid"]["x"][0] == {"token": rk.STOP_TOKEN, "estimate": 1.0}


def test_usability_types_each_failure():
    requests = ["F-1-continuous"] * 1000
    flat = torch.full((1000,), 0.1, dtype=torch.float64)
    norms = torch.ones(1000, dtype=torch.float64)
    assert rk.usability(True, 0, flat, norms, requests, 1000)["usable"]
    assert rk.usability(False, 0, flat, norms, requests, 1000)["typed_failure"] == "retired"
    assert rk.usability(True, rk.SKIP_CAP + 1, flat, norms, requests, 1000)["typed_failure"] == "skip_cap"
    risen = flat.clone()
    risen[-3:] = 1.0
    assert rk.usability(True, 0, risen, norms, requests, 1000)["typed_failure"] == "final_checkpoint_excursion"
