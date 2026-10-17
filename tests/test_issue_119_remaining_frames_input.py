import json
import math
from pathlib import Path

import pytest
import torch

from world_model.training import remaining_input as ri
from world_model.training import request_policy as rp
from world_model.training.cnn_hybrid import DIM, PAIRS, CNNHybridPredictor

OUTPUT = Path(__file__).resolve().parents[1] / ".local-artifacts/issue-119-remaining-frames-v1"


def predictor(seed=0):
    torch.manual_seed(seed)
    return CNNHybridPredictor().eval()


def windows(seed, lengths=(20, 13, 7), frames=21):
    generator = torch.Generator().manual_seed(seed)
    return [{"z": torch.randn(frames, DIM, generator=generator)*0.1, "action": torch.rand(5, generator=generator),
             "length": torch.tensor(length)} for length in lengths]


# ---------------------------------------------------------------------------
# feature variants
# ---------------------------------------------------------------------------

def test_variant_features():
    kappa = torch.tensor([1, 30, 60, 61, 165, 225])
    like = torch.zeros(1)

    def feature(variant):
        return rp.REMAINING_FEATURES[ri.VARIANTS[variant]["remaining_feature"]][1](kappa, like).flatten()

    assert torch.allclose(feature("a"), torch.tensor([1/60, 0.5, 1, 1, 1, 1]))
    assert torch.equal(feature("b"), kappa.clamp(max=225).float()/225)
    assert torch.allclose(feature("c"), torch.log1p(kappa.float())/math.log1p(225))
    assert feature("b")[-1] == 1 and feature("c")[-1] == 1
    # only the unsaturated variants separate kappa = 61 from 225
    assert feature("a")[3] == feature("a")[5] and feature("b")[3] < feature("b")[5] and feature("c")[3] < feature("c")[5]
    assert ri.VARIANTS["d"]["elapsed"] and ri.VARIANTS["d"]["remaining_feature"] == "clamp60"


def test_variant_policies_and_init():
    for variant in ("a", "b", "c"):
        mine = ri.new_policy(variant, PAIRS, 20260908, device="cpu")
        reference = rp.new_policy(PAIRS, 20260908, remaining_feature=ri.VARIANTS[variant]["remaining_feature"],
                                  device="cpu")
        assert type(mine) is rp.RequestPolicy and ri.variant_of(mine) == variant
        assert all(torch.equal(x, y) for x, y in zip(mine.state_dict().values(), reference.state_dict().values()))
        assert rp.policy_macs(mine) == (DIM+6)*128+128*9
    d1 = ri.new_policy("d", PAIRS, 20260908, device="cpu")
    d2 = ri.new_policy("d", PAIRS, 20260908, device="cpu")
    assert ri.uses_elapsed(d1) and ri.variant_of(d1) == "d"
    assert d1.layers[0].in_features == DIM+7 and rp.policy_macs(d1) == (DIM+7)*128+128*9
    assert all(torch.equal(x, y) for x, y in zip(d1.state_dict().values(), d2.state_dict().values()))
    loaded = ri.build_policy("d", PAIRS)
    loaded.load_state_dict(d1.state_dict(), strict=True)


def test_elapsed_policy_forward():
    torch.manual_seed(3)
    policy = ri.ElapsedRequestPolicy(PAIRS)
    z, action = torch.randn(5, DIM), torch.rand(5, 5)
    remaining = torch.tensor([1, 4, 14, 60, 225])
    logits = policy(z, action, remaining)
    assert torch.equal(logits, policy(z, action, remaining, 225-remaining))
    assert torch.equal(logits, ri.call(policy, z, action, remaining, None))
    deltas = torch.tensor([p.delta for p in PAIRS])
    assert torch.equal(torch.isinf(logits), deltas[None] > remaining[:, None])
    # the elapsed channel changes the logits (unlike the saturated kappa input past 60)
    assert not torch.equal(policy(z, action, remaining, torch.zeros_like(remaining)), logits)
    manual = policy.layers(torch.cat((z, action, remaining.clamp(max=60)[:, None].float()/60,
                                      (225-remaining).clamp(max=225)[:, None].float()/225), -1))
    assert torch.equal(logits[torch.isfinite(logits)], manual[torch.isfinite(logits)])
    with pytest.raises(ValueError):
        policy(z[:, :10], action, remaining)


# ---------------------------------------------------------------------------
# label rows, DAgger and distillation
# ---------------------------------------------------------------------------

def test_window_rows_abc_equal_request_policy_plus_elapsed():
    model = predictor()
    policy = ri.new_policy("b", PAIRS, 1, device="cpu")
    for aggregated in (False, True):
        mine = ri.window_rows(model, policy, windows(2), aggregated=aggregated, device="cpu")
        reference = rp.window_rows(model, policy, windows(2), aggregated=aggregated, device="cpu")
        assert all(torch.equal(mine[k], reference[k]) for k in reference)
        assert torch.equal(mine["elapsed"], torch.cat([torch.arange(n) for n in (20, 13, 7)]))


def test_elapsed_dagger_feeds_window_frames():
    model = predictor()
    policy = ri.new_policy("d", PAIRS, 1, device="cpu")
    seen = []
    hook = policy.register_forward_hook(lambda module, inputs, output: seen.append((int(inputs[2]), int(inputs[3]))))
    rows = ri.window_rows(model, policy, windows(4, lengths=(20,)), aggregated=True, device="cpu")
    hook.remove()
    assert seen and all(remaining + elapsed == 20 for remaining, elapsed in seen)
    assert seen[0] == (20, 0)
    assert torch.equal(rows["elapsed"] + rows["remaining"], torch.full((20,), 20))
    observed = ri.window_rows(model, policy, windows(4, lengths=(20,)), aggregated=False, device="cpu")
    z = windows(4, lengths=(20,))[0]
    labels = rp.prediction_error_labels(model, policy, z["z"][None], z["action"][None], z["length"].reshape(1))
    assert torch.equal(observed["labels"], labels[0, :20])


def rows_for(seed, units=3, size=40, elapsed=True):
    generator = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(units):
        remaining = torch.randint(1, 226, (size,), generator=generator)
        row = {"z": torch.randn(size, DIM, generator=generator), "action": torch.rand(size, 5, generator=generator),
               "remaining": remaining, "labels": torch.zeros(size, dtype=torch.long)}
        if elapsed:
            row["elapsed"] = 225-remaining
        out.append(row)
    return out


def test_distil_abc_is_request_policy_distil():
    rows = rows_for(7)
    first = ri.new_policy("c", PAIRS, 9, device="cpu")
    second = ri.new_policy("c", PAIRS, 9, device="cpu")
    mine = ri.distil_round(first, ri.aggregate_rounds([rows]), 0, seed=9, steps=6, batch_size=8, device="cpu")
    plain = [{k: u[k] for k in ("z", "action", "remaining", "labels")} for u in rows]
    reference = rp.distil_round(second, rp.aggregate_rounds([plain]), 0, seed=9, steps=6, batch_size=8, device="cpu")
    assert mine == reference
    assert all(torch.equal(x, y) for x, y in zip(first.state_dict().values(), second.state_dict().values()))


def test_distil_d_reads_elapsed():
    rows = rows_for(8)
    first = ri.new_policy("d", PAIRS, 9, device="cpu")
    second = ri.new_policy("d", PAIRS, 9, device="cpu")
    shifted = [{**u, "elapsed": torch.zeros_like(u["elapsed"])} for u in rows]
    a = ri.distil_round(first, ri.aggregate_rounds([rows]), 0, seed=9, steps=4, batch_size=8, device="cpu")
    b = ri.distil_round(second, ri.aggregate_rounds([shifted]), 0, seed=9, steps=4, batch_size=8, device="cpu")
    assert a != b
    assert len(ri.aggregate_rounds([rows, rows])[0]["elapsed"]) == 80


def test_decision_elapsed_is_position():
    rows = {"position": torch.tensor([0, 15, 30]), "remaining": torch.tensor([225, 210, 195])}
    assert torch.equal(ri.decision_elapsed(rows)["elapsed"], torch.tensor([0, 15, 30]))


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def test_kappa_bins_and_mix():
    assert [ri.kappa_bin(k) for k in (225, 166, 165, 106, 105, 61, 60, 31, 30, 1)] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    with pytest.raises(ValueError):
        ri.kappa_bin(0)
    fifteen_macro = PAIRS.index(next(p for p in PAIRS if p.delta == 15 and p.abstraction.value == "macro"))
    one_cont = PAIRS.index(next(p for p in PAIRS if p.delta == 1 and p.abstraction.value == "continuous"))
    counts = ri.kappa_mix([[fifteen_macro]*15, [one_cont]*225], PAIRS)
    # 15 x Delta=15: kappa 225, 210, 195, 180 (bin 0), 165, 150, 135, 120 (bin 1), 105, 90, 75 (bin 2), 60, 45 (bin 3), 30, 15 (bin 4)
    assert [row[fifteen_macro] for row in counts] == [4, 4, 3, 2, 2]
    assert [row[one_cont] for row in counts] == [60, 60, 45, 30, 30]


def test_label_accuracy():
    torch.manual_seed(0)
    policy = ri.new_policy("d", PAIRS, 3, device="cpu")
    rows = rows_for(11, units=1, size=50)[0]
    rows["labels"] = ri.call(policy, rows["z"], rows["action"], rows["remaining"], rows["elapsed"]).argmax(-1)
    full = ri.label_accuracy(policy, rows, device="cpu", chunk=7)
    assert full["accuracy"] == 1.0 and full["rows"] == 50
    assert sum(b["rows"] for b in full["by_kappa"]) == 50
    rows["labels"] = (rows["labels"] + 1) % len(PAIRS)
    assert ri.label_accuracy(policy, rows, device="cpu")["accuracy"] < 1.0
    plain = ri.new_policy("a", PAIRS, 3, device="cpu")
    without = {k: v for k, v in rows.items() if k != "elapsed"}
    assert ri.label_accuracy(plain, without, device="cpu")["rows"] == 50


# ---------------------------------------------------------------------------
# replication guard and disposition (published artefacts)
# ---------------------------------------------------------------------------

def published():
    path = OUTPUT / "compute.json"
    if not path.is_file():
        pytest.skip("issue-119 publication not present")
    return json.loads(path.read_text())


def test_replication_guard_passes_bit_for_bit():
    compute = published()
    guard = compute["guards"]["G5_replication_101_J"]
    assert guard["pass"] and guard["labels"] and guard["checkpoints"] and guard["policy"] and guard["records"]
    assert guard["blocks"] == len(compute["training"]["J-a-L60"]) * (311 + 76 + 78 + 47)  # Gate-B mixed cells


def test_variant_a_policies_equal_101_j():
    from scripts import run_decision_aware_policy as dap
    from scripts import run_remaining_frames_input as runner
    published()
    for seed in runner.SEEDS:
        mine = torch.load(runner.policy_path(OUTPUT, ("pe", "a", 60, seed)), map_location="cpu", weights_only=True)["model"]
        theirs = torch.load(dap.policy_path(dap.OUTPUT, "pe", 1e-4, seed), map_location="cpu", weights_only=True)["model"]
        assert mine.keys() == theirs.keys() and all(torch.equal(mine[k], theirs[k]) for k in mine)


def test_guard_failure_invalidates_every_token():
    from scripts import run_remaining_frames_input as runner
    tables = {"grid": {"contrasts": {"x": {"token": "supported", "estimate": {"mean": 0.1}}}},
              "wrong": [{"token": "not_supported_by_this_experiment"}]}
    out = runner.invalidate(tables)
    assert out["grid"]["contrasts"]["x"]["token"] == runner.STOP_TOKEN
    assert out["wrong"][0]["token"] == runner.STOP_TOKEN
    assert out["grid"]["contrasts"]["x"]["estimate"] == {"mean": 0.1}


def test_primary_token_rule():
    from scripts import run_decision_dynamics_controls as c2
    assert c2.token({"mean": 0.03, "interval": [0.001, 0.05]}) == "supported"
    assert c2.token({"mean": 0.01, "interval": [-0.01, 0.019]}) == "not_supported_by_this_experiment"
    assert c2.token({"mean": 0.03, "interval": [-0.001, 0.06]}) == "readiness_or_precision_insufficient"
