"""Issue-107 statistics helpers (CPU only, synthetic inputs)."""
import math

import pytest
import torch

from scripts import run_granularity_under_novelty as g


def block(mean, low, high):
    return {"mean": mean, "interval": [low, high], "clusters": 10, "units": 30, "label": "DESCRIPTIVE"}


def test_delta_and_non_inferiority_tokens():
    assert g.token(block(0.03, 0.001, 0.05)) == "supported"
    assert g.token(block(0.019, 0.001, 0.05)) == g.STOP_TOKEN
    assert g.token(block(0.0, -0.01, 0.0199)) == "not_supported_by_this_experiment"
    assert g.token(None) == g.STOP_TOKEN
    assert g.non_inferiority_token(block(-0.01, -0.0199, 0.0)) == "supported"
    assert g.non_inferiority_token(block(-0.05, -0.08, -0.021)) == "not_supported_by_this_experiment"
    assert g.non_inferiority_token(block(-0.02, -0.04, 0.0)) == g.STOP_TOKEN


def test_view_keys_and_marginals():
    assert g.view_key("zs", "F-15-macro") == "zs:F-15-macro"
    assert g.view_key("fs", "F-1-micro") == "predictor:F-1-micro"
    assert g.view_key("zs", "N-dec") == "zs:J_dec"
    assert g.view_key("fs", "N-pe") == "predictor:J"
    assert g.view_key("zs", "P-dec") == "policy:J_dec"
    assert g.view_key("fs", "B-pe") == "both:J"
    vector = [0.1, 0.0, 0.2, 0.0, 0.3, 0.0, 0.1, 0.1, 0.2]
    deltas, alphas = g.marginals(vector)
    assert deltas == pytest.approx([0.3, 0.3, 0.4])
    assert alphas == pytest.approx([0.2, 0.4, 0.4])


def test_share_block_shift_against_paired_reference():
    rows = []
    for level in range(6):
        for seed in (1, 2):
            shares = [0.0] * 9
            shares[8] = 1.0  # always (15, macro)
            rows.append({"member": f"L{level}", "seed": seed, "meta": {"pair": f"p{level}", "ordinal": 1},
                         "shares": {"zs:J_dec": shares}})
    reference_vector = [0.0] * 9
    reference_vector[6] = 1.0  # normal side: always (15, continuous)
    out = g.share_block(rows, "zs:J_dec", lambda r: reference_vector)
    assert out["shares"][8]["mean"] == pytest.approx(1.0)
    assert out["shift"][8]["mean"] == pytest.approx(1.0)
    assert out["shift"][6]["mean"] == pytest.approx(-1.0)
    assert out["total_variation"]["mean"] == pytest.approx(1.0)
    assert out["moves_toward"] == ["15-macro"]
    assert out["moves_away"] == ["15-continuous"]
    assert out["delta_shift"][2]["mean"] == pytest.approx(0.0)
    assert out["alpha_shift"][2]["mean"] == pytest.approx(1.0)
    assert out["levels"] == 6 and out["paired_units"] == 12


def test_cost_row_reads_v2_pig_columns():
    nv = g.heavy().nv
    z0 = torch.zeros(nv.DIM)
    z = torch.zeros(nv.DIM)
    z[nv.PIG_PRESENCE] = 0.4
    z[nv.PIG_CENTER[0]] = 0.3
    z[nv.PIG_CENTER[1]] = 0.4
    row = g.cost_row(3, z, z0)
    assert row["pig_presence"] == pytest.approx(0.4)
    assert row["pig_displacement"] == pytest.approx(0.5)
    assert row[g.PRIMARY_COST] == pytest.approx(0.4 - 0.1 * 0.5)
    assert g.tie_free_tensor(z[None], z0[None]).item() == pytest.approx(row[g.PRIMARY_COST])
    z[0] = math.inf
    assert g.cost_row(3, z, z0)["excluded"]


def test_wrong_anchor_rows_use_other_levels_of_the_group(monkeypatch):
    levels = {
        "a": {"cell": "1/x", "items": [{"ordinal": 0}, {"ordinal": 1}], "verdicts": {0: True, 1: False}},
        "b": {"cell": "1/y", "items": [{"ordinal": 0}, {"ordinal": 1}], "verdicts": {0: False, 1: False}},
        "c": {"cell": "1/z", "items": [{"ordinal": 0}, {"ordinal": 1}], "verdicts": {0: True, 1: False}},
    }
    monkeypatch.setattr(g, "eval_levels", lambda group: levels)
    monkeypatch.setattr(g, "SEEDS", (1,))

    def rows(first, second):
        return [{"ordinal": 0, "excluded": False, "tie_free": first},
                {"ordinal": 1, "excluded": False, "tie_free": second}]
    record = {"typed_failure": None, "levels": {"a": {"rows": rows(0.0, 1.0)},   # right for a: AUC 1
                                                "b": {"rows": rows(1.0, 0.0)},   # wrong for a: AUC 0
                                                "c": {"rows": rows(0.0, 1.0)}}}  # wrong for a: AUC 1
    records = {("L1", w, n, 1): record for w in g.worlds_of("L1") for n in g.names_of("L1", w)}
    out = g.wrong_anchor_rows(records, "L1", ("zs:J_dec",))
    by_member = {r["member"]: r["views"]["zs:J_dec"] for r in out}
    assert set(by_member) == {"a", "c"}  # b is not mixed
    assert by_member["a"] == {"right": 1.0, "wrong": 0.5, "wrong_anchors": 2}


def test_disposition_rule():
    def populations(gain, parity):
        return {"novel": {"few_shot": {"contrasts": {
            f"{a}: policy - zero-shot": {"token": g.token(gain), "estimate": gain} for a in ("J", "J_dec")} | {
            f"{a}: policy - predictor": {"token": g.non_inferiority_token(parity), "estimate": parity}
            for a in ("J", "J_dec")}}}}
    cheap = {"cheap": {"pe": True, "dec": True}}
    good = populations(block(0.05, 0.02, 0.08), block(0.0, -0.01, 0.01))
    assert g.disposition(good, cheap, True)["J_dec"]["token"] == "supported"
    assert g.disposition(good, cheap, False)["J_dec"]["token"] == g.STOP_TOKEN
    assert g.disposition(good, {"cheap": {"pe": True, "dec": False}}, True)["J_dec"]["token"] == \
        "not_supported_by_this_experiment"
    flat = populations(block(0.0, -0.01, 0.01), block(0.0, -0.01, 0.01))
    assert g.disposition(flat, cheap, True)["J_dec"]["token"] == "not_supported_by_this_experiment"
    wide = populations(block(0.03, -0.01, 0.07), block(0.0, -0.01, 0.01))
    assert g.disposition(wide, cheap, True)["J_dec"]["token"] == g.STOP_TOKEN
