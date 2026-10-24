"""Issue #127: tests of the shared protocol module (scripts/external_baselines.py).

(a) the statistics path reproduces #123's published grid blocks exactly from its published records and the
Gate-B / #101 references; (b) G-rank; (c) dispositions, the program claim, the budget safeguard; (d) the selection
rule; (e) MACs; (f) direct-predictor rows; (g) the fit shots, the shared anchor frames, pins, G-records, ledger.
CPU only (CUDA_VISIBLE_DEVICES="")."""
from __future__ import annotations

import json
import math
import tempfile

import numpy as np
import pytest
import torch
from torch import nn

from scripts import external_baselines as eb

HJ123 = eb.ROOT / ".local-artifacts/issue-123-hjepa-comparison-v1"


def canonical(value):
    """The published form (compute.json is json_text of the computed tables)."""
    return json.loads(json.dumps(value))


# ---------------------------------------------------------------------------
# (a) statistics regression against the published #123 compute.json
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def published():
    compute = eb.read_json(HJ123 / "compute.json")
    cells = eb.evaluation_cells()
    references = eb.references()
    records = {arm: {seed: eb.read_json(HJ123 / "records" / f"{arm}--seed{seed}.json") for seed in eb.SEEDS}
               for arm in ("HJ-px", "HJ-car-m")}
    arms = {"HJ-px": (records["HJ-px"], "tie_free"), "HJ-car-m": (records["HJ-car-m"], "tie_free"),
            "HJ-px:R2": (records["HJ-px"], "tie_free_R2")}
    rows = {inventory: eb.view_rows(cells, inventory, arms, references) for inventory in ("grid", "offset")}
    return {"compute": compute, "cells": cells, "records": records, "rows": rows}


def test_grid_rankers_reproduce_123(published):
    grid = published["compute"]["inventories"]["grid"]
    names = ["U", "C*", "P", "J", "J_dec", "HJ-px", "HJ-car-m", "HJ-px:R2"]
    table = canonical(eb.ranker_table(published["rows"]["grid"], names))
    assert table["cells"] == grid["cells"] == 933 and table["members"] == grid["members"] == 311
    for name in names:
        assert table["rankers"][name] == grid["rankers"][name], name
    u = table["rankers"]["U"]["auc"]
    assert (round(u["mean"], 4), round(u["interval"][0], 4), round(u["interval"][1], 4)) == (0.7444, 0.7269, 0.7620)
    assert round(table["rankers"]["C*"]["auc"]["mean"], 4) == 0.7834
    assert round(table["rankers"]["P"]["auc"]["mean"], 4) == 0.7203
    assert round(table["rankers"]["J"]["auc"]["mean"], 4) == 0.7057
    assert round(table["rankers"]["J_dec"]["auc"]["mean"], 4) == 0.7460


def test_offset_rankers_reproduce_123(published):
    """The per-inventory prior statistics (P) and the descriptive inventories go through the same path."""
    offset = published["compute"]["inventories"]["offset"]
    table = canonical(eb.ranker_table(published["rows"]["offset"], ["U", "P", "HJ-px"]))
    for name in ("U", "P", "HJ-px"):
        assert table["rankers"][name] == offset["rankers"][name], name


@pytest.mark.parametrize("label,first,second", [("PC1", "HJ-px", "U"), ("PC2", "HJ-car-m", "U"),
                                                ("S9_PC1_R2", "HJ-px:R2", "U"), ("S5_pixels", "HJ-px", "C*"),
                                                ("S6_pixels", "HJ-px", "P"), ("S11_pixels_J_dec", "HJ-px", "J_dec")])
def test_grid_contrasts_reproduce_123(published, label, first, second):
    expected = published["compute"]["inventories"]["grid"]["contrasts"][label]
    got = canonical(eb.contrast(published["rows"]["grid"], first, second, expected["guards_pass"]))
    assert got == expected
    if label == "PC1":
        est = got["estimate"]
        assert (round(est["mean"], 4), round(est["interval"][0], 4), round(est["interval"][1], 4)) == (
            -0.2447, -0.2688, -0.2204)


def test_contrast_tokens_when_guarded(published):
    """Guarded, the tokens are c2.token of the estimate and of its reverse (U - HJ-px is supported)."""
    got = eb.contrast(published["rows"]["grid"], "HJ-px", "U", True)
    assert got["token"] == "not_supported_by_this_experiment" and got["reverse_token"] == "supported"
    assert eb.disposition(got, True)["disposition"] == 2
    assert eb.disposition(got, False)["disposition"] == 4


def test_wrong_anchor_reproduces_123(published):
    expected = published["compute"]["inventories"]["grid"]["S10_wrong_anchor"]["HJ-px"]
    got = canonical(eb.wrong_anchor_table(published["cells"], published["records"]["HJ-px"], "HJ-px", "tie_free",
                                          expected["guards_pass"]))
    assert got == expected


def test_nonfinite_share_reproduces_123(published):
    expected = published["compute"]["guards"]["arms"]["HJ-px"]["nonfinite_shares"]
    records = list(published["records"]["HJ-px"].values())
    assert eb.nonfinite_share(records, "tie_free") == expected["R1"]
    assert eb.nonfinite_share(records, "tie_free_R2") == expected["R2"]
    assert eb.nonfinite_share([{"typed_failure": "x"}], "tie_free") is None


# ---------------------------------------------------------------------------
# (b) G-rank
# ---------------------------------------------------------------------------

def spread(levels, k=16):
    return np.tile(np.arange(k, dtype=np.float64), (levels, 1))


def test_g_rank_tie_scale():
    small = spread(1)
    small[0, 1] = small[0, 0] + 5e-7  # |s| <= 15 -> tolerance 1.5e-5: tied
    small[0, 2] = 0.1
    small[0, 3] = 0.1 + 2e-6  # tied as well (2e-6 <= 1.5e-5)
    assert eb.g_rank(small)["mean_tied_pair_share"] == pytest.approx(2 / 120)
    unit = np.linspace(0.0, 0.5, 16)[None]  # max |s| < 1 -> absolute tolerance 1e-6
    unit[0, 1] = unit[0, 0] + 5e-7
    unit[0, 2] = unit[0, 3] - 2e-6
    assert eb.g_rank(unit)["mean_tied_pair_share"] == pytest.approx(1 / 120)
    big = (1e4 + np.arange(16) * 1.0)[None]  # max |s| ~ 1e4 -> tolerance ~ 1e-2
    big[0, 1] = big[0, 0] + 5e-3
    big[0, 2] = big[0, 0] + 2e-2 + 3.0  # distinct
    big[0, 3] = big[0, 4] - 2e-2
    assert eb.g_rank(big)["mean_tied_pair_share"] == pytest.approx(1 / 120)


def test_g_rank_zero_spread_and_fully_tied_boundary():
    costs = spread(10)
    costs[0] = 3.25  # zero spread: fully tied
    result = eb.g_rank(costs)
    assert result["levels"] == 10 and result["fully_tied_level_share"] == pytest.approx(0.10)
    assert result["mean_tied_pair_share"] == pytest.approx(0.10) and result["pass"]
    costs[1] = 7.0 + np.arange(16) * 1e-7  # spread below the tolerance: fully tied too
    result = eb.g_rank(costs)
    assert result["fully_tied_level_share"] == pytest.approx(0.20) and not result["pass"]


def test_g_rank_mean_share_boundary():
    level = np.array([0.0] * 10 + [1.0] * 6)  # C(10,2) + C(6,2) = 60 of 120 pairs tied
    assert eb.g_rank(np.tile(level, (4, 1))) == {"levels": 4, "mean_tied_pair_share": 0.5,
                                                 "fully_tied_level_share": 0.0, "pass": True}
    over = np.array([0.0] * 10 + [1.0] * 5 + [1.0 + 1e-9])  # 45 + 15 = 60 still tied (1e-9 within tolerance)
    over[9] = over[0] + 1e-9
    worse = np.array([0.0] * 11 + [1.0] * 5)  # 55 + 10 = 65 > 60
    result = eb.g_rank(np.stack([level, level, level, worse]))
    assert result["mean_tied_pair_share"] > 0.5 and not result["pass"]
    assert eb.g_rank(np.tile(over, (2, 1)))["pass"]


def test_g_rank_nonfinite_counts_tied():
    costs = spread(1)
    costs[0, 0] = np.nan
    assert eb.g_rank(costs)["mean_tied_pair_share"] == pytest.approx(15 / 120)
    assert eb.g_rank(np.full((1, 16), np.nan))["fully_tied_level_share"] == 1.0


# ---------------------------------------------------------------------------
# (c) dispositions, program claim, budget safeguard
# ---------------------------------------------------------------------------

def xu(mean, low, high):
    return {"estimate": {"mean": mean, "interval": [low, high]}}


@pytest.mark.parametrize("block,guards,expected", [
    (xu(0.03, 0.01, 0.05), True, 1), (xu(0.03, 0.01, 0.05), False, 4),
    (xu(0.02, 0.001, 0.04), True, 1),  # mean == delta, lower > 0: supported
    (xu(0.03, 0.0, 0.05), True, 4),  # lower bound not > 0
    (xu(-0.03, -0.05, -0.01), True, 2), (xu(-0.03, -0.05, -0.01), False, 4),
    (xu(0.0, -0.015, 0.015), True, 3), (xu(0.0, -0.015, 0.015), False, 4),
    (xu(0.0, -0.02, 0.01), True, 4),  # the interval touches -delta: not inside (-delta, +delta)
    (xu(0.01, -0.005, 0.02), True, 4),  # touches +delta
    ({"estimate": None}, True, 4),
])
def test_disposition(block, guards, expected):
    out = eb.disposition(block, guards)
    assert out["disposition"] == expected and out["label"] == eb.DISPOSITION_LABELS[expected]


def full_set(**overrides):
    per_arm = {arm: {"disposition": 2, "guards_pass": True, "gfid_pass": True, "dropped": None}
               for arm in eb.PROGRAM_SET}
    per_arm.update(overrides)
    return per_arm


def test_program_claim():
    assert eb.PROGRAM_SET == ("A1", "A2", "A4", "B1", "C1-dino", "D1")
    assert eb.program_claim(full_set())["holds"]
    missing = full_set()
    del missing["B1"]
    assert eb.program_claim(missing)["vetoes"] == [{"arm": "B1", "reason": "not_run"}]
    guard = eb.program_claim(full_set(D1={"disposition": 2, "guards_pass": False, "gfid_pass": True, "dropped": None}))
    assert not guard["holds"] and guard["vetoes"] == [{"arm": "D1", "reason": "guard_failed"}]
    gfid = eb.program_claim(full_set(A1={"disposition": 2, "guards_pass": True, "gfid_pass": False, "dropped": None}))
    assert gfid["vetoes"] == [{"arm": "A1", "reason": "g_fid_failed"}]
    dropped = eb.program_claim(full_set(A4={"disposition": None, "guards_pass": False, "gfid_pass": False,
                                            "dropped": "fidelity_not_reproduced"}))
    assert not dropped["holds"] and {"arm": "A4", "reason": "dropped: fidelity_not_reproduced"} in dropped["vetoes"]
    one = eb.program_claim(full_set(**{"C1-dino": {"disposition": 1, "guards_pass": True, "gfid_pass": True,
                                                   "dropped": None}}))
    assert not one["holds"] and one["disposition_1_arms"] == ["C1-dino"]
    extra = full_set(A3={"disposition": 4, "guards_pass": False, "gfid_pass": False, "dropped": None})
    assert eb.program_claim(extra)["holds"]  # A3 is never a member


def test_budget_safeguard():
    assert eb.budget_safeguard("supported", 3000.0, 1294.0)["fires"]
    assert not eb.budget_safeguard("supported", 1294.0, 1294.0)["fires"]
    assert not eb.budget_safeguard("readiness_or_precision_insufficient", 9e9, 1.0)["fires"]
    assert not eb.budget_safeguard("supported", 1000.0, 1294.0)["fires"]


# ---------------------------------------------------------------------------
# (d) selection
# ---------------------------------------------------------------------------

def test_select_configuration():
    pick = eb.select_configuration
    assert pick({"a": {"auc": 0.74441, "parameters": 100}, "b": {"auc": 0.74439, "parameters": 50}})["selected"] == "b"
    assert pick({"a": {"auc": 0.7444, "parameters": 50}, "b": {"auc": 0.7444, "parameters": 50}})["selected"] == "a"
    assert pick({"a": {"auc": 0.7444, "parameters": 50}, "b": {"auc": 0.7446, "parameters": 900}})["selected"] == "b"
    assert pick({"a": {"auc": None, "parameters": 1}, "b": {"auc": 0.5, "parameters": 9}})["selected"] == "b"
    assert pick({"a": {"auc": None, "parameters": 1}})["selected"] is None
    out = pick({"a": {"auc": 0.6, "parameters": 1}})
    assert set(out) == {"selected", "rule", "results"}


# ---------------------------------------------------------------------------
# (e) MACs
# ---------------------------------------------------------------------------

def test_macs_per_candidate():
    mlp = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 3))
    assert eb.macs_per_candidate(mlp, torch.randn(4, 10), candidates=4) == 10 * 20 + 20 * 3
    conv = nn.Sequential(nn.Conv2d(3, 8, 3), nn.Flatten(), nn.Linear(8 * 8 * 8, 5))
    assert eb.macs_per_candidate(conv, torch.randn(2, 3, 10, 10), candidates=2) == 8 * 8 * 8 * 3 * 3 * 3 + 512 * 5


def test_wall_per_decision():
    assert eb.wall_per_decision(lambda d: None, range(3)) >= 0.0


# ---------------------------------------------------------------------------
# (f) direct-predictor rows
# ---------------------------------------------------------------------------

def test_score_rows_sign_and_nonfinite():
    from scripts import run_decision_chain_attribution as m99
    items = [{"ordinal": o} for o in range(5)]
    rows = eb.score_rows(items, torch.tensor([2.0, -1.0, float("nan"), 0.5, float("inf")]), extras={"p": [1, 2, 3, 4, 5]})
    assert [r["neg_score"] for r in rows] == [-2.0, 1.0, None, -0.5, None]
    assert [r["excluded"] for r in rows] == [False, False, True, False, True]
    assert rows[2]["p"] is None and rows[0]["p"] == 1.0
    entry = {"verdicts": {0: True, 1: False, 2: False, 3: False, 4: False}}
    view = m99.view(entry, rows, "neg_score")
    assert view["auc"] == 1.0 and view["top1_hit"] is True and view["nonfinite"] == 2
    flipped = {"verdicts": {0: False, 1: True, 2: False, 3: False, 4: False}}
    assert m99.view(flipped, rows, "neg_score")["auc"] == 0.0


# ---------------------------------------------------------------------------
# (g) fit shots, policy split, shared products, pins, G-records, ledger
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shots():
    return eb.FitShots()


def test_fit_shots_invariants(shots):
    from scripts import run_decision_dynamics_retrain as rt
    assert len(rt.fit_branches()) == 13920 and len(shots) == 13919  # one scheduled branch is a typed failure
    assert len(shots.levels) == 480 and shots.levels == sorted(shots.levels)
    assert shots.launch4.shape == (13919, 4) and shots.launch5.shape == (13919, 5) and shots.verdicts.shape == (13919,)
    fit = eb.read_json(rt.verdicts_path(rt.OUTPUT))["branches"]
    campaign = rt.fit_verdicts()
    assert shots.verdicts.tolist() == [bool(b["pig_removed"]) for b in fit] == [campaign[k] for k in shots.keys]
    i = shots.row_of["issue-109-sc-fit-type010103-001-c00"]
    assert shots.actions[i] == {"drag_x": -80, "drag_y": 10, "release_time_ms": 1000, "tap_time_ms": 0}
    assert torch.equal(shots.launch4[i], torch.tensor([-80 / 480, 10 / 480, 1.0, 0.0], dtype=torch.float32))
    assert torch.equal(shots.launch5[i], torch.tensor([-80 / 480, 10 / 480, 1.0, 0.0, 1.0], dtype=torch.float32))
    rows = shots.carrier_rows([i, i], [0, int(shots.ends[i])])
    assert rows.shape == (2, 236) and torch.equal(rows[1], shots._data.carrier(i, int(shots.ends[i])))


def test_grid_items():
    items = eb.grid_items()
    assert [i["ordinal"] for i in items] == list(range(16))
    assert {(i["action"]["drag_x"], i["action"]["drag_y"]) for i in items} == {
        (x, y) for x in (-40, -80, -120, -160) for y in (5, 30, 55, 80)}
    assert {i["action"]["release_time_ms"] for i in items} == {1000}


def test_anchor_frames_are_decision_frames(shots):
    """The shared products hold frame 0 of the anchor branches: equal to the stores and to a fresh decode."""
    from scripts import run_slot_encoder_fix as s99
    fit = eb.fit_anchors()
    frames = eb.fit_anchor_frames()
    assert len(fit["carriers"]) == len(frames) == 480 and set(frames) == set(fit["keys"])
    policy = eb.policy_anchors()
    pframes = eb.policy_anchor_frames()
    assert len(policy["carriers"]) == len(pframes) == 120 and set(pframes) == set(policy["keys"])
    for member in sorted(frames)[:2]:
        key = fit["keys"][member]
        assert torch.equal(frames[member], shots.frame(shots.row_of[key], 0))
        _, root, shot, _ = s99.shot_frames(eb.rt.CAMPAIGN / "issue109-fit" / "attempts" / key)
        assert torch.equal(frames[member], s99.image_tensor(root / shot[0]["agent_observation"]["relative_path"]))
    member = sorted(pframes)[0]
    key = policy["keys"][member]
    _, root, shot, _ = s99.shot_frames(eb.rt.CAMPAIGN / "issue109-policy" / "attempts" / key)
    assert torch.equal(pframes[member], s99.image_tensor(root / shot[0]["agent_observation"]["relative_path"]))
    manifest = eb.read_json(eb.SHARED / "MANIFEST.json")
    for product in manifest["products"]:
        assert eb.sha256_of(eb.ROOT / product["artifact"]) == product["sha256"]


def test_policy_cells_and_oracle_auc():
    cells = eb.policy_cells("grid")
    assert cells and all(set(e["verdicts"].values()) == {True, False} for e in cells.values())
    assert all(m.startswith("issue-109-sc-policy") for m in cells)
    oracle = {m: [{"ordinal": o, "tie_free": -float(v)} for o, v in e["verdicts"].items()] for m, e in cells.items()}
    assert eb.policy_grid_auc(oracle, "tie_free") == 1.0
    shots = eb.policy_shots()
    assert len(shots) == 3480 and all(int(s["positions"][0]) == 0 for s in shots.values())


def test_pins():
    entries = eb.shared_inputs()
    names = [e["name"] for e in entries]
    assert len(names) == len(set(names)) and "module" in names and "shared_MANIFEST.json" in names
    assert {"issue_110_handoff_107", "issue_107_runner", "issue_110_runner", "issue_110_novelty_contract",
            "issue_110_novelty_labels_v2", "issue_110_design_plan", "issue_110_label_failures",
            "issue_123_eval_positions"} <= set(names)
    eb.verify_pins(entries)
    drifted = [{**entries[0], "sha256": "sha256:" + "0" * 64}]
    with pytest.raises(ValueError):
        eb.verify_pins(drifted)
    from scripts import post_freeze_amendments as amendments
    entry = amendments.AMENDMENTS["scripts/run_hjepa_comparison.py"]
    eb.verify_pins([{"name": "rc", "artifact": "scripts/run_hjepa_comparison.py", "sha256": entry["replaces"][0]}])


def test_records_manifest_and_guard():
    with tempfile.TemporaryDirectory(dir=eb.ROOT / ".local-artifacts") as tmp:
        out = eb.Path(tmp)
        (out / "records").mkdir()
        eb.write_json(out / "records" / "A--seed1.json", {"x": 1})
        identity = "issue-999-test-v1"
        eb.write_json(out / "records-manifest.json", eb.records_manifest(out, identity, ["records/*.json"]))
        assert not eb.records_guard(out, identity)["pass"]  # no receipt yet
        eb.write_receipt(out, identity, {"pass": True})
        assert eb.records_guard(out, identity)["pass"]
        assert not eb.records_guard(out, "issue-998-other-v1")["pass"]
        eb.write_json(out / "records" / "B--seed1.json", {"x": 2})  # a record the manifest does not list
        assert not eb.records_guard(out, identity)["pass"]
        eb.write_receipt(out, identity, {"pass": False})
        (out / "records" / "B--seed1.json").unlink()
        assert not eb.records_guard(out, identity)["pass"]


def test_ledger_totals_counts_training_once():
    with tempfile.TemporaryDirectory(dir=eb.ROOT / ".local-artifacts") as tmp:
        out = eb.Path(tmp)
        eb.mark(out, "train-A1-seed1", 10.0, 10.0)
        eb.mark(out, "train-A1-seed1", 10.0, 10.0)  # a relaunch re-added it
        eb.mark(out, "readouts", 5.0, 4.0)
        report = out / "models" / "A1" / "report.json"
        eb.write_json(report, {"gpu_seconds": 12.0, "wall_seconds": 13.0})
        entries, gpu, wall = eb.ledger_totals(out, lambda phase: report if phase.startswith("train-") else None)
        assert entries["train-A1-seed1"]["ledger_gpu_seconds"] == 20.0
        assert (gpu, wall) == (16.0, 18.0)


def test_reference_compute():
    out = eb.reference_compute()
    assert out["U"]["trainable_parameters"] == 1837690 and out["C*"]["trainable_parameters"] == 1862396
    assert all(len(out[k]["training"]) == 3 and out[k]["rollout"]["steps"] == 15 for k in ("U", "C*"))
    assert all(math.isfinite(out[k]["inference_macs_per_candidate"]) and out[k]["inference_macs_per_candidate"] > 0
               for k in ("U", "C*"))


def test_batch_rng_and_launch_encodings():
    assert eb.batch_rng(20260908, 3).integers(0, 1 << 30) == np.random.Generator(
        np.random.PCG64([20260908, 3])).integers(0, 1 << 30)
    action = {"drag_x": -40, "drag_y": 80, "release_time_ms": 1000, "tap_time_ms": 0}
    assert torch.equal(eb.launch5(action)[:4], eb.launch4(action))
    assert eb.SEEDS == (20260908, 20260909, 20260910) and eb.DELTA == 0.02 and eb.EXIT_PAUSED == 75


# ---------------------------------------------------------------------------
# extension: guard block, per-arm block, child summary, formatter, position error
# ---------------------------------------------------------------------------

def seed_entry(**overrides):
    entry = {"G_rank": {"pass": True}, "G_action": 0.2,
             "G_collapse": {"effective_rank": 10.0, "dim": 100, "within_episode_share": 0.05},
             "policy_ceiling_auc": 0.7, "typed_failure": None}
    entry.update(overrides)
    return entry


def test_arm_guard_block_thresholds_and_seeds():
    checks = list(eb.GUARD_CHECKS)
    per = {s: seed_entry() for s in eb.SEEDS}
    block = eb.arm_guard_block(per, checks, nonfinite=0.10)  # every value at its boundary passes
    assert block["pass"] and all(block[c] for c in checks)
    weak = {**per, eb.SEEDS[2]: seed_entry(G_action=0.049)}
    block = eb.arm_guard_block(weak, checks, nonfinite=0.0)
    assert not block["G_action"] and not block["pass"] and block["seeds"][str(eb.SEEDS[0])]["G_action"]
    collapsed = {**per, eb.SEEDS[1]: seed_entry(G_collapse={"effective_rank": 9.9, "dim": 100,
                                                            "within_episode_share": 0.5})}
    assert not eb.arm_guard_block(collapsed, ["G_collapse"])["G_collapse"]
    ranked = {**per, eb.SEEDS[0]: seed_entry(G_rank={"pass": False})}
    assert not eb.arm_guard_block(ranked, ["G_rank"])["pass"]
    assert eb.arm_guard_block(ranked, ["G_action", "G_trained"])["pass"]  # only the requested checks count
    assert not eb.arm_guard_block(per, ["G_nonfinite"], nonfinite=0.11)["pass"]
    assert not eb.arm_guard_block(per, ["G_nonfinite"], nonfinite=None)["pass"]
    with pytest.raises(ValueError):
        eb.arm_guard_block(per, ["G_port"])
    assert eb.arm_guard_block({str(s): seed_entry() for s in eb.SEEDS}, checks, nonfinite=0.0)["pass"]


def test_arm_guard_block_readout_uses_mean_ceiling():
    ceilings = lambda values: {s: seed_entry(policy_ceiling_auc=v) for s, v in zip(eb.SEEDS, values)}  # noqa: E731
    block = eb.arm_guard_block(ceilings([0.7, 0.7, 0.6]), ["G_readout"])
    assert block["G_readout"] and block["policy_ceiling_auc"] == pytest.approx(2.0 / 3.0)
    assert not eb.arm_guard_block(ceilings([0.7, 0.6, 0.6]), ["G_readout"])["G_readout"]
    assert not eb.arm_guard_block({s: seed_entry(policy_ceiling_auc=None) for s in eb.SEEDS}, ["G_readout"])["pass"]


def test_arm_guard_block_missing_or_failed_seed():
    per = {s: seed_entry() for s in eb.SEEDS[:2]}
    block = eb.arm_guard_block(per, list(eb.GUARD_CHECKS), nonfinite=0.0)
    assert not block["G_trained"] and not block["G_rank"] and not block["G_action"] and block["G_readout"]
    assert block["seeds"][str(eb.SEEDS[2])]["typed_failure"] == "missing"
    failed = {**{s: seed_entry() for s in eb.SEEDS}, eb.SEEDS[1]: seed_entry(typed_failure="nonfinite_loss",
                                                                           policy_ceiling_auc=0.1)}
    block = eb.arm_guard_block(failed, ["G_trained", "G_readout"])
    assert not block["G_trained"] and block["G_readout"] and block["policy_ceiling_auc"] == pytest.approx(0.7)


def synthetic(x_good, members=40):
    """view_rows-shaped rows: U / C* / P / J_dec noisy rankers, X near-perfect (x_good) or random."""
    from scripts import run_decision_chain_attribution as m99
    rng = np.random.default_rng(5)
    rows, records = [], {s: {"typed_failure": None, "inventories": {"grid": {}}} for s in eb.SEEDS}
    for m in range(members):
        entry = {"verdicts": {o: o < 4 for o in range(16)}}
        for seed in eb.SEEDS:
            views = {}
            for name in ("U", "C*", "P", "J_dec", "X"):
                bias = 3.0 if name == "X" and x_good else (0.0 if name == "X" else 0.6)
                cost = [{"ordinal": o, "excluded": False, "tie_free": float(rng.normal() - bias * (o < 4))}
                        for o in range(16)]
                views[name] = m99.view(entry, cost, "tie_free")
                if name == "X":
                    records[seed]["inventories"]["grid"][f"m{m}"] = {"rows": cost}
            rows.append({"member": f"m{m}", "seed": seed, "views": views})
    return rows, records


GUARDS_OK = {"G_rank": True, "G_trained": True, "pass": True}


def compute_block(seconds):
    return {"trainable_parameters": 1234567, "training_gpu_seconds": {str(s): seconds for s in eb.SEEDS},
            "macs_per_candidate": 1e6, "wall_per_decision": 0.01}


@pytest.fixture(scope="module")
def good():
    return synthetic(True)


def test_arm_block_disposition_and_budget(good):
    rows, records = good
    block = eb.arm_block(rows, "X", records, "tie_free", GUARDS_OK, True, compute_block(5000.0), "primary", True)
    assert set(eb.ARM_BLOCK_FIELDS) <= set(block)
    assert set(block["contrasts"]) == {"X-U", "X-C*", "X-P", "X-J_dec"}
    assert all(c["guards_pass"] and c["first"] == "X" for c in block["contrasts"].values())
    assert block["contrasts"]["X-U"]["token"] == "supported" and block["disposition"]["disposition"] == 1
    u = np.mean([s["training_gpu_seconds"] for s in eb._reference_training("unit").values()])
    assert block["budget_safeguard"]["fires"] and block["budget_safeguard"]["u_training_gpu_s"] == pytest.approx(u)
    assert block["budget_safeguard"]["ratio"] == pytest.approx(5000.0 / u)
    assert block["ranker"]["auc"]["mean"] > 0.9 and block["nonfinite_share"] == 0.0
    cheap = eb.arm_block(rows, "X", records, "tie_free", GUARDS_OK, True, compute_block(100.0), "primary", True)
    assert cheap["disposition"]["disposition"] == 1 and not cheap["budget_safeguard"]["fires"]
    with pytest.raises(ValueError):
        eb.arm_block(rows, "X", records, "tie_free", GUARDS_OK, True, {"trainable_parameters": 1}, "primary", True)


@pytest.mark.parametrize("guards,global_ok,extra", [
    ({"G_rank": False, "G_trained": True, "pass": False}, True, None),
    (GUARDS_OK, True, {"G_fid": False}),
    (GUARDS_OK, False, None),
    ({"pass": True}, True, None),  # no G_* guard evaluated: not a pass
])
def test_arm_block_failing_guard_stops_every_token(good, guards, global_ok, extra):
    rows, records = good
    block = eb.arm_block(rows, "X", records, "tie_free", guards, global_ok, compute_block(5000.0), "primary", True,
                         extra_guards=extra)
    assert not block["guards_pass"]
    assert all(c["token"] == c["reverse_token"] == eb.STOP_TOKEN for c in block["contrasts"].values())
    assert block["disposition"]["disposition"] == 4 and not block["budget_safeguard"]["fires"]
    assert block["contrasts"]["X-U"]["estimate"]["mean"] > 0.2  # the estimate is still reported


def test_arm_block_u_better():
    rows, records = synthetic(False)
    block = eb.arm_block(rows, "X", records, "tie_free", GUARDS_OK, True, compute_block(9e9), "primary", True)
    assert block["contrasts"]["X-U"]["reverse_token"] == "supported" and block["disposition"]["disposition"] == 2
    assert not block["budget_safeguard"]["fires"]


def test_child_summary_roundtrip_and_formatter(good, tmp_path):
    rows, records = good
    a1 = eb.arm_block(rows, "X", records, "tie_free", GUARDS_OK, True, compute_block(10.0), "primary", True)
    a1["view"] = "A1"
    a3 = eb.arm_block(rows, "X", records, "tie_free", {"G_rank": False}, True, compute_block(10.0), "secondary",
                      False, dropped="engine_budget")
    a3["view"] = "A3"
    within = {"A2-A1": eb.within_family(rows, "X", "U", True)}
    assert within["A2-A1"]["label"] == "X - U"
    summary = eb.child_summary(128, "issue-128-test-v1", "2026-10-24T00:00:00Z", {"A1": a1, "A3": a3}, within,
                               {"G_records": {"pass": True}}, {"A1": {"pass": True}}, {"note": "synthetic"})
    assert summary["program_set_members"] == ["A1"] and summary["engine_seconds"] == 0
    eb.write_json(tmp_path / "summary.json", summary)
    loaded = eb.load_child_summary(tmp_path / "summary.json")
    claim = eb.program_claim(loaded["program_claim_entries"])
    assert not claim["holds"] and claim["disposition_1_arms"] == ["A1"]
    assert {"arm": "B1", "reason": "not_run"} in claim["vetoes"]
    with pytest.raises(ValueError):  # an arm block computed under other global guards
        eb.child_summary(128, "x", "t", {"A1": a1}, {}, {"G_records": {"pass": False}}, {}, {})
    with pytest.raises(ValueError):  # a program-set arm not named as in PROGRAM_SET
        eb.child_summary(128, "x", "t", {"X": {**a1, "view": "X"}}, {}, {"G_records": {"pass": True}}, {}, {})
    eb.write_json(tmp_path / "bad.json", {**summary, "schema": "other"})
    with pytest.raises(ValueError):
        eb.load_child_summary(tmp_path / "bad.json")
    lines = eb.format_arm_rows(summary["arms"])
    assert len(lines) == 4 and all(line.count("|") == 10 for line in lines)
    assert "fail (G_rank)" in lines[3] and "(dropped: engine_budget)" in lines[3] and "1,234,567" in lines[2]
    assert "supported" in lines[2] and lines[2].startswith("| A1 | primary (program set) | ")


def test_carrier_position_mse_toy():
    from world_model.planning import scoring_harness as harness
    assert (harness.CARRIER_HEADER, harness.SLOT_WIDTH, harness.CENTER_FEATURES) == (2, 13, (5, 6))
    pred = torch.zeros(236)
    pred[2 + 13 * 3 + 5], pred[2 + 13 * 3 + 6] = 0.5, 0.25
    pred[2 + 13 * 9 + 5] = 7.0  # an absent slot never counts
    presence = torch.zeros(18, dtype=torch.bool)
    presence[[3, 7]] = True
    centers = torch.zeros(18, 2)
    centers[3] = torch.tensor([0.5, 0.25])
    centers[7] = torch.tensor([0.1, 0.2])
    assert eb.carrier_position_mse(pred, presence, centers) == pytest.approx((0.01 + 0.04) / 4)
    assert eb.carrier_position_mse(pred, torch.zeros(18, dtype=torch.bool), centers) is None
    pred[0] = float("nan")
    assert eb.carrier_position_mse(pred, presence, centers) is None


def test_position_table_reproduces_123(published):
    """U / C* curves and the HJ-px record positions give #123's published position blocks exactly."""
    expected = published["compute"]["positions"]
    got = canonical(eb.position_table({**eb.reference_position_curves(), "HJ-px": published["records"]["HJ-px"]},
                                      published["cells"]))
    for name in ("U", "C*", "HJ-px"):
        assert got[name] == expected[name], name


# ---------------------------------------------------------------------------
# P-novelty: population structure (membership metadata only; no verdict value is read), reference records on a
# fit-split stand-in (CPU), the table on synthetic rows, the formatter
# ---------------------------------------------------------------------------

def test_novelty_population_structure():
    from pathlib import Path
    cells = eb.novelty_cells()
    levels = cells["levels"]
    assert len(cells["in_scope_cells"]) == 29 and {m["cell"] for m in levels.values()} == set(cells["in_scope_cells"])
    sides = {(m["side"], m["novelty_level"]) for m in levels.values()}
    assert {k for side, k in sides if side == "novel"} == set(range(1, 8)) and ("normal", 0) in sides
    assert len(levels) == 455 and sum(m["side"] == "novel" for m in levels.values()) == 199
    normal_units = {(m["pair"], m["pair_ordinal"]) for m in levels.values() if m["side"] == "normal"}
    for m in levels.values():
        if m["side"] == "novel":
            assert (m["pair"], m["pair_ordinal"]) in normal_units and m["pair_level"] == m["novelty_level"]
    grid = {i["ordinal"]: i["action"] for i in eb.grid_items()}
    for level, m in levels.items():
        assert m["anchor_key"] in {i["key"] for i in m["inventory"]} and m["anchor_key"].startswith(level)
        assert Path(m["attempt"]).name == m["anchor_key"] and len(m["inventory"]) in (15, 16)
        for item in m["inventory"]:
            expected = dict(grid[item["ordinal"]])
            if m["side"] == "novel" and m["novelty_level"] == 5:
                expected["drag_x"] = -expected["drag_x"]  # right slingshot: mirrored launches
            assert item["action"] == expected
    assert set(cells["mixed"]) <= set(levels)
    assert (Path(levels[sorted(levels)[0]]["attempt"]) / "decision-physics-v1.json").is_file()


def test_novelty_prior_rows_mirror():
    stats = eb.gb.prior_stats(eb.read_json(eb.rt.verdicts_path(eb.RETRAIN))["branches"], "grid")
    items = eb.grid_items()
    mirrored = {"inventory": [{**i, "action": {**i["action"], "drag_x": -i["action"]["drag_x"]}} for i in items]}
    plain = {"inventory": items}
    assert eb.novelty_prior_rows(mirrored, stats) == eb.novelty_prior_rows(plain, stats) == eb.gb.prior_rows(plain,
                                                                                                             stats)
    assert len({r["tie_free"] for r in eb.novelty_prior_rows(mirrored, stats)}) > 1


@pytest.fixture()
def cpu_scoring(monkeypatch):
    from scripts import run_decision_chain_attribution as m99
    monkeypatch.setattr(m99, "DEVICE", "cpu")
    monkeypatch.setattr(eb.gb, "DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)


def fit_stand_in():
    """Two mixed fit levels as one novel / normal level pair (fit data only)."""
    entries = eb.gb.inventory_entries(eb.rt.fit_levels(), eb.rt.complete_branches(), eb.rt.fit_verdicts(), "grid")
    chosen = [m for m, e in entries.items() if set(e["verdicts"].values()) == {True, False}][:2]
    levels = {}
    for member, side in zip(chosen, ("novel", "normal")):
        levels[member] = {"cell": f"{1 if side == 'novel' else 0}/stand-in", "pair": "pair-stand-in",
                          "pair_ordinal": 1, "side": side, "novelty_level": 1 if side == "novel" else 0,
                          "pair_level": 1, "inventory": entries[member]["inventory"],
                          "verdicts": entries[member]["verdicts"], "anchor_key": None, "attempt": None}
    return {"levels": levels, "in_scope_cells": ["0/stand-in", "1/stand-in"], "mixed": sorted(chosen)}


def test_novelty_reference_records_deterministic(cpu_scoring, tmp_path):
    from world_model.planning import scoring_harness as harness
    cells = fit_stand_in()
    anchors = eb.fit_anchors()["carriers"]
    load = lambda name, seed: (eb._reference_predictor(name, seed), None)  # noqa: E731
    outs = []
    for run in ("a", "b"):
        with harness.deterministic_scoring() as policy:
            outs.append(eb.novelty_reference_records(tmp_path / run, policy, cells=cells, anchors=anchors, load=load))
    files = sorted(p.name for p in eb.novelty_reference_dir(tmp_path / "a").glob("*.json"))
    assert len(files) == 7
    for name in files:
        assert (eb.novelty_reference_dir(tmp_path / "a") / name).read_bytes() == (
            eb.novelty_reference_dir(tmp_path / "b") / name).read_bytes(), name
    # Gate-B's request_block equals the fixed-pair rollout + EndpointCosts rows of the same checkpoint
    from scripts import run_lookahead_control as wlc
    from scripts import run_tau_ad_within_checkpoint as t96
    from world_model.training.matched_dynamics import pairs_for
    model = eb._reference_predictor("unit", eb.SELECTION_SEED)
    pair = pairs_for(model)[t96.arm_spec("F-15-macro")["pair_index"]]
    level = cells["mixed"][0]
    items = cells["levels"][level]["inventory"]
    z_end = eb._reference_rollout(model, pair, anchors[level], torch.stack([eb.launch5(i["action"]) for i in items]))
    expected = eb.endpoint_rows(items, z_end, anchors[level], harness.EndpointCosts(wlc.load_objective()))
    assert outs[0]["U"][0][eb.SELECTION_SEED]["inventories"]["grid"][level]["rows"] == canonical(expected)
    assert outs[0]["P"][0][eb.SEEDS[0]] is outs[0]["P"][0][eb.SEEDS[2]]
    table = eb.novelty_table(outs[0], cells)
    assert set(table["rankers"]) == {"U", "C*", "P"} and table["population"]["paired_units"] == {"1": 1}


def synthetic_novelty():
    """Level pairs u0..u5 at novelty levels 1 / 2; u5's novel side is not mixed; one unpaired normal side."""
    rng = np.random.default_rng(11)
    levels, mixed = {}, []
    spec = [(u, 1 if u < 3 else 2) for u in range(6)]
    for u, k in spec:
        for side in ("novel", "normal"):
            name = f"{side}-{u}"
            is_mixed = not (u == 5 and side == "novel")
            levels[name] = {"cell": f"{k if side == 'novel' else 0}/s", "pair": f"pair-{u % 2}", "pair_ordinal": u,
                            "side": side, "novelty_level": k if side == "novel" else 0, "pair_level": k,
                            "inventory": [{"ordinal": o, "action": None, "key": f"{name}-{o}"} for o in range(16)],
                            "verdicts": {o: (o < 4) and is_mixed for o in range(16)}}
            mixed += [name] if is_mixed else []
    levels["normal-x"] = {**levels["normal-0"], "pair_ordinal": 99, "pair_level": 8,
                          "inventory": levels["normal-0"]["inventory"]}
    mixed.append("normal-x")
    records = {}
    for name in ("A", "B"):
        records[name] = ({seed: {"typed_failure": None, "inventories": {"grid": {
            lv: {"rows": [{"ordinal": o, "excluded": False, "c": float(rng.normal() - (o < 4) * (1.0 if name == "A"
                                                                                                   else 0.2))}
                          for o in range(16)]} for lv in levels}}} for seed in eb.SEEDS}, "c")
    return {"levels": levels, "in_scope_cells": ["0/s", "1/s", "2/s"], "mixed": sorted(mixed)}, records


def test_novelty_table_blocks_and_pairing():
    from scripts import run_decision_chain_attribution as m99
    cells, records = synthetic_novelty()
    table = eb.novelty_table(records, cells)
    assert table["novelty_levels"] == [1, 2] and table["population"]["paired_units"] == {"1": 3, "2": 2}
    a = table["rankers"]["A"]

    def auc(lv, seed):
        meta = cells["levels"][lv]
        return m99.view({"inventory": meta["inventory"], "verdicts": meta["verdicts"]},
                        records["A"][0][seed]["inventories"]["grid"][lv]["rows"], "c")["auc"]

    assert a["novel"]["1"]["clusters"] == 3 and a["novel"]["1"]["units"] == 9
    assert a["novel"]["2"]["clusters"] == 2  # novel-5 is not mixed
    assert a["normal_paired"]["2"]["clusters"] == 3  # its normal side still counts as paired normal
    assert a["normal_all"]["clusters"] == 7 and a["normal_paired_pooled"]["clusters"] == 6
    assert a["novel_pooled"]["mean"] == pytest.approx(np.mean([np.mean([auc(f"novel-{u}", s) for s in eb.SEEDS])
                                                               for u in range(5)]))
    diffs = [np.mean([auc(f"novel-{u}", s) - auc(f"normal-{u}", s) for s in eb.SEEDS]) for u in range(5)]
    pooled = a["paired_pooled"]
    assert pooled["cells"] == 15 and pooled["members"] == 5  # clustered by level pair (pair/pair_ordinal)
    assert pooled["estimate"]["mean"] == pytest.approx(np.mean(diffs))
    assert a["paired"]["1"]["estimate"]["mean"] == pytest.approx(np.mean(diffs[:3]))
    assert table["rankers"]["B"]["novel_pooled"]["mean"] < a["novel_pooled"]["mean"]
    lines = eb.format_novelty_rows(table)
    assert len(lines) == 2 + 2 * 4 and all(line.count("|") == 7 for line in lines)
    assert lines[2].startswith("| A | 1 | ") and lines[4].startswith("| A | 1–2 pooled | ") and lines[4].endswith("| 5 |")
    assert lines[5].startswith("| A | 0 (all normal sides) | n/a | ")

