"""Issue #104 work item 3: the held-out #96 contrast runner's tables, G6 reading and guard propagation.

Synthetic records only (random costs on a few real level identities); no rollout and no evaluation score.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import run_heldout_96_contrasts as h
from world_model.planning import scoring_harness as harness

LEVELS_PER_INVENTORY = 4
SIZES = {"grid": 5, "offset": 4, "angle": 4, "power": 6}


def _cells(levels):
    rng = np.random.default_rng(0)
    out = {}
    for k, inventory in enumerate(h.INVENTORIES):
        out[inventory] = {}
        for level in levels[k:k + LEVELS_PER_INVENTORY]:
            n = SIZES[inventory]
            verdicts = [bool(v) for v in rng.integers(0, 2, n)]
            verdicts[0], verdicts[1] = True, False
            out[inventory][level] = {
                "inventory": [{"ordinal": o, "action": {"drag_x": -40 - o, "drag_y": 5, "release_time_ms": 1000,
                                                        "tap_time_ms": 0}, "key": f"{level}-c{o:02d}"} for o in range(n)],
                "verdicts": {str(o): verdicts[o] for o in range(n)}}
    return out


def _rle_for(arm, rng):
    pairs = 3 if h.t96.arm_family(arm) == "continuous" else 9
    first = int(rng.integers(0, pairs))
    return h.t96.rle([first] * 100 + [0] * 125)


def _record(arm, seed, cells, rng, frequencies=None, schedules=None):
    inventories = {}
    for inventory in h.INVENTORIES:
        inventories[inventory] = {}
        for level, entry in sorted(cells[inventory].items()):
            rows = [{"ordinal": item["ordinal"], "excluded": False, "tie_free": float(rng.normal()),
                     "count": float(rng.integers(0, 3)), "pig_presence": 0.5, "pig_displacement": 0.1, "steps": 225}
                    for item in entry["inventory"]]
            inventories[inventory][level] = {
                "rows": rows,
                "schedules": ({str(r["ordinal"]): _rle_for(arm, rng) for r in rows}
                              if arm in h.CONTROLLER_ARMS else None),
                "ol_schedule": h.t96.rle(schedules[(inventory, level, seed)]) if arm == "OL" else None,
                "transition_calls": 10, "controller_calls": 0, "linear_macs": 100, "gpu_seconds": 0.01}
    record = {"schema": h.SCHEMA_RECORD, "plan_identity": h.IDENTITY, "plan_version": 1,
              "scoring_policy": dict(harness.SCORING_POLICY), "cell": {"arm": arm, "seed": seed},
              "inventories": inventories, "totals": {}, "wall_seconds": 1.0, "engine_seconds": 0}
    if arm == "T":
        record["alpha_T"] = frequencies["coordinates"][str(seed)]["alpha_T"]
    if arm == "D":
        record["delta_D"] = frequencies["coordinates"][str(seed)]["delta_D"]
    return record


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    output = tmp_path_factory.mktemp("issue104")
    levels = sorted(h.load_anchor_store()["anchors"])[:LEVELS_PER_INVENTORY + len(h.INVENTORIES)]
    raw = _cells(levels)
    (output / "data").mkdir()
    h.write_json(h.cells_path(output), {"schema": h.SCHEMA_CELLS, "identity": h.IDENTITY, "rule": "synthetic",
                                        "inventories": raw})
    cells = h.load_cells(output)
    rng = np.random.default_rng(1)
    (output / "records").mkdir()
    for arm in h.PHASES[1]:
        for seed in h.SEEDS:
            h.write_json(h.record_path(output, arm, seed), _record(arm, seed, raw, rng))
    frequencies = h.j_frequencies(output, cells)
    h.write_json(output / "frequencies.json", frequencies)
    schedules = h.t96.ol_schedules(cells, frequencies)
    for arm in h.PHASES[2]:
        for seed in h.SEEDS:
            h.write_json(h.record_path(output, arm, seed), _record(arm, seed, raw, rng, frequencies, schedules))
    (output / "ledger").mkdir()
    h.write_json(output / "ledger" / "phase1-worker0.json", {
        "schema": h.SCHEMA_LEDGER, "phase": 1, "worker": 0, "pid": 1, "device": "0", "started_at": "2026-10-18T00:00:00Z",
        "finished_at": "2026-10-18T00:10:00Z", "wall_seconds": 600.0, "gpu_seconds": 500.0, "status": "terminal",
        "tasks": []})
    smoke = {"checks": {"S1_exact_96_record": True, "S2_96_g1_criterion": True},
             "census": {"exact_record": {"record": "x"}, "g1_96_criterion": [], "wall_seconds": 1.0},
             "cells_and_anchors": {"reparse_bit_equal": True}, "wall_seconds": 2.0,
             "projection": {"projected_gpu_seconds": 3.0}}
    h.write_json(output / "smoke.json", smoke)
    plan = {"version": 1, "frozen_at": "2026-10-18T00:00:00Z", "claim_boundary": "synthetic",
            "stack": {"predictor": "p", "controller": "c", "anchors": "a"}}
    return output, plan


def test_tables_tokens_and_rendering(synthetic):
    output, plan = synthetic
    compute = h.compute_tables(output, plan)
    assert compute["structure"]["records"] == len(h.ARMS) * len(h.SEEDS)
    for cost in h.COSTS:
        block = compute["costs"][cost]
        assert block["primary"]["token"] in h.t96.DISPOSITION_TOKENS
        for inventory in h.INVENTORIES:
            inv = block["inventories"][inventory]
            assert set(inv["contrasts"]) == set(h.CONTRAST_ORDER)
            g6 = inv["g6"]
            assert g6["pass"] == (g6["half_width"] <= h.G6_HALF_WIDTH)
            assert g6["levels_for_threshold_by_sqrt_scaling"] >= 1
    files = h.rendered(plan, compute, output)
    assert set(files) == {"compute.json", "summary.json", "findings.md", "comparisons.csv", "records-manifest.json"}
    assert json.loads(files["summary.json"])["disposition"] == compute["costs"][h.PRIMARY_COST]["primary"]["token"]
    assert h.rendered(plan, h.compute_tables(output, plan), output) == files


def test_failed_guard_turns_every_token_into_the_stop_token(synthetic):
    output, plan = synthetic
    smoke_path = Path(output) / "smoke.json"
    smoke = h.read_json(smoke_path)
    smoke["checks"]["S1_exact_96_record"] = False
    h.write_json(smoke_path, smoke)
    try:
        compute = h.compute_tables(output, plan)
    finally:
        smoke["checks"]["S1_exact_96_record"] = True
        h.write_json(smoke_path, smoke)
    for cost in h.COSTS:
        block = compute["costs"][cost]
        assert not block["guards"]["G1"]["pass"] and not block["guards_pass"]
        assert block["primary"]["token"] == h.STOP_TOKEN
        assert block["primary"]["token_at_96_literal_bound"] == h.STOP_TOKEN
        assert all(s["token"] == h.STOP_TOKEN for s in block["secondary_tokens"].values())
        assert block["replication"]["interpretable"] is False


def test_ol_schedule_mismatch_refuses(synthetic):
    output, plan = synthetic
    path = h.record_path(output, "OL", h.SEEDS[0])
    record = h.read_json(path)
    original = h.json_text(record)
    level = sorted(record["inventories"]["grid"])[0]
    record["inventories"]["grid"][level]["ol_schedule"] = "0x225"
    h.write_json(path, record)
    try:
        with pytest.raises(ValueError, match="OL schedule"):
            h.compute_tables(output, plan)
    finally:
        path.write_text(original)


def test_half_width_statement():
    statement = h.half_width_statement({"half_width": 0.04}, 47, 0.02)
    assert statement["pass"] is False
    assert statement["levels_for_threshold_by_sqrt_scaling"] == 188
    assert h.half_width_statement({"half_width": 0.015}, 311, 0.02)["pass"] is True
    assert h.half_width_statement(None, 10, 0.02) is None


def test_g6_threshold_is_the_96_margin_and_design_follows_the_104_depth_rule():
    assert h.G6_HALF_WIDTH == h.t96.DELTA == 0.02
    assert h.HALF_WIDTH_96_LITERAL == 0.05
    design = h.read_json(h.POWER_PLAN)["depth"]
    assert design["grid"]["required_mixed_members_max"] == 293
    assert h.DESIGN_DELTA == {"grid": 0.02, "offset": 0.05, "angle": 0.05, "power": 0.05}
