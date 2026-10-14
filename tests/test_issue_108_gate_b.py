from scripts import run_gate_b as gb


def action(x):
    return {"drag_x": x, "drag_y": 10, "release_time_ms": 1000, "tap_time_ms": 0}


def branch(level, suffix, entries, x):
    return {"key": f"{level}-{suffix}", "member": level,
            "candidate": {"suffix": suffix, "inventory_entries": entries, "action": action(x)}}


def test_cells_keep_complete_candidates_in_ordinal_order_and_only_mixed_levels():
    levels = [{"identity": "L1"}, {"identity": "L2"}, {"identity": "L3"}]
    branches = [
        branch("L1", "c13", [{"inventory": "grid", "ordinal": 1}], -40),
        branch("L1", "c14", [{"inventory": "grid", "ordinal": 0}], -30),
        branch("L1", "c15", [{"inventory": "grid", "ordinal": 2}], -20),  # incomplete: no verdict
        branch("L2", "c13", [{"inventory": "grid", "ordinal": 0}], -40),
        branch("L2", "c14", [{"inventory": "grid", "ordinal": 1}], -30),
        branch("L2", "c00", [{"inventory": "angle", "ordinal": 0}], -80),
    ]
    verdicts = {"L1-c13": True, "L1-c14": False, "L2-c13": True, "L2-c14": True, "L2-c00": False}
    entries = gb.inventory_entries(levels, branches, verdicts, "grid")
    assert list(entries) == ["L1", "L2"]  # L3 holds no grid capture
    assert [i["ordinal"] for i in entries["L1"]["inventory"]] == [0, 1]
    assert entries["L1"]["verdicts"] == {0: False, 1: True}
    assert entries["L1"]["inventory"][0]["action"] == action(-30)
    assert list(gb.mixed_cells(entries)) == ["L1"]  # L2 grid holds only removals
    assert gb.inventory_entries(levels, branches, verdicts, "angle") == {
        "L2": {"inventory": [{"ordinal": 0, "action": action(-80), "key": "L2-c00"}], "verdicts": {0: False}}}


def test_action_prior_uses_same_inventory_fit_executions_and_zero_for_unseen_actions():
    fit = [{"action": action(-40), "inventory_entries": [{"inventory": "grid"}], "pig_removed": True},
           {"action": action(-40), "inventory_entries": [{"inventory": "grid"}], "pig_removed": False},
           {"action": action(-40), "inventory_entries": [{"inventory": "grid"}], "pig_removed": True},
           {"action": action(-30), "inventory_entries": [{"inventory": "angle"}], "pig_removed": True}]
    stats = gb.prior_stats(fit, "grid")
    assert stats == {(-40, 10, 1000, 0): [2, 3]}
    entry = {"inventory": [{"ordinal": 0, "action": action(-40)}, {"ordinal": 1, "action": action(-30)}]}
    rows = gb.prior_rows(entry, stats)
    assert [r[gb.PRIMARY_COST] for r in rows] == [-2 / 3, 0.0]


def test_gate_decision_needs_mean_lower_bound_and_finite_share():
    block = {"mean": 0.70, "interval": [0.60, 0.80]}
    assert gb.gate_decision(block, 0.0) == "pass"
    assert gb.gate_decision({"mean": 0.70, "interval": [0.55, 0.80]}, 0.0) == "fail"
    assert gb.gate_decision({"mean": 0.64, "interval": [0.60, 0.70]}, 0.0) == "fail"
    assert gb.gate_decision(block, 0.11) == "fail"
    assert gb.gate_decision(None, None) == "fail"
