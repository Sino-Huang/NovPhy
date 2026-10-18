import gzip
import json

import numpy as np
import pytest
import torch

from scripts import run_cost_alignment as ca
from world_model.data import cost_verdicts as cv
from world_model.planning import scoring_harness as harness
from world_model.training.cnn_hybrid import PAIRS


# ---------------------------------------------------------------- synthetic native trace

def entity(entity_id, position, present=True):
    return {"entity_id": entity_id, "lifecycle": "active" if present else "destroyed", "body_present": present,
            "body": {"body_type": "dynamic", "simulated": True, "position": list(position), "velocity": [0, 0],
                     "angular_velocity_degrees_per_second": 0} if present else None}


def sample(step):
    """block:0000 slides from (0, 0) to (3, 4) by step 100; block:0001 is destroyed at step 120 after moving to
    (1, 0); the pig never moves; the slingshot and the ground are not structure."""
    block0 = (0.0, 0.0) if step < 100 else (3.0, 4.0)
    block1 = (0.0, 2.0) if step < 50 else (1.0, 2.0)
    entities = [entity("runtime:bird:0000", (-10.0, 0.0)), entity("runtime:block:0000", block0),
                entity("runtime:block:0001", block1, present=step < 120), entity("runtime:pig:0000", (5.0, 5.0)),
                {"entity_id": "runtime:slingshot:0000", "body_present": True,
                 "body": {"body_type": "static", "position": [-11.0, 0.0]}}]
    return {"fixed_step": step, "entities": entities, "contacts": [], "supports": []}


EVENTS = [
    {"event_type": "bird_launched", "fixed_step": 10, "participants": ["runtime:bird:0000"]},
    {"event_type": "collision", "fixed_step": 30, "participants": ["runtime:bird:0000", "runtime:slingshot:0000"]},
    {"event_type": "collision", "fixed_step": 60, "participants": ["runtime:bird:0000", "runtime:block:0001"]},
    {"event_type": "entity_destroyed", "fixed_step": 120, "participants": ["runtime:block:0001"]},
]


def write_trace(root, first=0, last=149, chunk=75):
    root.mkdir(parents=True)
    (root / "native").mkdir()
    chunks = []
    for start in range(first, last + 1, chunk):
        end = min(start + chunk - 1, last)
        body = {"schema": "canonical_native_chunk", "fixed_step_samples": [sample(s) for s in range(start, end + 1)],
                "events": [e for e in EVENTS if start <= e["fixed_step"] <= end]}
        path = f"native/chunk-{len(chunks):06d}.json.gz"
        with gzip.open(root / path, "wb") as stream:
            stream.write(json.dumps(body).encode())
        chunks.append({"path": path, "first_fixed_step": start, "last_fixed_step": end})
    manifest = {"schema": "canonical_native_trace_v1", "status": "complete", "complete_every_native_step": True,
                "first_fixed_step": first, "last_fixed_step": last, "chunks": chunks,
                "frame_records": [{"fixed_step": s} for s in range(first, last + 1, 50)] + [{"fixed_step": last}]}
    (root / "native-manifest.json").write_text(json.dumps(manifest))
    return root


def test_branch_quantities_read_events_positions_and_presence(tmp_path):
    root = write_trace(tmp_path / "attempt" / "aligned" / "capture")
    assert cv.native_root(tmp_path / "attempt") == root
    q = cv.branch_quantities(root)
    assert q["initial"] == {"block": 2, "pig": 1}
    assert q["destroyed"] == {"block": 1, "pig": 0}
    assert q["present_end"] == {"block": 1, "pig": 1}
    # block:0000 moved 5 by the last step, block:0001 moved 1 before its removal, the pig 0
    assert q["structure_displacement"] == pytest.approx(6.0)
    assert q["pig_displacement"] == 0.0
    # the slingshot collision does not count; the bird-block collision at step 60 does
    assert q["first_structure_contact_step"] == 60
    assert q["first_structure_contact_frame"] == pytest.approx(1.2)
    assert q["pig_channel_events"] == []
    assert cv.pig_removed_from_events(q) is False


def test_chunk_events_tail_parse_equals_full_parse(tmp_path):
    root = write_trace(tmp_path / "trace")
    manifest = json.loads((root / "native-manifest.json").read_text())
    for descriptor in manifest["chunks"]:
        raw = gzip.open(root / descriptor["path"]).read()
        assert cv.chunk_events(raw) == json.loads(raw)["events"]


def quantities(destroyed=0, displacement=0.0, contact=None, first=30000):
    return {"destroyed": {"block": destroyed, "pig": 0}, "structure_displacement": displacement,
            "first_structure_contact_step": contact, "first_step": first}


def test_verdict_thresholds_are_the_frozen_ones():
    assert cv.verdicts(quantities(), True) == {"a": True, "b": False, "c": False, "d": False}
    assert cv.verdicts(quantities(destroyed=1), False)["b"] is True
    assert cv.verdicts(quantities(displacement=0.4999), False)["c"] is False
    assert cv.verdicts(quantities(displacement=0.5), False)["c"] is True
    edge = 30000 + cv.FRAME_STEPS * cv.EARLY_FRAMES
    assert cv.verdicts(quantities(contact=edge), False)["d"] is True
    assert cv.verdicts(quantities(contact=edge + 1), False)["d"] is False
    assert (cv.BLOCKS_DESTROYED_MIN, cv.DISPLACEMENT_MIN, cv.EARLY_FRAMES) == (1, 0.5, 150)


# ---------------------------------------------------------------- model costs

def column(slot, feature):
    return harness.CARRIER_HEADER + harness.SLOT_WIDTH * slot + feature


def test_cost_row_reads_the_four_costs_from_one_rollout():
    objective = ca.objective()
    z0 = torch.zeros(236)
    z0[column(objective.pig_slots[0], harness.PRESENCE_FEATURE)] = 1.0
    for slot in objective.block_slots:
        z0[column(slot, harness.PRESENCE_FEATURE)] = 1.0
    z_end = z0.clone()
    z_end[column(objective.pig_slots[0], harness.PRESENCE_FEATURE)] = 0.25
    z_end[column(objective.pig_slots[0], harness.CENTER_FEATURES[0])] = 0.3
    z_end[column(objective.block_slots[0], harness.PRESENCE_FEATURE)] = 0.0
    z_end[column(objective.block_slots[1], harness.CENTER_FEATURES[1])] = 0.4
    z_early = z0.clone()
    z_early[column(objective.block_slots[2], harness.CENTER_FEATURES[0])] = 0.2
    row = ca.cost_row(3, z_end, z_early, 150, z0)
    reference = harness.EndpointCosts(objective).row(3, z_end, z0)
    assert row["a"] == reference[harness.PRIMARY_COST]
    assert row["a"] == pytest.approx(0.25 - 0.1 * 0.3)
    assert row["b"] == pytest.approx(len(objective.block_slots) - 1 - 0.1 * 0.4)
    assert row["c"] == pytest.approx(-(0.3 + 0.4))
    assert row["d"] == pytest.approx(-0.2)
    assert row["early_frame"] == 150 and not row["excluded"] and not row["early_excluded"]


def test_cost_row_excludes_nonfinite_carriers_per_readout():
    z0 = torch.zeros(236)
    bad = z0.clone()
    bad[5] = float("nan")
    late = ca.cost_row(0, bad, z0, 150, z0)
    assert late["excluded"] and late["a"] is None and late["b"] is None and late["c"] is None
    assert late["d"] == 0.0
    early = ca.cost_row(0, z0, bad, 90, z0)
    assert early["early_excluded"] and early["d"] is None and early["a"] is not None


class Additive:
    """carrier(z, a, pair) adds pair.delta to column 0 (a position counter); stands in for a predictor."""

    def carrier(self, z, actions, pair):
        out = z.clone()
        out[:, 0] += pair.delta
        return out


def test_rollout_keeps_the_endpoint_and_the_last_carrier_at_or_before_frame_150(monkeypatch):
    monkeypatch.setattr(ca.rp, "pairs_for", lambda model: PAIRS)
    monkeypatch.setattr(ca.rp, "linear_macs", lambda model, pair: 1)
    z0 = torch.zeros(236)
    actions = torch.zeros(2, 5)
    for arm, delta in (("F-15-macro", 15), ("F-5-continuous", 5), ("F-1-micro", 1)):
        z, z_early, early_at, timing = ca.rollout(Additive(), None, z0, actions, ca.t96.arm_spec(arm))
        assert z[:, 0].tolist() == [225.0, 225.0]
        assert z_early[:, 0].tolist() == [150.0, 150.0] and early_at.tolist() == [150, 150]
        assert timing["transition_calls"] == 2 * 225 // delta


# ---------------------------------------------------------------- statistics

def test_correlation_block_is_level_clustered_and_undefined_for_constants():
    x = np.array([0, 1, 0, 1, 0, 1, 0, 1], float)
    clusters = ["L1", "L1", "L2", "L2", "L3", "L3", "L4", "L4"]
    block = ca.correlation_block(x, x, clusters)
    assert block["mean"] == pytest.approx(1.0) and block["interval"] == pytest.approx([1.0, 1.0])
    assert block["clusters"] == 4
    assert ca.correlation_block(x, np.ones(8), clusters) is None
    assert ca.pearson(ca.demeaned(x, clusters), ca.demeaned(1 - x, clusters)) == pytest.approx(-1.0)


def item(ordinal, x, key):
    return {"ordinal": ordinal, "action": {"drag_x": x, "drag_y": 10, "release_time_ms": 1000, "tap_time_ms": 0},
            "key": key}


def test_cost_entries_and_mixed_cells_follow_each_cost():
    items = {"L1": [item(0, -40, "L1-c0"), item(1, -30, "L1-c1")], "L2": [item(0, -40, "L2-c0"), item(1, -30, "L2-c1")]}
    verdicts = {"L1-c0": {"a": True, "b": False}, "L1-c1": {"a": False, "b": False},
                "L2-c0": {"a": False, "b": True}, "L2-c1": {"a": False, "b": False}}
    assert list(ca.mixed(ca.cost_entries(items, verdicts, "a"))) == ["L1"]
    assert list(ca.mixed(ca.cost_entries(items, verdicts, "b"))) == ["L2"]
    assert ca.cost_entries(items, verdicts, "b")["L2"]["verdicts"] == {0: True, 1: False}


def test_prior_uses_the_cost_verdict_of_identical_fit_actions(monkeypatch):
    fit = [{"key": "F1", "action": item(0, -40, "")["action"], "inventory_entries": [{"inventory": "grid"}]},
           {"key": "F2", "action": item(0, -40, "")["action"], "inventory_entries": [{"inventory": "grid"}]},
           {"key": "F3", "action": item(0, -30, "")["action"], "inventory_entries": [{"inventory": "angle"}]}]
    monkeypatch.setattr(ca, "fit_branches", lambda: fit)
    verdicts = {"F1": {"c": True}, "F2": {"c": False}, "F3": {"c": True}}
    stats = ca.prior_stats(verdicts, "grid", "c")
    assert stats == {(-40, 10, 1000, 0): [1, 2]}
    rows = ca.prior_rows({"inventory": [item(0, -40, "k"), item(1, -30, "k2")]}, stats, "c")
    assert [r["c"] for r in rows] == [-0.5, 0.0]


def view(auc):
    return {"auc": auc}


def test_macro_minus_continuous_is_the_delta_mean_paired_difference():
    rows = []
    for member in ("L1", "L2", "L3"):
        views = {}
        for d in ca.DELTAS:
            views[f"F-{d}-macro"] = view(0.8)
            views[f"F-{d}-continuous"] = view(0.7 if d != 15 else 0.8)
        rows.append({"member": member, "seed": 1, "views": views})
    out = ca.macro_minus_continuous(rows)
    assert out["delta_mean"]["estimate"]["mean"] == pytest.approx(0.2 / 3)
    assert out["delta_1"]["estimate"]["mean"] == pytest.approx(0.1)
    assert out["delta_15"]["estimate"]["mean"] == pytest.approx(0.0)
    assert out["delta_1"]["token"] == "supported"
    assert out["delta_15"]["token"] == "not_supported_by_this_experiment"


def cost(alignment, token, guard=True, combined=None):
    return {"guard_pass": guard, "alignment": alignment, "token": token,
            "alignment_combined": alignment if combined is None else combined}


SUP, NOT, STOP = "supported", "not_supported_by_this_experiment", ca.STOP_TOKEN


def test_disposition_absolute_rule_is_primary():
    base = {"a": cost(0.06, SUP), "b": cost(0.09, SUP), "c": cost(0.05, SUP), "d": cost(0.3, NOT)}
    out = ca.disposition(base, True)
    assert out["token"] == SUP and out["least_aligned"] == "c" and out["rule"] == "absolute"
    assert ca.disposition(base, False)["token"] == STOP
    # the least-aligned cost fails the mixed-cell guard: the next eligible one (|r| < 0.1) decides
    out = ca.disposition({**base, "c": cost(0.05, SUP, guard=False)}, True)
    assert out["least_aligned"] == "b" and out["token"] == SUP
    # d is aligned (|r| >= 0.1), so its NOT never decides; b's does once b is the least aligned
    assert ca.disposition({**base, "c": cost(0.2, SUP)}, True)["least_aligned"] == "b"
    assert ca.disposition({**base, "b": cost(0.09, NOT), "c": cost(0.2, SUP)}, True)["token"] == NOT
    # no macro advantage under the control
    assert ca.disposition({**base, "a": cost(0.06, NOT)}, True)["token"] == NOT
    # the margin straddled under c*
    assert ca.disposition({**base, "c": cost(0.05, STOP)}, True)["token"] == STOP
    # no alternative below 0.1
    out = ca.disposition({**base, "b": cost(0.2, SUP), "c": cost(0.1, SUP)}, True)
    assert out["token"] == STOP and out["least_aligned"] is None


def test_disposition_sensitivity_rules_compare_with_the_control():
    base = {"a": cost(0.06, SUP, combined=0.18), "b": cost(0.09, SUP, combined=0.21),
            "c": cost(0.08, SUP, combined=0.10), "d": cost(0.14, NOT, combined=0.14)}
    relative = ca.disposition(base, True, "relative")
    assert relative["token"] == STOP and relative["least_aligned"] is None
    combined = ca.disposition(base, True, "combined")
    assert combined["least_aligned"] == "c" and combined["token"] == SUP
