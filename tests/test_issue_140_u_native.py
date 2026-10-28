"""#140 U-native runner: CPU checks of step scaling, schedule stretching, G-port, pause / resume, the A4 - U-native
contrast assembly and the publish / validate round trip (the policy split stands in for the evaluation split; the
evaluation split is never read by these tests)."""
from collections import Counter
import math
import shutil
import uuid

import numpy as np
import pytest
import torch

from scripts import external_baselines as eb
from scripts import run_decision_dynamics_retrain as rt
from scripts import run_gate_b as gb
from scripts import run_issue_140_u_native as r
from scripts import run_slot_encoder_fix as s99

torch.set_num_threads(4)


@pytest.fixture(scope="module")
def data():
    with r.on_device("cpu"):
        return r.training_data()


@pytest.fixture()
def scratch():
    output = eb.ROOT / ".local-artifacts" / f"issue-140-test-{uuid.uuid4().hex[:8]}"
    yield output
    shutil.rmtree(output, ignore_errors=True)


def stand_in_plan(steps):
    return {"frozen_at": "2000-01-01T00:00:00Z", "recipe": {"steps": steps, "u_steps": rt.STEPS,
                                                            "update_ratio": steps / rt.STEPS,
                                                            "factor": {"value": r.FACTOR}},
            "skip_cap": r.SKIP_CAP, "u_reference": {"mean": 1294.19}, "a4_reference": {"mean": 2434.73},
            "queues": {"train": {"2": list(eb.SEEDS)}, "readouts": {"3": list(eb.SEEDS)}}, "decision": r.DECISION,
            "projection": r.projection()}


def test_step_scaling():
    assert r.FACTOR == 2435.0 / 1294.19
    assert r.STEPS == 16_929 and r.STEPS % r.REQUEST_CYCLE == 0
    assert r.STEPS // 9 == round(1000 * r.FACTOR) and abs(r.STEPS / rt.STEPS - r.FACTOR) < 1e-3
    assert r.SKIP_CAP == 85 == round(45 * r.FACTOR)
    # every request trains equally often, as in U (1 000 cycles -> 1 881 cycles)
    counts = Counter(rt.step_requests(r.FAMILY, r.STEPS))
    assert len(counts) == 9 and set(counts.values()) == {1881}
    # eb's U reference (the retrain reports) and #128's published A4 mean are the declared factor's terms
    u = r.u_reference()
    assert u["steps"] == [rt.STEPS] and abs(u["mean"] - r.U_GPU_S) < 0.005
    a4 = eb.read_json(r.A128 / "summary.json")["arms"]["A4"]["compute"]["training_gpu_seconds"]
    assert abs(np.mean(list(a4.values())) - r.A4_GPU_S) < 0.5
    projection = r.projection()
    assert abs(projection["training_gpu_s_per_seed"] - u["mean"] / rt.STEPS * r.STEPS) < 1e-6
    assert 0.6 < projection["training_gpu_h"] / 3 < 0.75


def test_schedule_stretching():
    native, reference = r.implementation(r.STEPS), rt.implementation()
    assert native is not reference and native.OUTPUT == r.OUTPUT and native.IDENTITY == r.IDENTITY
    assert (native.LONG["every"], native.RANK["start"], native.RANK["ramp"]) == (941, 1881, 1881)
    assert (reference.LONG["every"], reference.RANK["start"], reference.RANK["ramp"]) == (500, 1000, 1000)
    assert {k: v for k, v in native.LONG.items() if k != "every"} == {
        k: v for k, v in reference.LONG.items() if k != "every"}
    assert {k: v for k, v in native.RANK.items() if k not in ("start", "ramp")} == {
        k: v for k, v in reference.RANK.items() if k not in ("start", "ramp")}
    assert native.recipe() == {**s99.training_recipe(), "steps": r.STEPS} and reference.recipe()["steps"] == rt.STEPS
    assert native.TRACE_EVERY == reference.TRACE_EVERY and native.PROGRESS_EVERY == reference.PROGRESS_EVERY
    summary = r.schedule_summary(native, reference, r.STEPS)
    assert summary["first_update_differing_from_U"] == 500
    assert summary["horizon_reaches_225_at"] == 10_351 and summary["rank_weight_positive_from"] == 1881
    assert summary["rank_weight_full_from"] == 3761
    u = r.schedule_summary(reference, reference, rt.STEPS)
    assert u["horizon_reaches_225_at"] == 5500 and u["rank_weight_full_from"] == 1999
    assert abs(summary["horizon_225_fraction"] - u["horizon_225_fraction"]) < 1e-3
    # proportional: U-native at update k follows U at update k / f within one curriculum increment
    for k in range(0, r.STEPS, 37):
        h, w = r.schedule(native, k)
        uh, uw = r.schedule(reference, int(k / r.FACTOR))
        assert abs(h - uh) <= reference.LONG["increment"] and abs(w - uw) <= 0.1 * 2 / 1000


def test_gport_is_bitwise_and_detects_a_defect(data):
    result = r.gport_compare(updates=2, data=data)
    assert result["pass"] and all(result["checks"].values()) and result["max_abs_delta"]["parameters"] == 0.0
    defect = rt.implementation.__wrapped__()
    defect.LONG = {**defect.LONG, "first": 75}  # a different curriculum start from update 0
    bad = r.gport_compare(updates=2, native=defect, data=data)
    assert not bad["pass"] and not bad["checks"]["parameters_equal"]
    recorded = r.OUTPUT / "gport" / "gport.json"
    if recorded.is_file():
        record = eb.read_json(recorded)
        assert record["G_port"] and record["arms"][r.VIEW]["updates"] == r.GPORT_UPDATES
        assert record["schedule"]["first_update_differing_from_U"] == 500 > r.GPORT_UPDATES


def test_pause_resume_is_bitwise_and_reports_once(data, scratch, monkeypatch):
    plan = stand_in_plan(4)
    seed = eb.SEEDS[1]
    full, split = scratch / "full", scratch / "split"
    with r.on_device("cpu"):
        report = r.train_seed(full, plan, seed, data, "cpu")
        impl = r.implementation(4)
        calls = []
        for each in r._IMPLEMENTATIONS:  # restored at teardown
            monkeypatch.setattr(each, "PROGRESS_EVERY", each.PROGRESS_EVERY)
        monkeypatch.setattr(eb, "request_stop", lambda signum, frame: calls.append(signum))
        monkeypatch.setattr(eb, "STOP_REQUESTED", [15])
        r.request_pause(15, None)  # the handler: progress after the current update
        assert calls == [15] and impl.PROGRESS_EVERY == 1
        with pytest.raises(eb.TrainingPaused):
            r.train_seed(split, plan, seed, data, "cpu")
        progress = rt.implementation().progress_path(split, r.ARM, r.FAMILY, r.seed_index(seed))
        assert torch.load(progress, weights_only=False)["step"] == 1 and not r.report_path(split, seed).exists()
        monkeypatch.setattr(eb, "STOP_REQUESTED", [])
        impl.PROGRESS_EVERY = rt.implementation().PROGRESS_EVERY
        resumed = r.train_seed(split, plan, seed, data, "cpu")
    assert not progress.exists()
    a = torch.load(r.predictor_path(full, seed, 4), weights_only=True)
    b = torch.load(r.predictor_path(split, seed, 4), weights_only=True)
    assert a["steps"] == b["steps"] == 4 and a["complete"] and b["complete"]
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    ua = torch.load(r.updates_path(full, seed), weights_only=True)
    ub = torch.load(r.updates_path(split, seed), weights_only=True)
    assert all(torch.equal(ua["logs"][k], ub["logs"][k]) for k in ua["logs"]) and len(ua["requests"]) == 4
    assert report["gpu_seconds"] > 0 and resumed["steps"] == 4 and resumed["ratio_to_u"] == pytest.approx(
        resumed["gpu_seconds"] / 1294.19)
    ledger, _, _ = eb.ledger_totals(split, r.training_report_of(split))
    assert ledger[r.train_phase(seed)]["gpu_seconds"] == resumed["gpu_seconds"]  # counted once, from the report


def test_end_state_rule_types_an_excursion(scratch):
    plan = stand_in_plan(900)
    seed = eb.SEEDS[0]
    requests = rt.step_requests(r.FAMILY, 900)
    local = torch.ones(900, 1)
    local[-3 * 9:] = 10.0  # the last EXCURSION_LAST (3) updates of every request rise 10x over its reference window
    eb.write_json(r.report_path(scratch, seed), {"complete": True, "retired_at": None})
    eb.atomic_torch(r.updates_path(scratch, seed), {"requests": requests, "skipped": torch.tensor([0]),
                                                    "logs": {"local": local, "grad_norm": torch.ones(900, 1)}})
    state = r.end_state(scratch, plan, seed)
    assert state["typed_failure"] == "final_checkpoint_excursion" and not state["usable"] and state["G2_stability"]
    eb.atomic_torch(r.updates_path(scratch, seed), {"requests": requests, "skipped": torch.tensor([0]),
                                                    "logs": {"local": torch.ones(900, 1), "grad_norm": torch.ones(900, 1)}})
    assert r.end_state(scratch, plan, seed)["usable"]
    eb.write_json(r.report_path(scratch, seed), {"complete": False, "retired_at": 3})
    assert r.end_state(scratch, plan, seed)["typed_failure"] == "retired"


def _synthetic(cells, value):
    """{seed: record} with one row per candidate: value(member, entry, item, seed) under tie_free and neg_score."""
    out = {}
    for seed in eb.SEEDS:
        inventory = {}
        for member, entry in sorted(cells.items()):
            rows = []
            for item in entry["inventory"]:
                v = value(member, entry, item, seed)
                rows.append({"ordinal": item["ordinal"], "excluded": False, "tie_free": v, "neg_score": v})
            inventory[member] = {"rows": rows}
        out[seed] = {"typed_failure": None, "inventories": {"grid": inventory}}
    return out


def _a4_info(records, **overrides):
    return {"records": records, "key": "neg_score", "guards_pass": True, "arm_guards_pass": True,
            "global_guards_pass": True, "records_guard": True,
            "records_in_manifest": {str(s): True for s in eb.SEEDS}, "training_gpu_seconds": {
                str(s): 2434.73 for s in eb.SEEDS}, "summary_sha256": "sha256:x", "records_manifest_sha256": "sha256:y",
            **overrides}


def _references(records):
    fit = eb.read_json(rt.verdicts_path(rt.OUTPUT))["branches"]
    return {"U": records, "C*": records, "J": records, "J_dec": records,
            "prior": {inventory: gb.prior_stats(fit, inventory) for inventory in eb.INVENTORIES}}


def test_a4_minus_u_native_contrast_assembly():
    cells = eb.policy_cells("grid")
    rng = np.random.default_rng(0)
    noise = {(m, it["ordinal"], s): float(rng.random()) for m, e in cells.items() for it in e["inventory"]
             for s in eb.SEEDS}
    native = _synthetic(cells, lambda m, e, it, s: noise[(m, it["ordinal"], s)])
    a4 = _synthetic(cells, lambda m, e, it, s: 0.0 if e["verdicts"][it["ordinal"]] else 1.0)  # lower cost first
    rows = eb.view_rows({"grid": {"mixed": cells}}, "grid", {r.VIEW: (native, eb.PRIMARY_COST), "A4": (a4, "neg_score")},
                        _references(native))
    budget = {"ratio_to_a4_mean": 1.0}
    extra = r.u_native_extra(rows, _a4_info(a4), True, budget)
    block = extra["A4 - U-native"]
    assert (block["first"], block["second"]) == ("A4", r.VIEW) and block["guards_pass"]
    assert block["token"] == "supported" and block["reverse_token"] != "supported"
    assert block["estimate"]["mean"] > 0.3 and extra["reading"].startswith("A4 - U-native supported")
    assert extra["budget"] is budget and extra["a4_source"]["key"] == "neg_score"
    descriptive = extra["U-native - U"]
    assert descriptive["descriptive"] and "token" not in descriptive and descriptive["estimate"]["mean"] == 0.0
    # either side's guards failing is the stop token (both directions); the reading keeps the qualification
    for info, native_ok in ((_a4_info(a4, records_guard=False), True), (_a4_info(a4), False),
                            (_a4_info(a4, records_in_manifest={"20260908": False}), True)):
        stopped = r.u_native_extra(rows, info, native_ok, budget)
        assert stopped["A4 - U-native"]["token"] == eb.STOP_TOKEN == stopped["A4 - U-native"]["reverse_token"]
        assert stopped["reading"].startswith("A4 - U-native not supported")


def test_publish_validate_round_trip_on_policy_stand_in(data, scratch, monkeypatch):
    output = scratch
    plan = stand_in_plan(2)
    monkeypatch.setattr(r, "load_plan", lambda out: plan)
    fit = eb.fit_anchors()
    subset = {m: fit["carriers"][m] for m in sorted(fit["carriers"])[:24]}
    monkeypatch.setattr(eb, "fit_anchors", lambda: {"carriers": subset, "keys": {m: fit["keys"][m] for m in subset}})
    monkeypatch.setattr(gb, "fit_anchors", lambda: subset)
    gport = {"G_port": True}
    eb.write_json(output / "gport" / "gport.json", gport)
    with r.on_device("cpu"):
        for seed in eb.SEEDS:
            r.train_seed(output, plan, seed, data, "cpu")
            r.readouts_seed(output, plan, seed, "cpu")
    readout = eb.read_json(r.readouts_path(output, eb.SEEDS[0]))
    assert readout["typed_failure"] is None and readout["G_rank"]["levels"] == 24
    assert readout["trainable_parameters"] == 1_837_690 and readout["macs_per_candidate"] > 0
    assert 0.0 <= readout["policy_auc"] <= 1.0 and math.isfinite(readout["G_action"])
    assert r.freeze_readouts(output) == 0
    cells = {inv: {"mixed": eb.policy_cells(inv)} for inv in eb.INVENTORIES}
    monkeypatch.setattr(eb, "evaluation_cells", lambda: cells)

    def stand_in(out, policy):
        payload = {"carriers": eb.policy_anchors()["carriers"], "read_at": eb.utc_now(), "scoring_policy": policy}
        eb.atomic_torch(eb.evaluation_anchors_path(out), payload)
        return payload

    monkeypatch.setattr(eb, "read_evaluation_anchors", stand_in)
    monkeypatch.setattr(r, "score_positions", lambda *args: {})
    assert r.score(output, "cpu") == 0
    records = r.load_records(output)
    record = records[eb.SEEDS[2]]
    assert record["schema"] == r.SCHEMA_RECORD and record["gate_b_schema"] == gb.SCHEMA_RECORD
    assert record["cell"]["name"] == r.VIEW and record["cell"]["request"] == "F-15-macro"
    assert set(record["inventories"]) == set(eb.INVENTORIES) and len(record["wrong_anchor"]) == 24
    assert record["checkpoint_sha256"] == eb.sha256_of(r.predictor_path(output, eb.SEEDS[2], 2))
    assert record["wall_per_decision"] > 0 and eb.nonfinite_share(records.values(), eb.PRIMARY_COST) == 0.0
    a4 = {s: {**rec, "inventories": {inv: {m: {"rows": [{**row, "neg_score": -row["tie_free"]} for row in c["rows"]]}
                                           for m, c in block.items()} for inv, block in rec["inventories"].items()}}
          for s, rec in records.items()}
    monkeypatch.setattr(r, "a4_source", lambda: _a4_info(a4))
    monkeypatch.setattr(eb, "references", lambda: _references(records))
    monkeypatch.setattr(r, "reference_curves", lambda: {"U": {s: {"typed_failure": None, "positions": {}}
                                                             for s in eb.SEEDS}})
    assert r.publish(output) == 0
    assert eb.load_child_summary(output / "summary.json")["global_guards"]["G_records"]["pass"] is False
    assert r.validate(output, "cpu") == 0
    summary = eb.load_child_summary(output / "summary.json")
    assert summary["issue"] == 140 and summary["program_set_members"] == [] and summary["global_guards_pass"]
    block = summary["arms"][r.VIEW]
    assert block["role"] == "control" and not block["program_set"] and block["guards"]["G_port"] is True
    assert block["contrasts"]["X-U"]["estimate"]["mean"] == 0.0  # U stands in as U-native itself
    u = summary["extra"]["u_native"]
    assert set(u) == {"A4 - U-native", "U-native - U", "budget", "reading", "a4_source"}
    assert u["A4 - U-native"]["first"] == "A4" and u["budget"]["steps"] == 2
    assert u["budget"]["ratio_to_a4"][str(eb.SEEDS[0])] == pytest.approx(
        u["budget"]["u_native_training_gpu_s"][str(eb.SEEDS[0])] / 2434.73)
    receipt = eb.read_json(output / eb.VALIDATION_RECEIPT)
    assert receipt["pass"] and receipt["spot_check"]["record"] == f"{r.VIEW}--seed{eb.SEEDS[0]}"
    text = (output / "findings.md").read_text()
    assert "A4 - U-native" in text and not any(w in text for w in ("beats", "outperforms", "SOTA"))
