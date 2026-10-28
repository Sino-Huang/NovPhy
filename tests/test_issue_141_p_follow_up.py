"""CPU tests of the #141 P-follow-up runner (F1 = frozen U + A2's FiLM classifier; F1-D no-rollout control).

Fit / policy data and the frozen Gate-B U checkpoints only; every evaluation / novelty source is a synthetic or
fit-split stand-in (CUDA_VISIBLE_DEVICES="")."""
from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import external_baselines as eb
from scripts import run_issue_129_slotformer as s129
from scripts import run_issue_141_p_follow_up as runner
from world_model.baselines.phyre_dqn import novphy as nv

SEEDS = eb.SEEDS


@pytest.fixture
def out_dir():
    """An output directory under the repository (records manifests are ROOT-relative)."""
    base = runner.ROOT / ".local-artifacts"
    base.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="test-issue-141-", dir=base))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def tiny_configs(monkeypatch):
    configs = {name: {**c, "updates": 6} for name, c in runner.CONFIGS.items()}
    monkeypatch.setattr(runner, "CONFIGS", configs)
    return configs


def _tiny_data(n=96, positives=12, seed=3, second=None):
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(n, 236, generator=g)
    second = torch.randn(n, 236, generator=g) if second is None else second
    solved = torch.zeros(n, dtype=torch.bool)
    solved[torch.randperm(n, generator=g)[:positives]] = True
    return nv.TrainSet(task_indices=torch.arange(n), is_solved=solved, actions=torch.rand(n, 4, generator=g),
                       observations=torch.cat([z0, second], 1), levels=tuple(f"k{i}" for i in range(n)))


# ---------------------------------------------------------------------------------------------------------------
# constants, classifier, examples
# ---------------------------------------------------------------------------------------------------------------

def test_pre_registered_constants():
    assert list(runner.CONFIGS) == ["lr3e-4", "lr1e-4", "lr1e-3"]
    assert [c["learning_rate"] for c in runner.CONFIGS.values()] == [3e-4, 1e-4, 1e-3]
    assert {c["updates"] for c in runner.CONFIGS.values()} == {9000}
    assert runner.BATCH == 64 and runner.BASE_WIDTH == 70 and runner.CARRIER_DIM == 472
    assert nv.trainable_parameters(runner.build_classifier(1)) == 1_855_301
    assert runner.CUT_POINTS == (0.3246440972222222, 0.4763107638888889)
    assert set(runner.SELECTION_QUEUES.values()) == set(runner.ARMS) == {"F1", "F1-D"}
    assert runner.ARMS["F1"]["role"] == "follow-up" and runner.ARMS["F1-D"]["role"] == "control"


def test_same_seed_gives_identical_initialisation_in_both_arms():
    a, b = runner.build_classifier(20260909), runner.build_classifier(20260909)
    assert runner.state_sha(a.state_dict()) == runner.state_sha(b.state_dict())
    assert runner.state_sha(runner.build_classifier(1).state_dict()) != runner.state_sha(a.state_dict())


def test_fit_observations_concatenate_anchor_and_candidate_feature(monkeypatch):
    anchors = {"a": torch.full((236,), 1.0), "b": torch.full((236,), 2.0)}
    monkeypatch.setattr(eb, "fit_anchors", lambda: {"carriers": anchors, "keys": {}})
    members, keys = ["a", "b", "a"], ["k0", "k1", "k2"]
    ends = torch.arange(3 * 236, dtype=torch.float32).reshape(3, 236)
    f1 = runner.fit_observations("F1", members, keys, {"fit": {"keys": keys, "endpoints": ends}})
    assert f1.shape == (3, 472)
    for i, m in enumerate(members):
        assert torch.equal(f1[i, :236], anchors[m]) and torch.equal(f1[i, 236:], ends[i])
    f1d = runner.fit_observations("F1-D", members, keys)
    for i, m in enumerate(members):
        assert torch.equal(f1d[i, :236], anchors[m]) and torch.equal(f1d[i, 236:], anchors[m])
    with pytest.raises(ValueError):
        runner.fit_observations("F1", members, ["k0", "k2", "k1"], {"fit": {"keys": keys, "endpoints": ends}})


def test_cell_rows_follow_candidate_order_and_read_the_candidate_feature():
    model = runner.build_classifier(5).eval()
    grid = eb.grid_items()
    items = [grid[3], grid[0], grid[7]]
    z0 = torch.randn(236)
    ends = torch.randn(3, 236)
    rows = runner.cell_rows("F1", model, items, z0, ends)
    assert [r["ordinal"] for r in rows] == [it["ordinal"] for it in items]
    logits = runner.classifier_logits(model, z0[None].expand(3, -1), ends,
                                      torch.stack([eb.launch4(it["action"]) for it in items]))
    assert [r["neg_score"] for r in rows] == pytest.approx((-logits).tolist(), abs=0)
    # each candidate's row reads its own feature: permuting candidates and features permutes the rows
    perm = runner.cell_rows("F1", model, items[::-1], z0, ends.flip(0))
    assert [r["neg_score"] for r in perm] == [r["neg_score"] for r in rows[::-1]]
    f1d = runner.cell_rows("F1-D", model, items, z0)
    same = runner.cell_rows("F1", model, items, z0, z0[None].expand(3, -1))
    assert f1d == same
    with pytest.raises(ValueError):
        runner.cell_rows("F1", model, items, z0)


# ---------------------------------------------------------------------------------------------------------------
# training: paired streams, resume, typed failure, classifier-only
# ---------------------------------------------------------------------------------------------------------------

def test_paired_streams_between_arms(out_dir, tiny_configs):
    data = _tiny_data()
    f1 = runner.train_run("F1", "lr3e-4", 11, out_dir / "f1", data)
    f1d = runner.train_run("F1-D", "lr3e-4", 11, out_dir / "f1d", data._replace(
        observations=torch.cat([data.observations[:, :236]] * 2, 1)))
    assert f1["complete"] and f1d["complete"]
    assert f1["init_sha256"] == f1d["init_sha256"]
    assert f1["batch_stream_sha256"] == f1d["batch_stream_sha256"]
    assert f1["classifier_only"]["pass"] and f1["examples"] == len(data.is_solved)
    other = runner.batch_stream_sha(data.is_solved, 12, 6)
    assert other != f1["batch_stream_sha256"]
    batch = runner.batch_indices(data.is_solved, 11, 0)
    assert len(batch) == 64 and int(data.is_solved[batch].sum()) == 32  # 32 + 32, class balanced


def test_train_run_resume_is_bitwise_identical(out_dir, tiny_configs):
    data = _tiny_data(seed=4)
    straight = runner.train_run("F1", "lr1e-3", 7, out_dir / "a", data)
    partial = runner.train_run("F1", "lr1e-3", 7, out_dir / "b", data, max_steps=2)
    assert partial["partial"] and (out_dir / "b" / "resume.pt").is_file()
    resumed = runner.train_run("F1", "lr1e-3", 7, out_dir / "b", data)
    assert resumed["init_sha256"] == straight["init_sha256"]
    sa = torch.load(out_dir / "a" / "model.pt", weights_only=False)["state_dict"]
    sb = torch.load(out_dir / "b" / "model.pt", weights_only=False)["state_dict"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    model, payload = runner.load_model(out_dir / "a" / "model.pt")
    assert model is not None and payload["parameters"] == 1_855_301


def test_nonfinite_loss_is_a_typed_terminal_failure(out_dir, tiny_configs):
    data = _tiny_data()
    data = data._replace(observations=torch.full_like(data.observations, float("nan")))
    report = runner.train_run("F1-D", "lr3e-4", 1, out_dir / "nan", data)
    assert not report["complete"] and report["typed_failure"]["typed_failure"] == "nonfinite_loss"
    model, payload = runner.load_model(out_dir / "nan" / "model.pt")
    assert model is None and payload["state_dict"] is None


def test_classifier_only_detects_foreign_parameters():
    model = runner.build_classifier(2)
    optimizer, _ = nv.make_optimizer(model, 3e-4, 10)
    assert runner.classifier_only(model, optimizer)["pass"]
    optimizer.add_param_group({"params": [torch.nn.Parameter(torch.zeros(3))]})
    assert not runner.classifier_only(model, optimizer)["pass"]


# ---------------------------------------------------------------------------------------------------------------
# strata
# ---------------------------------------------------------------------------------------------------------------

def test_strata_assignment_and_cut_points(out_dir, monkeypatch):
    c1, c2 = runner.CUT_POINTS
    assert runner.stratum_of(c1) == "near" and runner.stratum_of(np.nextafter(c1, 1)) == "mid"
    assert runner.stratum_of(c2) == "mid" and runner.stratum_of(np.nextafter(c2, 2)) == "far"
    assert runner.stratum_of(0.0) == "near" and runner.stratum_of(5.0) == "far"
    assert runner.tertile_cuts([0.0, 1.0, 2.0, 3.0]) == pytest.approx((1.0, 2.0))
    fit = {f"fit-{i}": torch.full((3, 4, 4), 10 * i, dtype=torch.uint8) for i in range(3)}
    monkeypatch.setattr(eb, "fit_anchor_frames", lambda: fit)
    eb.write_json(runner.cut_points_path(out_dir), {"cut_points": [2.0, 6.0]})
    frames = {"m0": torch.full((3, 4, 4), 1, dtype=torch.uint8),  # d = 1 -> near
              "m1": torch.full((3, 4, 4), 14, dtype=torch.uint8),  # d = 4 -> mid
              "m2": torch.full((3, 4, 4), 40, dtype=torch.uint8)}  # d = 20 -> far
    cells = {inv: {"mixed": {m: {} for m in frames}} for inv in eb.INVENTORIES}
    strata = runner.write_strata(out_dir, frames, cells)
    assert {m: v["stratum"] for m, v in strata["levels"].items()} == {"m0": "near", "m1": "mid", "m2": "far"}
    assert strata["levels"]["m1"]["nearest_fit"] == "fit-1" and strata["levels"]["m1"]["distance"] == 4.0
    assert strata["grid_counts"] == {"near": 1, "mid": 1, "far": 1}
    assert runner.write_strata(out_dir, {}, cells) == eb.read_json(runner.strata_path(out_dir))  # written once


def test_frozen_cut_points_file_matches_the_posted_values():
    path = runner.cut_points_path(runner.OUTPUT)
    if not path.is_file():
        pytest.skip("cut points not written in this checkout")
    stored = eb.read_json(path)
    assert tuple(stored["cut_points"]) == runner.CUT_POINTS and stored["levels"] == runner.POLICY_GRID_CELLS
    assert runner.tertile_cuts([v["distance"] for v in stored["distances"].values()]) == runner.CUT_POINTS


# ---------------------------------------------------------------------------------------------------------------
# synthetic evaluation population: contrasts, strata table, publish / validate round trip
# ---------------------------------------------------------------------------------------------------------------

def _cells(members=("m0", "m1", "m2", "m3")):
    items = eb.grid_items()[:4]
    cells = {}
    for inv in eb.INVENTORIES:
        mixed = {}
        for member in members:
            inventory = [{"ordinal": it["ordinal"], "action": it["action"], "key": f"{inv}/{member}/{it['ordinal']}"}
                         for it in items]
            mixed[member] = {"inventory": inventory, "verdicts": {it["ordinal"]: j % 2 == 0 for j, it in enumerate(items)}}
        cells[inv] = {"all": mixed, "mixed": mixed}
    return cells


def _record(cells, key, rng, signal=0.0):
    inventories = {}
    for inv in eb.INVENTORIES:
        inventories[inv] = {}
        for member, entry in cells[inv]["mixed"].items():
            rows = []
            for item in entry["inventory"]:
                value = float(rng.normal() - signal * entry["verdicts"][item["ordinal"]])
                rows.append({"ordinal": item["ordinal"], "excluded": False, "score": -value, key: value})
            inventories[inv][member] = {"rows": rows}
    return {"typed_failure": None, "inventories": inventories}


def _views(cells, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name, key, signal in (("A1", "neg_score", 2.0), ("A2", "neg_score", 1.5), ("U+head", runner.HEAD_KEY, 0.5)):
        out[name] = ({s: _record(cells, key, rng, signal) for s in SEEDS}, key)
    for name in runner.NN_NAMES:
        record = _record(cells, "neg_score", rng, 1.0)
        out[name] = ({s: record for s in SEEDS}, "neg_score")
    return out


def _references(cells, seed=1):
    rng = np.random.default_rng(seed)
    refs = {label: {s: _record(cells, "tie_free", rng, 0.3) for s in SEEDS} for label in ("U", "C*", "J", "J_dec")}
    refs["prior"] = {inv: {} for inv in eb.INVENTORIES}
    return refs


def test_contrast_assembly_and_strata_table():
    cells = _cells()
    rng = np.random.default_rng(5)
    records = {arm: {s: _record(cells, "neg_score", rng, 1.0) for s in SEEDS} for arm in runner.ARMS}
    views = {arm: (records[arm], "neg_score") for arm in runner.ARMS} | _views(cells)
    rows = eb.view_rows(cells, "grid", views, _references(cells))
    within, descriptive = runner.grid_contrasts(rows, True, True, True, {"A1": True, "A2": False})
    assert list(within) == ["F1 - F1-D", "F1 - A1", "F1 - A2", "F1 - U+head"]
    assert within["F1 - F1-D"]["guards_pass"] and within["F1 - A1"]["guards_pass"]
    assert not within["F1 - A2"]["guards_pass"] and within["F1 - A2"]["token"] == eb.STOP_TOKEN
    assert within["F1 - U+head"]["second"] == "U+head" and within["F1 - U+head"]["guards_pass"]
    assert within["F1 - F1-D"]["estimate"]["clusters"] == 4
    assert set(descriptive) == {"F1 - NN-px", "F1 - NN-car"}
    assert all(b["descriptive"] and b["token"] == eb.STOP_TOKEN for b in descriptive.values())
    blocked, _ = runner.grid_contrasts(rows, False, True, True, {"A1": True, "A2": True})
    assert all(b["token"] == eb.STOP_TOKEN for b in blocked.values())
    strata = {"cut_points": [1.0, 2.0], "metric": "m", "levels": {"m0": {"stratum": "near"}, "m1": {"stratum": "near"},
                                                                  "m2": {"stratum": "far"}, "m3": {"stratum": "mid"}}}
    table = runner.strata_table(strata, rows)
    assert {s: b["cells"] for s, b in table["strata"].items()} == {"near": 2, "mid": 1, "far": 1}
    assert set(table["strata"]["near"]["table"]["rankers"]) == set(runner.STRATA_RANKERS)


def _publication_fixture(output, monkeypatch):
    cells = _cells()
    rng = np.random.default_rng(9)
    for arm in runner.ARMS:
        for s in SEEDS:
            record = {**_record(cells, "neg_score", rng, 1.0), "schema": runner.SCHEMA_RECORD, "arm": arm, "seed": s,
                      "scoring_policy": dict(runner.harness.SCORING_POLICY), "scored_at": "2026-10-30T00:00:00Z",
                      "wall_per_decision": 0.01}
            eb.write_json(runner.record_path(output, arm, s), record)
            eb.write_json(runner.readouts_dir(output, arm, s) / "readouts.json",
                          {"policy_grid_auc": 0.9, "G_rank": {"pass": True}, "typed_failure": None,
                           "config_name": "lr3e-4", "macs": {"total": 1.0}, **(
                               {"G_action_inherited": {"value": 0.02}} if arm == "F1" else {})})
    for s in SEEDS:
        eb.write_json(runner.replication_path(output, s), {"pass": True, "inventories": {}, "rule": "r",
                                                           "tolerance": 1e-3, "scoring_policy": {}})
    eb.write_json(runner.strata_path(output), {"cut_points": list(runner.CUT_POINTS), "metric": "m",
                                               "levels": {m: {"stratum": runner.STRATA[i % 3]}
                                                          for i, m in enumerate(cells["grid"]["mixed"])}})
    eb.write_json(output / "smoke.json", {"u": {str(s): {"replication": {"pass": True}} for s in SEEDS}})
    eb.write_json(output / "selection.json", {"arms": {a: {"results": {}} for a in runner.ARMS}})
    plan = {"frozen_at": "2026-10-29T00:00:00Z", "selected": {"F1": "lr3e-4", "F1-D": "lr3e-4"},
            "budget": {"projection": {"remaining_gpu_hours": 1.0}}}
    passing = {"G_rank": True, "G_nonfinite": True, "G_trained": True, "pass": True}
    monkeypatch.setattr(runner, "load_plan", lambda out: plan)
    monkeypatch.setattr(runner, "evaluation_cells", lambda: cells)
    monkeypatch.setattr(runner, "references", lambda: _references(cells))
    monkeypatch.setattr(runner, "imported_views", lambda: _views(cells))
    monkeypatch.setattr(runner, "imported_guards_pass", lambda: {"A1": True, "A2": True})
    monkeypatch.setattr(runner, "a2_inherited", lambda: {"G_port": True, "G_fid": True})
    monkeypatch.setattr(runner, "reference_compute", lambda: {"U": {"trainable_parameters": 1_837_690, "training": {}}})
    monkeypatch.setattr(runner, "arm_guards", lambda out, records, c: {
        "F1": (passing, {"G_u_identity": True, "G_rollout_replication": True, "G_alignment": True,
                         "G_classifier_only": True}),
        "F1-D": (passing, {"G_alignment": True, "G_classifier_only": True})})
    monkeypatch.setattr(runner, "arm_compute", lambda out, plan, arm, records, reference: {
        "trainable_parameters": 1_855_301, "training_gpu_seconds": {str(s): 60.0 for s in SEEDS},
        "macs_per_candidate": 1.0, "wall_per_decision": 0.01})
    monkeypatch.setattr(eb, "_reference_training", lambda name: {str(s): {"training_gpu_seconds": 3600.0} for s in SEEDS})
    monkeypatch.setattr(runner, "global_guards", lambda out, plan, records, c: {
        "G_isolation": {"pass": True}, "G_chronology": {"pass": True}, "G_scoring_policy": {"pass": True},
        "G_records": eb.records_guard(out, runner.IDENTITY)})
    monkeypatch.setattr(runner, "spot_check", lambda out, device=None: {"record": "F1--seed20260908", "pass": True})
    return cells


def test_publish_validate_round_trip(out_dir, monkeypatch):
    _publication_fixture(out_dir, monkeypatch)
    assert runner.publish(out_dir) == 0
    summary = eb.read_json(out_dir / "summary.json")
    assert summary["schema"] == eb.CHILD_SUMMARY_SCHEMA and summary["issue"] == 141
    assert {v: b["role"] for v, b in summary["arms"].items()} == {"F1": "follow-up", "F1-D": "control"}
    assert summary["program_set_members"] == []
    assert summary["global_guards"]["G_records"]["pass"] is False  # before --validate
    assert all(c["token"] == eb.STOP_TOKEN for b in summary["arms"].values() for c in b["contrasts"].values())
    assert all(c["token"] == eb.STOP_TOKEN for c in summary["within_family"].values())
    manifest = eb.read_json(out_dir / "records-manifest.json")
    listed = {Path(e["artifact"]).relative_to(out_dir.relative_to(runner.ROOT)).as_posix() for e in manifest["records"]}
    assert "strata/evaluation-distances.json" in listed and "replication/U--seed20260908.json" in listed
    assert runner.validate(out_dir) == 0
    summary = eb.read_json(out_dir / "summary.json")
    assert summary["global_guards"]["G_records"]["pass"] is True and summary["global_guards_pass"] is True
    assert set(summary["within_family"]) == {"F1 - F1-D", "F1 - A1", "F1 - A2", "F1 - U+head"}
    assert summary["arms"]["F1"]["guards"]["G_rollout_replication"] is True
    assert "G_u_identity" not in summary["arms"]["F1-D"]["guards"]
    extra = summary["extra"]
    assert "G_action" not in summary["arms"]["F1"]["guards"] and extra["scope"]["g_action_inherited"]
    assert set(extra["within_family_descriptive"]) == {"F1 - NN-px", "F1 - NN-car"}
    assert set(extra["strata"]["strata"]) == set(runner.STRATA)
    assert set(extra["inventories"]) == set(eb.INVENTORIES)
    first = {n: (out_dir / n).read_bytes() for n in runner.PUBLISHED}
    runner.write_publication(out_dir)
    assert first == {n: (out_dir / n).read_bytes() for n in runner.PUBLISHED}
    # a changed record after the manifest: validation fails and every token returns to the stop token
    path = runner.record_path(out_dir, "F1", SEEDS[1])
    record = eb.read_json(path)
    record["wall_per_decision"] = 0.02
    eb.write_json(path, record)
    with pytest.raises(ValueError):
        runner.validate(out_dir)
    assert eb.read_json(out_dir / eb.VALIDATION_RECEIPT)["pass"] is False
    summary = eb.read_json(out_dir / "summary.json")
    assert summary["global_guards"]["G_records"]["pass"] is False
    assert all(c["token"] == eb.STOP_TOKEN for c in summary["within_family"].values())


# ---------------------------------------------------------------------------------------------------------------
# U rollout path on the CPU (frozen Gate-B checkpoints; fit-split stand-in cells)
# ---------------------------------------------------------------------------------------------------------------

def _fit_cells(levels=2):
    anchors = eb.fit_anchors()["carriers"]
    members = sorted(anchors)[:levels]
    items = eb.grid_items()
    entry = {"inventory": [{"ordinal": it["ordinal"], "action": it["action"], "key": f"k{it['ordinal']}"} for it in items],
             "verdicts": {it["ordinal"]: it["ordinal"] % 3 == 0 for it in items}}
    return members, {m: anchors[m] for m in members}, entry


def test_u_endpoints_replication_and_primary_rows():
    members, carriers, entry = _fit_cells()
    cells = {inv: {"mixed": {m: entry for m in members}} for inv in eb.INVENTORIES}
    unit = s129.UnitRollout(SEEDS[0], "cpu")
    with runner.harness.deterministic_scoring():
        ends = runner.u_endpoints(unit, carriers, cells)
        launch5 = torch.stack([eb.launch5(it["action"]) for it in entry["inventory"]])
        direct = unit.endpoints(carriers[members[0]][None].expand(16, -1), launch5)
        assert torch.equal(ends["grid"][members[0]], direct)
        costs = s129.objective_costs()
        reference = {"typed_failure": None, "inventories": {
            inv: {m: {"rows": eb.endpoint_rows(entry["inventory"], ends[inv][m], carriers[m], costs)} for m in members}
            for inv in eb.INVENTORIES}}
        rep = runner.replication_block(SEEDS[0], ends, carriers, cells, costs, reference)
        assert rep["pass"] and all(b["exact_rows"] == b["rows"] for b in rep["inventories"].values())
        shifted = {inv: {m: e + 0.01 for m, e in by.items()} for inv, by in ends.items()}
        assert not runner.replication_block(SEEDS[0], shifted, carriers, cells, costs, reference)["pass"]
        model = runner.build_classifier(3).eval()
        inventories, wall = runner.score_arm("F1", SEEDS[0], model, carriers, cells, ends, unit)
        assert wall > 0
        assert inventories["angle"][members[1]]["rows"] == runner.cell_rows("F1", model, entry["inventory"],
                                                                            carriers[members[1]], ends["angle"][members[1]])
        inventories, _ = runner.score_arm("F1-D", SEEDS[0], model, carriers, cells)
        assert inventories["grid"][members[0]]["rows"] == runner.cell_rows("F1-D", model, entry["inventory"],
                                                                           carriers[members[0]])
    assert runner.records_aligned({SEEDS[0]: {"typed_failure": None, "inventories": inventories}}, cells)
    swapped = json.loads(json.dumps(inventories))
    swapped["grid"][members[0]]["rows"].reverse()
    assert not runner.records_aligned({SEEDS[0]: {"typed_failure": None, "inventories": swapped}}, cells)


def test_score_novelty_refuses_before_primary_records(out_dir):
    with pytest.raises(SystemExit):
        runner.score_novelty(out_dir, "cpu")
    eb.write_json(out_dir / "readouts-manifest.json", {"entries": {}, "frozen_at": "x"})
    with pytest.raises(SystemExit):
        runner.score_novelty(out_dir, "cpu")


def test_score_novelty_rows_equal_the_primary_scoring_path(out_dir, monkeypatch, tiny_configs):
    members, carriers, entry = _fit_cells()
    entries = {}
    for arm in runner.ARMS:
        for s in SEEDS:
            path = out_dir / "models" / arm / f"seed-{s}"
            runner.train_run(arm, "lr3e-4", s, path, _tiny_data(seed=s))
            entries[f"{arm}--seed{s}"] = {"model": runner.rel(path / "model.pt"),
                                          "model_sha256": eb.sha256_of(path / "model.pt")}
            eb.write_json(runner.record_path(out_dir, arm, s), {"scored_at": "2026-10-30T00:00:00Z"})
    for s in SEEDS:
        entries[f"U+head--seed{s}"] = {"head": runner.rel(runner.uhead_path(s)),
                                       "head_sha256": eb.sha256_of(runner.uhead_path(s))}
    eb.write_json(out_dir / "readouts-manifest.json", {"entries": entries, "frozen_at": "2026-10-29T00:00:00Z"})
    cells = {"levels": {m: {**entry, "side": "normal"} for m in members}, "mixed": list(members), "in_scope_cells": []}
    monkeypatch.setattr(runner, "load_plan", lambda out: {})
    monkeypatch.setattr(runner, "evaluation_cells", lambda: {})
    passing = {"G_rank": True, "pass": True}
    monkeypatch.setattr(runner, "arm_guards", lambda out, records, c: {a: (passing, {"G_alignment": True})
                                                                        for a in runner.ARMS})
    monkeypatch.setattr(eb, "novelty_cells", lambda: cells)
    monkeypatch.setattr(eb, "read_novelty_anchors", lambda out, policy, c=None: {
        "carriers": carriers, "frames": {}, "read_at": "2026-10-31T00:00:00Z", "scoring_policy": policy})
    monkeypatch.setattr(eb, "novelty_reference_records", lambda *a, **k: {})
    monkeypatch.setattr(runner, "novelty_nn_records", lambda *a, **k: {})
    assert runner.score_novelty(out_dir, "cpu") == 0
    status = eb.read_json(out_dir / "novelty" / "arms.json")["arms"]
    assert all(status[a]["scored"] for a in ("F1", "F1-D", "U+head"))
    costs = s129.objective_costs()
    seed = SEEDS[1]
    unit = s129.UnitRollout(seed, "cpu")
    head, _ = s129.load_head(runner.uhead_path(seed), "cpu")
    launch5 = torch.stack([eb.launch5(it["action"]) for it in entry["inventory"]])
    with runner.harness.deterministic_scoring():
        for arm in (*runner.ARMS, "U+head"):
            record = eb.read_json(runner.novelty_record_path(out_dir, arm, seed))
            assert record["schema"] == runner.SCHEMA_NOVELTY_RECORD and set(record["inventories"]["grid"]) == set(members)
            for m in members:
                if arm == "U+head":
                    expected = s129.cell_rows("U+head", unit.endpoints, head, entry["inventory"], carriers[m], costs)
                else:
                    model, _ = runner.load_model(runner.ROOT / entries[f"{arm}--seed{seed}"]["model"])
                    ends = unit.endpoints(carriers[m][None].expand(16, -1), launch5) if arm == "F1" else None
                    expected = runner.cell_rows(arm, model, entry["inventory"], carriers[m], ends)
                assert record["inventories"]["grid"][m]["rows"] == json.loads(eb.json_text(expected))


def test_report_of_maps_training_phases(out_dir):
    plan = {"selected": {"F1": "lr1e-4", "F1-D": "lr3e-4"}}
    report_of = runner.report_of_factory(out_dir, plan)
    assert report_of("train-F1-D-lr3e-4-seed20260908") == runner.selection_dir(out_dir, "F1-D", "lr3e-4") / "report.json"
    assert report_of("train-F1-lr1e-4-seed20260910") == runner.main_dir(out_dir, "F1", 20260910) / "report.json"
    assert report_of("features-U-seed20260908") is None and report_of("smoke") is None
    assert runner.model_dir(out_dir, "F1", 20260908, plan["selected"]) == runner.selection_dir(out_dir, "F1", "lr1e-4")
    assert runner.train_queues() == {"2": ["F1--seed20260909", "F1--seed20260910"],
                                      "3": ["F1-D--seed20260909", "F1-D--seed20260910"]}
    assert math.isclose(runner.RESERVE_GPU_HOURS, 60.0)
