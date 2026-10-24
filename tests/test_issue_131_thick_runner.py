"""Issue #131 (D1 THICK): CPU checks of the NovPhy runner's uncertain plumbing (scripts/run_issue_131_thick.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import external_baselines as eb
from scripts import run_issue_131_thick as runner
from world_model.baselines.thick import novphy as tn

@pytest.fixture(autouse=True)
def _one_thread():
    """Tiny models on a shared host: one intra-op thread (torch's default of one per core oversubscribes)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


TINY = {"rssm.hidden": 8, "rssm.deter": 8, "rssm.stoch": 4, "rssm.discrete": 4, "encoder.mlp_layers": (16, 16),
        "decoder.mlp_layers": (16, 16), "reward_head.units": 8, "reward_head.layers": 1, "thick_feat": 16,
        "thick_readouts.units": 8}


def tiny_spec(steps=6, batch=4):
    return {"name": "tiny", "arm": "D1", "size": "native", "configuration": "sparsity-1", "overrides": TINY,
            "steps": steps, "batch": batch}


def tiny_data(n=10, seed=5):
    g = torch.Generator().manual_seed(seed)
    launch = torch.zeros(n, 5)
    launch[:, 0] = -torch.rand(n, generator=g) * 0.3
    launch[:, 1] = torch.rand(n, generator=g) * 0.15
    launch[:, 2], launch[:, 4] = 1.0, 1.0
    return {"carriers": torch.randn(n, len(tn.POSITIONS), tn.CARRIER_DIM, generator=g) * 0.1, "launch5": launch}


def test_matched_width_factor_hits_u_parameters():
    def count(f):
        return tn.parameter_count(runner.build_model({**runner.arm_spec("D1-m", "sparsity-1"),
                                                      "overrides": runner.matched_overrides(f)}))
    n = count(runner.MATCHED_WIDTH_FACTOR)
    assert abs(n - runner.U_PARAMETERS) / runner.U_PARAMETERS <= runner.MATCHED_TOLERANCE
    neighbours = [count(runner.MATCHED_WIDTH_FACTOR + d) for d in (-0.001, 0.001)]
    assert all(abs(m - runner.U_PARAMETERS) >= abs(n - runner.U_PARAMETERS) for m in neighbours)
    native = runner.build_model(runner.arm_spec("D1", "sparsity-1"))
    assert tn.parameter_count(native) == 21_852_337
    matched = runner.build_model(runner.arm_spec("D1-m", "sparsity-1"))
    assert native.config.rssm.stoch == matched.config.rssm.stoch == 32
    assert matched.config.context.context == native.config.context.context == 16


def test_specs_paths_and_queues(tmp_path):
    assert runner.arm_spec("D1", "sparsity-10")["overrides"] == {}
    assert runner.arm_spec("D1", "sparsity-10")["steps"] == 200_000 and runner.arm_spec("D1", "x")["batch"] == 16
    m = runner.arm_spec("D1-m", "sparsity-10")
    assert (m["steps"], m["batch"], m["name"], m["configuration"]) == (9000, 64, "D1-m@sparsity-10", "sparsity-10")
    assert runner.model_dir(tmp_path, "D1", 20260908, "sparsity-10") == tmp_path / "selection/sparsity-10/seed-20260908"
    assert runner.model_dir(tmp_path, "D1-m", 20260908, "sparsity-10") == tmp_path / "models/D1-m/seed-20260908"
    assert sorted(sum(runner.SELECTION_QUEUES.values(), [])) == sorted(runner.CONFIGS)
    plan = {"selected": "sparsity-1", "arms": {a: {"spec": runner.arm_spec(a, "sparsity-1")} for a in runner.ARMS}}
    report_of = runner.training_report_of(tmp_path, plan)
    assert report_of("train-sparsity-1-seed20260908") == runner.selection_dir(tmp_path, "sparsity-1") / "report.json"
    assert report_of("train-sparsity-1-seed20260910") == tmp_path / "models/D1/seed-20260910/report.json"
    assert report_of("train-D1-m@sparsity-1-seed20260909") == tmp_path / "models/D1-m/seed-20260909/report.json"
    assert report_of("train-sparsity-25-seed20260908") == runner.selection_dir(tmp_path, "sparsity-25") / "report.json"
    assert report_of("readouts-D1-seed20260908") is None
    assert [(a, s) for a, s, _, _ in runner.remaining_items(tmp_path, "sparsity-1")] == [
        ("D1", 20260909), ("D1", 20260910), ("D1-m", 20260908), ("D1-m", 20260909), ("D1-m", 20260910)]
    assert len(runner.readout_items()) == 6
    queues, loads = runner.lpt([("a", 5.0), ("b", 4.0), ("c", 3.0), ("d", 2.0), ("e", 1.0)], bins=2)
    assert queues == {0: ["a", "d", "e"], 1: ["b", "c"]} and loads == [8.0, 7.0]


class FakeShots:
    """FitShots stand-in: carriers[shot, position] = shot * 1000 + position in every dimension."""

    def __init__(self, ends):
        self.keys = [f"k{i}" for i in range(len(ends))]
        self.ends = np.asarray(ends)
        self.launch5 = torch.zeros(len(ends), 5)

    def carrier_rows(self, shots, positions):
        values = np.asarray(shots, dtype=np.float32) * 1000 + np.asarray(positions, dtype=np.float32)
        return np.repeat(values[:, None], tn.CARRIER_DIM, axis=1)


def test_fit_store_repeats_the_final_carrier_after_the_end():
    store = runner.fit_store(FakeShots([285, 100, 225]), chunk=2)
    c = store["carriers"][:, :, 0]
    assert c.shape == (3, 46)
    assert c[0].tolist() == [float(p) for p in tn.POSITIONS]
    assert c[1].tolist() == [1000.0 + min(p, 100) for p in tn.POSITIONS]
    assert c[2, -1] == 2000.0 + 225
    batch = runner.episode_batch(store, [2, 0], "cpu")
    assert batch["carrier"].shape == (2, 46, 236) and batch["is_first"][:, 0].all()
    assert torch.equal(batch["carrier"][0], store["carriers"][2])


def test_train_run_resume_is_bitwise_identical(tmp_path, monkeypatch):
    data, spec = tiny_data(), tiny_spec()
    full = runner.train_run(tmp_path / "full", spec, 3, data, "cpu")
    assert full["complete"] and full["steps"] == 6
    original, calls = runner.episode_batch, []

    def stopping(d, rows, device):
        calls.append(1)
        if len(calls) == 4:
            eb.STOP_REQUESTED.append(15)
        return original(d, rows, device)

    monkeypatch.setattr(runner, "episode_batch", stopping)
    with pytest.raises(eb.TrainingPaused):
        runner.train_run(tmp_path / "split", spec, 3, data, "cpu")
    eb.STOP_REQUESTED.clear()
    assert (tmp_path / "split/resume.pt").is_file() and not (tmp_path / "split/model.pt").exists()
    assert torch.load(tmp_path / "split/resume.pt", weights_only=False)["step"] == 4
    monkeypatch.setattr(runner, "episode_batch", original)
    split = runner.train_run(tmp_path / "split", spec, 3, data, "cpu")
    assert split["complete"] and not (tmp_path / "split/resume.pt").exists()
    a = torch.load(tmp_path / "full/model.pt", weights_only=False)["state_dict"]
    b = torch.load(tmp_path / "split/model.pt", weights_only=False)["state_dict"]
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    model, payload = runner.load_trained(tmp_path / "split/model.pt", "cpu")
    assert payload["complete"] and payload["parameters"] == tn.parameter_count(model)
    log = [json.loads(line) for line in (tmp_path / "full/train-log.jsonl").read_text().splitlines()]
    assert log[0]["step"] == 0 and np.isfinite(log[0]["model_loss"]) and "gate_activity" in log[0]


def test_nonfinite_loss_is_a_typed_terminal_failure(tmp_path):
    data = tiny_data()
    data["carriers"][:, 5] = float("nan")
    report = runner.train_run(tmp_path, tiny_spec(), 3, data, "cpu")
    assert not report["complete"] and report["typed_failure"]["typed_failure"] == "nonfinite_loss"
    model, payload = runner.load_trained(tmp_path / "model.pt", "cpu")
    assert model is None and payload["state_dict"] is None
    before = (tmp_path / "model.pt").read_bytes()
    again = runner.train_run(tmp_path, tiny_spec(), 3, tiny_data(), "cpu")  # never retried
    assert again == eb.read_json(tmp_path / "report.json") and again["typed_failure"] == report["typed_failure"]
    assert (tmp_path / "model.pt").read_bytes() == before


def _tiny_model(tmp_path):
    runner.train_run(tmp_path / "m", tiny_spec(steps=2), 3, tiny_data(), "cpu")
    return runner.load_trained(tmp_path / "m/model.pt", "cpu")[0]


def test_predict_rows_and_exploratory_keys(tmp_path):
    model = _tiny_model(tmp_path)
    items, actions = runner.grid_actions()
    z0 = torch.randn(tn.CARRIER_DIM) * 0.1
    out = runner.predict(model, z0[None].expand(len(items), -1), actions, carriers=True, jump=True, batch=5)
    assert out["endpoint"].shape == (16, 236) and out["carriers"].shape == (16, 46, 236)
    assert torch.equal(out["carriers"][:, -1], out["endpoint"]) and len(out["commit_steps"]) == 16
    assert out["hl_carrier"].shape == (16, 236) and out["hl_time"].shape == (16,)
    again = runner.predict(model, z0[None].expand(len(items), -1), actions, jump=True)
    assert torch.allclose(again["endpoint"], out["endpoint"], atol=1e-5)  # deterministic (no noise), any batching
    assert torch.equal(runner.predict(model, z0[None].expand(len(items), -1), actions, batch=5)["endpoint"],
                       out["endpoint"])  # same batching: bitwise (re-scoring under --validate)
    rows = runner.candidate_rows(items, out, z0, runner.objective_costs())
    assert len(rows) == 16
    for row in rows:
        assert {eb.PRIMARY_COST, "coarse_tie_free", "hl_tie_free", "commit_count", "hl_time"} <= set(row)
    matrix = runner.tie_free_matrix(out["endpoint"].reshape(1, 16, -1), z0[None], runner.objective_costs())
    assert matrix.shape == (1, 16)
    assert eb.g_rank(np.repeat(matrix, 2, 0))["levels"] == 2


def test_posterior_features_and_collapse_inputs(tmp_path):
    model = _tiny_model(tmp_path)
    data = tiny_data(n=5)
    z, episodes = runner.posterior_features(model, data, batch=2)
    assert z.shape == (5 * 16, model.rssm.feat_dim) and z.dtype == torch.float64
    assert episodes.tolist() == [i for i in range(5) for _ in range(16)]
    z2, _ = runner.posterior_features(model, data, batch=5)
    assert torch.allclose(z, z2, atol=1e-5)
    assert 0.0 <= eb.within_episode_share(z, episodes) <= 1.0 and eb.effective_rank(z) > 0
    assert runner.COLLAPSE_INDEX == [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45]


def test_commit_statistics():
    stats = runner.commit_statistics(torch.tensor([0, 2, 1]), [[], [1, 45], [3]])
    assert stats["candidates"] == 3 and stats["mean_commits"] == 1.0 and stats["share_with_commit"] == 2 / 3
    hist = stats["commit_step_histogram"]
    assert len(hist) == 45 and hist[0] == 1 and hist[2] == 1 and hist[44] == 1 and sum(hist) == 3


def test_readiness_and_scoring_gates(tmp_path):
    assert not runner.readiness(tmp_path)["pass"] and not runner.gfid_pass(tmp_path)
    assert runner.selection(tmp_path, 0, "cpu") == 1 and not (tmp_path / "selection").exists()
    for name, value in (("gport/gport.json", {"pass": True}), ("gport/agent/gport.json", {"pass": True}),
                        ("fidelity/gfid-timing.json", {"feasible": False})):
        eb.write_json(tmp_path / name, value)
    assert not runner.readiness(tmp_path)["pass"]
    eb.write_json(tmp_path / "fidelity/gfid-timing.json", {"feasible": True})
    assert runner.readiness(tmp_path)["pass"]
    eb.write_json(tmp_path / "fidelity/gfid.json", {"pass": False})
    assert not runner.gfid_pass(tmp_path) and runner.guard_extras(tmp_path) == {"G_port": True, "G_fid": False}
    eb.write_json(tmp_path / "fidelity/gfid.json", {"pass": True})
    assert runner.gfid_pass(tmp_path) and runner.guard_extras(tmp_path) == {"G_port": True, "G_fid": True}


def test_position_rows_read_the_rollout_at_each_time():
    times = eb.POSITION_TIMES
    rolled = torch.zeros(1, len(tn.POSITIONS), tn.CARRIER_DIM)
    for k in range(len(tn.POSITIONS)):
        rolled[0, k, 2 + 5] = k  # slot 0 centre x = frame index
    at = {t: rolled[:, t // tn.STRIDE] for t in times}
    presence = torch.zeros(len(times), 18, dtype=torch.bool)
    presence[:, 0] = True
    cache = {"entries": {"k": {"presence": presence, "centers": torch.zeros(len(times), 18, 2)}}}
    rows = runner._position_rows(at, [("m", {"key": "k"})], cache, times)
    for t in times:
        assert rows["k"][str(t)]["pred"] == pytest.approx((t / 5) ** 2 / 2)


def test_fidelity_projection_source():
    out = Path(runner.OUTPUT)
    if not (out / "fidelity/plan.json").is_file():
        pytest.skip("#131 fidelity plan absent")
    hours, source = runner.fidelity_gpu_hours(out)
    assert hours > 0 and source


def test_training_precision_tf32_on_and_restored_scoring_off(tmp_path, monkeypatch):
    from world_model.planning import scoring_harness as harness
    seen = []
    original = runner.episode_batch

    def spy(d, rows, device):
        seen.append(runner.training_flags())
        return original(d, rows, device)

    monkeypatch.setattr(runner, "episode_batch", spy)
    runner._set_training_flags({"matmul_allow_tf32": False, "cudnn_allow_tf32": False})
    report = runner.train_run(tmp_path / "m", tiny_spec(steps=2), 3, tiny_data(), "cpu")
    assert seen and all(f == {"matmul_allow_tf32": True, "cudnn_allow_tf32": True} for f in seen)
    assert runner.training_flags() == {"matmul_allow_tf32": False, "cudnn_allow_tf32": False}  # restored
    assert report["precision"]["flags"] == {"matmul_allow_tf32": True, "cudnn_allow_tf32": True}
    assert report["precision"]["dtype"] == "float32"
    assert torch.load(tmp_path / "m/model.pt", weights_only=False)["precision"] == report["precision"]
    with harness.deterministic_scoring():
        assert runner.training_flags() == {"matmul_allow_tf32": False, "cudnn_allow_tf32": False}


def _novelty_setup(tmp_path, monkeypatch):
    """Tiny trained models for every arm-seed, frozen-plan / manifest stand-ins, primary records and a fit-split
    stand-in for the novelty cells (3 fit anchors as 'levels', the 16 grid launches)."""
    plan = {"selected": "sparsity-1"}
    for arm, seed in runner.readout_items():
        spec = {**tiny_spec(steps=1), "name": f"{arm}-tiny"}
        runner.train_run(runner.model_dir(tmp_path, arm, seed, "sparsity-1"), spec, seed, tiny_data(), "cpu")
    monkeypatch.setattr(runner, "load_plan", lambda output: plan)
    monkeypatch.setattr(runner, "load_manifest", lambda output: {})
    fit = eb.fit_anchors()["carriers"]
    members = sorted(fit)[:3]
    inventory = [{**item, "key": f"k{item['ordinal']}"} for item in eb.grid_items()]
    cells = {"levels": {m: {"inventory": inventory} for m in members}, "mixed": members}

    def fake_anchors(output, policy, cells_=None):
        payload = {"carriers": {m: fit[m] for m in members}, "read_at": eb.utc_now(), "scoring_policy": policy}
        eb.atomic_torch(eb.novelty_anchors_path(output), payload)
        return payload

    monkeypatch.setattr(eb, "read_novelty_anchors", fake_anchors)
    monkeypatch.setattr(eb, "novelty_reference_records", lambda *a, **k: {})
    return plan, cells, members, fit


def test_score_novelty_refuses_then_scores_guard_passing_arms(tmp_path, monkeypatch):
    plan, cells, members, fit = _novelty_setup(tmp_path, monkeypatch)
    assert runner.score_novelty(tmp_path, "cpu", cells) == 1  # before the primary records
    assert not runner.novelty_dir(tmp_path).exists()
    for arm, seed in runner.readout_items():
        eb.write_json(runner.record_path(tmp_path, arm, seed), {"arm": arm, "seed": seed, "typed_failure": None,
                                                                 "scored_at": "2026-10-25T00:00:00Z"})
    primary = {p: p.read_bytes() for p in (tmp_path / "records").iterdir()}
    passing = {"G_rank": True, "G_action": True, "G_collapse": True, "G_nonfinite": True, "G_trained": True}
    monkeypatch.setattr(runner, "arm_guards", lambda output, records: {
        "D1": passing, "D1-m": {**passing, "G_collapse": False}})
    monkeypatch.setattr(runner, "guard_extras", lambda output: {"G_port": True, "G_fid": True})
    assert runner.score_novelty(tmp_path, "cpu", cells) == 0
    status = eb.read_json(runner.novelty_dir(tmp_path) / "arms.json")["arms"]
    assert status["D1"] == {"guards_pass": True, "failed": []}
    assert status["D1-m"] == {"guards_pass": False, "failed": ["G_collapse"]}
    assert not any(runner.novelty_record_path(tmp_path, "D1-m", s).exists() for s in eb.SEEDS)
    assert {p: p.read_bytes() for p in (tmp_path / "records").iterdir()} == primary  # primary untouched
    costs = runner.objective_costs()
    exploratory = {"coarse_tie_free", "commit_count", "hl_tie_free", "hl_time"}
    for seed in eb.SEEDS:
        record = eb.read_json(runner.novelty_record_path(tmp_path, "D1", seed))
        assert record["schema"] == "issue_131_novelty_record_v1" and (record["arm"], record["seed"]) == ("D1", seed)
        assert record["scoring_policy"] == harness_policy() and sorted(record["inventories"]["grid"]) == members
        model, _ = runner.load_trained(runner.model_dir(tmp_path, "D1", seed, "sparsity-1") / "model.pt", "cpu")
        for m in members:
            items = cells["levels"][m]["inventory"]
            actions = torch.stack([eb.launch5(it["action"]) for it in items])
            from world_model.planning import scoring_harness as harness
            with harness.deterministic_scoring():
                out = runner.predict(model, fit[m][None].expand(len(items), -1), actions, jump=True)
                rows = runner.candidate_rows(items, out, fit[m], costs)  # the primary scoring path's rows
            expected = json.loads(eb.json_text([{k: v for k, v in r.items() if k not in exploratory} for r in rows]))
            assert record["inventories"]["grid"][m]["rows"] == expected
    before = runner.novelty_record_path(tmp_path, "D1", 20260908).read_bytes()
    assert runner.score_novelty(tmp_path, "cpu", cells) == 0  # resumable: existing records are kept
    assert runner.novelty_record_path(tmp_path, "D1", 20260908).read_bytes() == before


def harness_policy():
    from world_model.planning import scoring_harness as harness
    return dict(harness.SCORING_POLICY)


def test_novelty_section_lists_unscored_arms_and_chronology(tmp_path, monkeypatch):
    assert runner.novelty_section(tmp_path, {}) is None
    eb.write_json(runner.novelty_dir(tmp_path) / "arms.json", {"arms": {
        "D1": {"guards_pass": True, "failed": []}, "D1-m": {"guards_pass": False, "failed": ["G_fid"]}}})
    eb.atomic_torch(eb.novelty_anchors_path(tmp_path), {"read_at": "2026-10-26T00:00:00Z"})
    for seed in eb.SEEDS:
        eb.write_json(runner.novelty_record_path(tmp_path, "D1", seed), {"seed": seed})
    monkeypatch.setattr(eb, "novelty_reference_records", lambda *a, **k: {"U": ({}, "tie_free")})
    seen = {}
    monkeypatch.setattr(eb, "novelty_table", lambda views, cells: seen.update(views=sorted(views)) or {"label": "x"})
    records = {"D1": {s: {"scored_at": "2026-10-25T00:00:00Z"} for s in eb.SEEDS}}
    section = runner.novelty_section(tmp_path, records, cells={})
    assert section["scored_arms"] == ["D1"] and section["not_scored"] == {"D1-m": ["G_fid"]}
    assert section["chronology"]["pass"] and seen["views"] == ["D1", "U"]
    assert "novelty/records/*.json" in runner.RECORD_PATTERNS and "novelty/references/*.json" in runner.RECORD_PATTERNS
