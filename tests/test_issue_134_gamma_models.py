"""#134 T3 γ-models: CPU tests of the adapter, the runner and the fidelity harness (CUDA_VISIBLE_DEVICES="")."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import subprocess
import textwrap

import numpy as np
import pytest
import torch

from scripts import b127_gamma_models_fidelity as fid
from scripts import external_baselines as eb
from scripts import run_issue_134_gamma_models as runner
from world_model.baselines.gamma_models import novphy as gn

ROOT = Path(__file__).resolve().parents[1]
GAMMA_PY = Path("/srv/home/shua0171/miniconda3/envs/b127-gamma_models/bin/python")
OUTPUT = ROOT / ".local-artifacts/issue-134-gamma-models-v1"


@pytest.fixture(autouse=True)
def _one_thread():
    """Tiny models on a shared host: one intra-op thread."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


TINY = {"hidden_dims": [8], "num_layers": 1, "batch_size": 16, "n_burnin": 2, "n_steps": 6}


def tiny_spec(steps=6, configuration="g0.99"):
    config = gn.config_of(configuration, {**TINY, "n_steps": steps})
    return {"name": "tiny", "arm": "T3", "size": "native", "configuration": configuration, "config": config,
            "steps": steps, "batch": config["batch_size"], "samples": gn.SAMPLES, "sample_seed": gn.SAMPLE_SEED}


def tiny_data(n=6, seed=5):
    g = torch.Generator().manual_seed(seed)
    launch = torch.zeros(n, 5)
    launch[:, 0] = -torch.rand(n, generator=g) * 0.3
    launch[:, 1] = torch.rand(n, generator=g) * 0.15
    launch[:, 2], launch[:, 4] = 1.0, 1.0
    carriers = torch.rand(n, len(gn.POSITIONS), gn.CARRIER_DIM, generator=g)
    carriers[:, :, 118:] = 0.0
    return {"carriers": carriers, "launch5": launch}


def _tiny_model(tmp_path, steps=2, seed=3):
    runner.train_run(tmp_path / f"m{seed}", tiny_spec(steps=steps), seed, tiny_data(), "cpu")
    return runner.load_trained(tmp_path / f"m{seed}/model.pt", "cpu")[0]


class FakeShots:
    """FitShots stand-in: carriers[shot, position] = shot * 1000 + position in every dimension."""

    def __init__(self, ends):
        self.keys = [f"k{i}" for i in range(len(ends))]
        self.ends = np.asarray(ends)
        self.launch5 = torch.zeros(len(ends), 5)

    def carrier_rows(self, shots, positions):
        values = np.asarray(shots, dtype=np.float32) * 1000 + np.asarray(positions, dtype=np.float32)
        return np.repeat(values[:, None], gn.CARRIER_DIM, axis=1)


# ---------------------------------------------------------------------------------------------------------------
# adapter: configuration, γ weighting, transitions
# ---------------------------------------------------------------------------------------------------------------

def test_native_model_is_the_official_notebook_configuration():
    config = gn.config_of("g0.99")
    assert {k: config[k] for k in ("hidden_dims", "sigma", "batch_size", "lr", "decay", "tau", "sample_discount",
                                   "burnin_discount", "n_burnin", "n_steps")} == {
        "hidden_dims": [256, 256, 256], "sigma": 0.1, "batch_size": 1024, "lr": 1e-4, "decay": 1e-5, "tau": 0.005,
        "sample_discount": 0.9, "burnin_discount": 0.5, "n_burnin": 2000, "n_steps": 50000}
    assert (config["K"], config["B"], config["num_layers"], config["delta"]) == (16, 10, 3, False)
    torch.manual_seed(0)
    assert gn.parameter_count(gn.build_model(config)) == 10_230_696
    assert len(gn.CONFIGURATIONS) <= 3 and list(gn.CONFIGURATIONS) == ["g0.95", "g0.978", "g0.99"]
    for name, expected in (("g0.95", 20.0), ("g0.978", 45.0), ("g0.99", 100.0)):
        c = gn.config_of(name)
        assert type(c["discount"]) is float and type(c["sample_discount"]) is float
        assert gn.effective_horizon(c["discount"]) == pytest.approx(expected)
        assert {k: v for k, v in c.items() if k != "discount"} == {k: v for k, v in config.items() if k != "discount"}


@pytest.mark.parametrize("gamma", [0.95, 1 - 1 / 45, 0.99])
def test_occupancy_weights_are_geometric_with_an_absorbing_tail(gamma):
    w = gn.occupancy_weights(gamma)
    assert w.shape == (45,) and w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx(1 - gamma) and w[1] == pytest.approx((1 - gamma) * gamma)
    assert w[10] == pytest.approx((1 - gamma) * gamma ** 10)  # weight (1 - γ) γ^k on transition k
    assert w[-1] == pytest.approx(gamma ** 44)  # frames >= 225: the absorbing carrier


def test_transition_batch_places_the_launch_first_and_noops_after():
    data = tiny_data(n=3)
    shot = np.array([0, 1, 2, 2, 1])
    k = np.array([0, 0, 1, 45, 44])
    cond, nxt = gn.transition_batch(data["carriers"], data["launch5"], shot, k)
    assert torch.equal(cond["a"][0], data["launch5"][0]) and torch.equal(cond["a"][1], data["launch5"][1])
    assert torch.equal(cond["a"][2:], torch.zeros(3, 5)) and torch.equal(nxt["a"], torch.zeros(5, 5))
    assert torch.equal(cond["s"][2], data["carriers"][2, 1]) and torch.equal(nxt["s"][2], data["carriers"][2, 2])
    assert torch.equal(cond["s"][3], data["carriers"][2, 45]) and torch.equal(nxt["s"][3], data["carriers"][2, 45])
    assert torch.equal(nxt["s"][4], data["carriers"][1, 45])
    assert gn.TRANSITIONS_PER_SHOT == 46 and gn.POSITIONS[-1] == 225 and gn.POSITIONS[1] == 5


def test_training_batches_are_uniform_rows_from_the_batch_generator():
    data, spec = tiny_data(n=7), tiny_spec()
    a, b = runner.training_batch(data, 4, 9, spec), runner.training_batch(data, 4, 9, spec)
    assert all(torch.equal(x[k], y[k]) for x, y in zip(a, b) for k in ("s", "a"))
    shot, k = gn.transition_rows(eb.batch_rng(9, 4), 7, 16)
    assert torch.equal(a[0]["s"], data["carriers"][shot, k])
    big_shot, big_k = gn.transition_rows(eb.batch_rng(1, 0), 7, 20000)
    assert big_shot.min() == 0 and big_shot.max() == 6 and big_k.min() == 0 and big_k.max() == 45
    assert abs((big_k == 0).mean() - 1 / 46) < 0.01


def test_fit_store_repeats_the_final_carrier_after_the_end():
    store = runner.fit_store(FakeShots([285, 100, 225]), chunk=2)
    c = store["carriers"][:, :, 0]
    assert c.shape == (3, 46)
    assert c[0].tolist() == [float(p) for p in gn.POSITIONS]
    assert c[1].tolist() == [1000.0 + min(p, 100) for p in gn.POSITIONS]
    assert c[2, -1] == 2000.0 + 225


# ---------------------------------------------------------------------------------------------------------------
# adapter: sampling and the expected cost
# ---------------------------------------------------------------------------------------------------------------

def test_sampling_is_deterministic_batch_invariant_and_equals_official_sample(tmp_path):
    model = _tiny_model(tmp_path)
    data = tiny_data(n=5)
    z0, launch = data["carriers"][:, 0], data["launch5"]
    u = gn.base_noise(8)
    assert torch.equal(u, gn.base_noise(8)) and u.shape == (8, 236)
    x, failed = gn.sample_occupancy(model, z0, launch, u)
    assert x.shape == (5, 8, 236) and not failed.any() and torch.isfinite(x).all()
    y, _ = gn.sample_candidates(model, z0, launch, u, chunk=2)
    assert torch.allclose(x, y, atol=1e-6, rtol=0)  # chunking does not change samples
    single, _ = gn.sample_occupancy(model, z0[3:4], launch[3:4], u)
    assert torch.allclose(single[0], x[3], atol=1e-6, rtol=0)
    torch.manual_seed(gn.SAMPLE_SEED)
    official = model.sample(8, {"s": z0[3:4].repeat(8, 1), "a": launch[3:4].repeat(8, 1)})
    assert torch.allclose(official, single[0], atol=1e-6, rtol=0)  # model.sample after torch.manual_seed(seed)
    other = launch.clone()
    other[:, 0] -= 0.2
    moved, _ = gn.sample_occupancy(model, z0, other, u)
    assert not torch.allclose(moved, x)  # the launch conditions the occupancy
    assert gn.sample_means(x).shape == (5, 236)


def test_sampling_assertion_becomes_a_typed_candidate(tmp_path, monkeypatch):
    model = _tiny_model(tmp_path)
    data = tiny_data(n=4)
    z0, launch = data["carriers"][:, 0], data["launch5"]
    u = gn.base_noise(4)
    clean, _ = gn.sample_occupancy(model, z0, launch, u)
    original = model.flow.transform

    def failing(z, condition):
        if bool((condition["a"][:, 0] == 9.0).any()):
            raise AssertionError("discriminant")
        return original(z, condition)

    monkeypatch.setattr(model.flow, "transform", failing)
    bad = launch.clone()
    bad[2, 0] = 9.0
    x, failed = gn.sample_occupancy(model, z0, bad, u)
    assert failed.tolist() == [False, False, True, False] and torch.isnan(x[2]).all()
    assert torch.allclose(x[[0, 1, 3]], clean[[0, 1, 3]], atol=1e-6, rtol=0)
    items = [{"ordinal": i} for i in range(4)]
    rows = gn.expected_rows(items, x, z0[0], runner.objective_costs(), failed)
    assert rows[2]["excluded"] and rows[2]["typed_failure"] == "sampling_assertion" and rows[2][eb.PRIMARY_COST] is None
    assert not rows[0]["excluded"] and rows[0]["typed_failure"] is None


def test_expected_row_is_the_mean_endpoint_cost_over_samples():
    costs = runner.objective_costs()
    g = torch.Generator().manual_seed(2)
    z0 = torch.rand(236, generator=g)
    samples = torch.rand(6, 236, generator=g)
    row = gn.expected_row(7, samples, z0, costs)
    singles = [costs.row(7, x, z0) for x in samples]
    for key in (eb.PRIMARY_COST, "pig_presence", "pig_displacement"):
        assert row[key] == pytest.approx(float(np.mean([r[key] for r in singles])), abs=1e-12)
    assert row["samples"] == 6 and row["nonfinite_samples"] == 0 and not row["excluded"]
    samples[4, 3] = float("inf")
    bad = gn.expected_row(7, samples, z0, costs)
    assert bad["excluded"] and bad["typed_failure"] == "nonfinite_samples" and bad["nonfinite_samples"] == 1
    assert bad[eb.PRIMARY_COST] is None and bad["pig_presence"] is None
    view = eb.score_rows  # rows feed the same m99 view as EndpointCosts rows (lower cost first)
    assert callable(view)


def test_level_rows_and_fit_readouts(tmp_path, monkeypatch):
    model = _tiny_model(tmp_path)
    original = gn.base_noise
    monkeypatch.setattr(gn, "base_noise", lambda samples=4, seed=gn.SAMPLE_SEED, dim=236: original(4, seed, dim))
    items, _ = runner.grid_actions()
    fit = eb.fit_anchors()["carriers"]
    member = sorted(fit)[0]
    costs = runner.objective_costs()
    rows = runner.level_rows(model, items, fit[member], costs)
    assert [r["ordinal"] for r in rows] == [it["ordinal"] for it in items] and all(r["samples"] == 4 for r in rows)
    assert rows == runner.level_rows(model, items, fit[member], costs)  # deterministic
    members, matrix, means, typed = runner.fit_anchor_readouts(model, costs)
    assert matrix.shape == (480, 16) and means.shape == (480, 16, 236) and members == sorted(fit)
    assert matrix[0].tolist() == pytest.approx([r[eb.PRIMARY_COST] for r in rows])
    assert typed == int(np.isnan(matrix).sum())
    assert eb.g_rank(matrix)["levels"] == 480 and np.isfinite(float(eb.action_sensitivity(means)))


# ---------------------------------------------------------------------------------------------------------------
# runner: specs, training, gates
# ---------------------------------------------------------------------------------------------------------------

def test_specs_paths_and_queues(tmp_path):
    spec = runner.arm_spec("T3", "g0.978")
    assert (spec["steps"], spec["batch"], spec["samples"]) == (50_000, 1024, 64)
    assert spec["config"]["discount"] == pytest.approx(1 - 1 / 45)
    assert runner.model_dir(tmp_path, "T3", 20260908, "g0.95") == tmp_path / "selection/g0.95/seed-20260908"
    assert runner.model_dir(tmp_path, "T3", 20260909, "g0.95") == tmp_path / "models/T3/seed-20260909"
    assert sorted(sum(runner.SELECTION_QUEUES.values(), [])) == sorted(runner.CONFIGS)
    assert [(a, s) for a, s, _, _ in runner.remaining_items(tmp_path, "g0.99")] == [("T3", 20260909), ("T3", 20260910)]
    report_of = runner.training_report_of(tmp_path, {"selected": "g0.99"})
    assert report_of("train-g0.99-seed20260908") == runner.selection_dir(tmp_path, "g0.99") / "report.json"
    assert report_of("train-g0.99-seed20260910") == tmp_path / "models/T3/seed-20260910/report.json"
    assert report_of("readouts-T3-seed20260908") is None
    projection = runner.child_projection({n: 0.05 for n in runner.CONFIGS})
    assert projection["selection_gpu_h"]["g0.95"] == pytest.approx(50_000 * 0.05 / 3600)
    assert projection["training_gpu_h"] == pytest.approx(2 * 50_000 * 0.05 / 3600)
    assert runner.ARM_CHECKS == ["G_rank", "G_action", "G_nonfinite", "G_trained"]


def test_train_run_resume_is_bitwise_identical(tmp_path, monkeypatch):
    data, spec = tiny_data(), tiny_spec()
    full = runner.train_run(tmp_path / "full", spec, 3, data, "cpu")
    assert full["complete"] and full["steps"] == 6
    original, calls = runner.training_batch, []

    def stopping(d, step, seed, s, device="cpu"):
        calls.append(1)
        if len(calls) == 4:
            eb.STOP_REQUESTED.append(15)
        return original(d, step, seed, s, device)

    monkeypatch.setattr(runner, "training_batch", stopping)
    with pytest.raises(eb.TrainingPaused):
        runner.train_run(tmp_path / "split", spec, 3, data, "cpu")
    eb.STOP_REQUESTED.clear()
    assert (tmp_path / "split/resume.pt").is_file() and not (tmp_path / "split/model.pt").exists()
    resume = torch.load(tmp_path / "split/resume.pt", weights_only=False)
    assert resume["step"] == 4 and all(resume["trainer"]["model_actnorm"][0::3])  # ActNorm init flags kept
    monkeypatch.setattr(runner, "training_batch", original)
    split = runner.train_run(tmp_path / "split", spec, 3, data, "cpu")
    assert split["complete"] and not (tmp_path / "split/resume.pt").exists()
    a = torch.load(tmp_path / "full/model.pt", weights_only=False)
    b = torch.load(tmp_path / "split/model.pt", weights_only=False)
    assert a["state_dict"].keys() == b["state_dict"].keys()
    assert all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
    model, payload = runner.load_trained(tmp_path / "split/model.pt", "cpu")
    assert payload["complete"] and payload["parameters"] == gn.parameter_count(model)
    assert all(gn.actnorm_flags(model)[i] for i in range(0, len(model.flow.flows), 3))
    log = [json.loads(line) for line in (tmp_path / "full/train-log.jsonl").read_text().splitlines()]
    assert log[0]["step"] == 0 and np.isfinite(log[0]["loss"])


def test_burnin_discounts_follow_the_notebook():
    trainer = gn.Trainer(gn.config_of("g0.95", TINY), "cpu")
    assert trainer.discounts(0) == (0.5, 0.5) and trainer.discounts(1) == (0.5, 0.5)
    assert trainer.discounts(2) == (0.95, 0.9)
    target = trainer.target
    assert target.prior is trainer.model.prior and target is not trainer.model
    assert all(not p.requires_grad or p.data_ptr() != q.data_ptr()
               for p, q in zip(trainer.model.parameters(), target.parameters()))


def test_nonfinite_loss_is_a_typed_terminal_failure(tmp_path):
    data = tiny_data()
    data["carriers"][:, :, 5] = float("nan")
    report = runner.train_run(tmp_path, tiny_spec(), 3, data, "cpu")
    assert not report["complete"] and report["typed_failure"]["typed_failure"] in ("nonfinite_loss",
                                                                                    "sampling_assertion")
    model, payload = runner.load_trained(tmp_path / "model.pt", "cpu")
    assert model is None and payload["state_dict"] is None
    before = (tmp_path / "model.pt").read_bytes()
    again = runner.train_run(tmp_path, tiny_spec(), 3, tiny_data(), "cpu")  # never retried
    assert again == eb.read_json(tmp_path / "report.json") and (tmp_path / "model.pt").read_bytes() == before


def test_funding_marker_gates_selection_and_training(tmp_path):
    assert not runner.funded(tmp_path) and runner.readiness(tmp_path) == {
        "funded.json": False, "gport/gport.json": False, "fidelity/gfid.json": False, "pass": False}
    assert runner.selection(tmp_path, 0, "cpu") == 1 and not (tmp_path / "selection").exists()
    assert runner.train(tmp_path, 0, "cpu") == 1 and not (tmp_path / "models").exists()
    eb.write_json(tmp_path / "gport/gport.json", {"pass": True})
    eb.write_json(tmp_path / "fidelity/gfid.json", {"pass": True})
    assert runner.selection(tmp_path, 0, "cpu") == 1 and not (tmp_path / "selection").exists()  # still unfunded
    eb.write_json(tmp_path / "funded.json", {"funded": False})
    assert not runner.funded(tmp_path) and runner.train(tmp_path, 0, "cpu") == 1
    eb.write_json(tmp_path / "funded.json", {"funded": True, "decision": "url"})
    assert runner.readiness(tmp_path)["pass"]
    eb.write_json(tmp_path / "fidelity/gfid.json", {"pass": False})
    assert not runner.readiness(tmp_path)["pass"] and runner.guard_extras(tmp_path) == {"G_port": True, "G_fid": False}


def test_smoke_projection_uses_exclusive_equivalent_device_time(tmp_path):
    throughput = {"g0.95": {"seconds_per_step": 0.5, "steps": 50_000, "gpu_seconds": 25_000.0},
                  "g0.978": {"seconds_per_step": 0.3, "steps": 50_000, "gpu_seconds": 15_000.0},
                  "g0.99": {"seconds_per_step": 0.066, "steps": 50_000, "gpu_seconds": 3_300.0}}
    remaining = runner.remaining_items(tmp_path, "g0.95")
    gfid = {"compute": {"gpu_seconds": 0.0, "cpu_seconds_wall": 4689.0}}
    shared = runner.smoke_projection(throughput, remaining, "g0.95", gfid)
    assert shared["basis"] == "process" and not shared["within_caps"]  # shared-GPU wall seconds: 12.0 + 13.9 GPU-h
    assert shared["total_gpu_h"] == pytest.approx((43_300 + 2 * 50_000 * 0.5) / 3600 + 0.3 + 0.5)
    p = runner.smoke_projection(throughput, remaining, "g0.95", gfid, exclusive=0.06)
    assert p["basis"] == "exclusive_equivalent" and p["within_caps"]
    assert p["selection_gpu_h"] == pytest.approx(150_000 * 0.06 / 3600)
    assert p["training_gpu_h"] == pytest.approx(2 * 50_000 * 0.06 / 3600)
    assert p["bases"]["process"] == shared["bases"]["process"]
    assert p["training_queues"] == {0: ["T3--seed20260909"], 1: ["T3--seed20260910"], 2: [], 3: []}
    assert runner.exclusive_rate(tmp_path) is None
    eb.write_json(tmp_path / "timing.json", {"device": "cuda", "t3": {"seconds_per_update_steady_median": 0.061}})
    assert runner.exclusive_rate(tmp_path) == 0.061


def test_timing_probe_needs_no_funding_and_keeps_only_its_record(tmp_path):
    assert not runner.funded(tmp_path)
    assert runner.timing(tmp_path, updates=2, device="cpu", data=tiny_data(), decisions=2) == 0
    record = eb.read_json(tmp_path / "timing.json")
    assert record["schema"] == "issue_134_timing_v1" and record["device"] == "cpu"
    assert record["t3"]["parameters"] == 10_230_696 and record["t3"]["finite"]
    assert sum(c["updates"] for c in record["t3"]["chunks"]) == 2 and record["scoring"]["decisions"] == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ledger", "timing.json"]  # nothing else kept
    gpu_like = {**record, "t3": {**record["t3"], "seconds_per_update_steady_median": 0.05},
                "scoring": {**record["scoring"], "seconds_per_decision_median": 0.1}}
    measured = runner.measured_projection(gpu_like)
    assert measured["child"]["selection_gpu_h"]["g0.99"] == pytest.approx(50_000 * 0.05 / 3600)
    assert measured["child"]["readouts_gpu_h_estimate"] == pytest.approx(3 * 680 * 0.1 / 3600)


def _novelty_setup(tmp_path, monkeypatch):
    plan = {"selected": "g0.99"}
    for arm, seed in runner.readout_items():
        runner.train_run(runner.model_dir(tmp_path, arm, seed, "g0.99"), tiny_spec(steps=1), seed, tiny_data(), "cpu")
    monkeypatch.setattr(runner, "load_plan", lambda output: plan)
    monkeypatch.setattr(runner, "load_manifest", lambda output: {})
    original = gn.base_noise
    monkeypatch.setattr(gn, "base_noise", lambda samples=4, seed=gn.SAMPLE_SEED, dim=236: original(4, seed, dim))
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


def test_score_novelty_refuses_then_scores_with_the_primary_rows(tmp_path, monkeypatch):
    from world_model.planning import scoring_harness as harness
    plan, cells, members, fit = _novelty_setup(tmp_path, monkeypatch)
    assert runner.score_novelty(tmp_path, "cpu", cells) == 1  # before the primary records
    assert not runner.novelty_dir(tmp_path).exists()
    for arm, seed in runner.readout_items():
        eb.write_json(runner.record_path(tmp_path, arm, seed), {"arm": arm, "seed": seed, "typed_failure": None,
                                                                 "scored_at": "2026-10-25T00:00:00Z"})
    primary = {p: p.read_bytes() for p in (tmp_path / "records").iterdir()}
    passing = {"G_rank": True, "G_action": True, "G_nonfinite": True, "G_trained": True}
    monkeypatch.setattr(runner, "arm_guards", lambda output, records: {"T3": passing})
    monkeypatch.setattr(runner, "guard_extras", lambda output: {"G_port": True, "G_fid": True})
    assert runner.score_novelty(tmp_path, "cpu", cells) == 0
    assert eb.read_json(runner.novelty_dir(tmp_path) / "arms.json")["arms"]["T3"] == {"guards_pass": True,
                                                                                       "failed": []}
    assert {p: p.read_bytes() for p in (tmp_path / "records").iterdir()} == primary  # primary untouched
    costs = runner.objective_costs()
    for seed in eb.SEEDS:
        record = eb.read_json(runner.novelty_record_path(tmp_path, "T3", seed))
        assert record["schema"] == "issue_134_novelty_record_v1" and sorted(record["inventories"]["grid"]) == members
        model, _ = runner.load_trained(runner.model_dir(tmp_path, "T3", seed, "g0.99") / "model.pt", "cpu")
        for m in members:
            with harness.deterministic_scoring():
                rows = runner.level_rows(model, cells["levels"][m]["inventory"], fit[m], costs)
            assert record["inventories"]["grid"][m]["rows"] == json.loads(eb.json_text(rows))
    monkeypatch.setattr(runner, "arm_guards", lambda output, records: {"T3": {**passing, "G_action": False}})
    (runner.novelty_dir(tmp_path) / "arms.json").unlink()
    for seed in eb.SEEDS:
        runner.novelty_record_path(tmp_path, "T3", seed).unlink()
    assert runner.score_novelty(tmp_path, "cpu", cells) == 0
    assert eb.read_json(runner.novelty_dir(tmp_path) / "arms.json")["arms"]["T3"]["failed"] == ["G_action"]
    assert not any(runner.novelty_record_path(tmp_path, "T3", s).exists() for s in eb.SEEDS)


def test_score_refuses_without_gfid(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "load_plan", lambda output: {"selected": "g0.99"})
    monkeypatch.setattr(runner, "load_manifest", lambda output: {})
    assert runner.score(tmp_path, "cpu") == 1 and not eb.evaluation_anchors_path(tmp_path).exists()


# ---------------------------------------------------------------------------------------------------------------
# G-port record, fidelity band, fidelity resume
# ---------------------------------------------------------------------------------------------------------------

def test_gport_record_passes_on_the_current_adapter():
    gport = eb.read_json(OUTPUT / "gport/gport.json")
    assert gport["pass"] and gport["same_torch"] and gport["schema"] == "issue_127_gport_v1"
    assert gport["adapter_sha256"] == eb.sha256_of(ROOT / "world_model/baselines/gamma_models/novphy.py")
    assert set(gport["arms"]) == {"pendulum-official", "car-g0.95", "car-g0.978", "car-g0.99"}
    assert all(d["detected"] for d in gport["injected_defects"].values())
    vendor = json.loads((ROOT / "world_model/baselines/gamma_models/vendor/VENDOR.json").read_text())
    for rel, entry in vendor["files"].items():
        assert eb.sha256_of(ROOT / "world_model/baselines/gamma_models/vendor" / rel) == entry["vendored_sha256"]
        if not entry["edits"]:
            assert entry["vendored_sha256"] == entry["upstream_sha256"]


def test_band_frozen_and_recorded_before_any_run():
    band_path = OUTPUT / "fidelity/band.json"
    band = eb.read_json(band_path)
    sha = eb.sha256_of(band_path).split(":")[-1]
    assert sha in (OUTPUT / "decisions.md").read_text()
    assert band["band"] == [0.633, 1.0] and band["paper"]["value"] == pytest.approx(0.7828, abs=1e-4)
    digitised = eb.read_json(OUTPUT / "fidelity/digitised/fig4.json")
    assert digitised["grids"]["41"]["rho_reward_panel_vs_analytic_reward"] > 0.999  # orientation / colour-map check
    assert digitised["grids"]["41"]["rho_analytic_reward_vs_ground_truth"] < band["band"][0]


def test_fidelity_grid_sampling_matches_get_states():
    panel = np.arange(16 * 8, dtype=float).reshape(16, 8)  # rows: +8 .. -8, columns: 0 .. 2π
    s = fid.sample_panel(panel, 5)
    assert s.shape == (5, 5) and s[0, 0] == panel[0, 0] and s[-1, -1] == panel[-1, -1]
    assert s[2, 2] == panel[8, 4]  # (π, 0) -> the pixel containing the centre
    rewards = fid.analytic_rewards(41)
    assert rewards[20, 0] == pytest.approx(0.0) and rewards.argmin() // 41 in (0, 40)


def test_fidelity_train_refuses_without_band(tmp_path, monkeypatch):
    monkeypatch.setattr(fid, "FID", tmp_path / "fidelity")
    with pytest.raises(SystemExit):
        fid.main(["--run"])
    assert not (tmp_path / "fidelity").exists()


@pytest.mark.skipif(not GAMMA_PY.is_file(), reason="b127-gamma_models env missing")
def test_fidelity_training_resume_equals_uninterrupted(tmp_path):
    script = textwrap.dedent(f"""
        import sys, torch
        sys.path.insert(0, {str(ROOT)!r})
        from scripts import b127_gamma_models_fidelity as fid
        class A(fid.Args):
            hidden_dims = [8]
            batch_size = 32
            n_steps = 6
            n_burnin = 2
        assert fid.train({str(tmp_path / 'full')!r}, A) == 0
        assert fid.train({str(tmp_path / 'split')!r}, A, max_steps=3) == 0
        state = torch.load({str(tmp_path / 'split/checkpoint.pt')!r}, weights_only=False)
        assert state["step"] == 3 and all(state["model_actnorm"][0::3]) and all(state["target_actnorm"][0::3])
        assert fid.train({str(tmp_path / 'split')!r}, A) == 0
        a = torch.load({str(tmp_path / 'full/model-final.pt')!r}, weights_only=False)["model"]
        b = torch.load({str(tmp_path / 'split/model-final.pt')!r}, weights_only=False)["model"]
        assert all(torch.equal(a[k], b[k]) for k in a)
        print("ok")
    """)
    out = subprocess.run([str(GAMMA_PY), "-W", "ignore", "-c", script], capture_output=True, text=True,
                         env={"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("ok")


FORBIDDEN = {"read_evaluation_anchors", "evaluation_cells", "position_cache", "read_novelty_anchors", "novelty_cells"}


@pytest.mark.parametrize("path", ["world_model/baselines/gamma_models/novphy.py", "scripts/b127_gamma_models_fidelity.py",
                                  "scripts/b127_gamma_models_port_reference.py"])
def test_no_evaluation_access(path):
    tree = ast.parse((ROOT / path).read_text())
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not names & FORBIDDEN
    assert not any("issue109-evaluation" in s for s in strings)


def test_copy_target_shim_keeps_sampling_usable():
    torch.manual_seed(0)
    model = gn.build_model(gn.config_of("g0.99", TINY))
    target = gn.copy_target(model)
    cond = {"s": torch.rand(3, 236), "a": torch.zeros(3, 5)}
    assert target.sample(3, cond).shape == (3, 236)
    broken = copy.deepcopy(model)
    with pytest.raises(AssertionError):  # the torch 2.13 deepcopy behaviour the shim works around
        broken.sample(3, cond)
