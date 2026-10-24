"""Issue #133 (T1 DreamerV3, coverage arm): CPU checks of the adapter, the runner, the PushT fidelity harness and the
G-port's sensitivity. Run with CUDA_VISIBLE_DEVICES=""."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from scripts import b127_dreamerv3_fidelity as fid
from scripts import external_baselines as eb
from scripts import run_issue_133_dreamerv3 as runner
from world_model.baselines.dreamerv3 import novphy as dn

ROOT = Path(__file__).resolve().parents[1]
DINO_PY = Path("/srv/home/shua0171/miniconda3/envs/b127-dino_wm/bin/python")
GPORT = ROOT / ".local-artifacts/issue-133-dreamerv3-v1/gport"


@pytest.fixture(autouse=True)
def _one_thread():
    """Tiny models on a shared host: one intra-op thread."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


TINY = {"dyn_hidden": 16, "dyn_deter": 16, "dyn_stoch": 4, "dyn_discrete": 4, "units": 16, "encoder.mlp_layers": 1,
        "encoder.mlp_units": 16, "decoder.mlp_layers": 1, "decoder.mlp_units": 16}


def tiny_spec(steps=6, batch=3, length=12, configuration="native"):
    return {"name": "tiny", "arm": "T1", "size": "native", "configuration": configuration, "overrides": TINY,
            "steps": steps, "batch": batch, "length": length}


def tiny_data(n=6, seed=5):
    g = torch.Generator().manual_seed(seed)
    launch = torch.zeros(n, 5)
    launch[:, 0] = -torch.rand(n, generator=g) * 0.3
    launch[:, 1] = torch.rand(n, generator=g) * 0.15
    launch[:, 2], launch[:, 4] = 1.0, 1.0
    return {"carriers": torch.randn(n, len(dn.POSITIONS), dn.CARRIER_DIM, generator=g), "launch5": launch}


def _tiny_model(tmp_path, steps=2):
    runner.train_run(tmp_path / "m", tiny_spec(steps=steps), 3, tiny_data(), "cpu")
    return runner.load_trained(tmp_path / "m/model.pt", "cpu")[0]


# ---------------------------------------------------------------------------------------------------------------
# adapter
# ---------------------------------------------------------------------------------------------------------------

def test_native_model_is_the_official_default_vector_configuration():
    config = dn.build_config()
    assert (config.dyn_deter, config.dyn_hidden, config.dyn_stoch, config.dyn_discrete) == (512, 512, 32, 32)
    assert (config.model_lr, config.opt_eps, config.grad_clip, config.kl_free) == (1e-4, 1e-8, 1000, 1.0)
    assert (config.dyn_scale, config.rep_scale, config.batch_size, config.batch_length) == (0.5, 0.1, 16, 64)
    assert config.encoder["mlp_keys"] == ".*" and config.encoder["cnn_keys"] == "$^"
    assert config.encoder["symlog_inputs"] is True and config.decoder["vector_dist"] == "symlog_mse"
    model = dn.build_model()
    assert dn.parameter_count(model) == 14_671_084 and list(model.heads) == ["decoder"]
    assert model.encoder.mlp_shapes == {"carrier": (236,)} and model.heads["decoder"].mlp_shapes == {"carrier": (236,)}
    assert {id(p) for g in model._model_opt._opt.param_groups for p in g["params"]} == {
        id(p) for p in model.parameters()}  # the rebuilt official optimizer covers exactly the remaining parameters
    assert dn.build_config(dn.CONFIGURATIONS["lr3e-4"]).model_lr == 3e-4
    assert dn.build_config(dn.CONFIGURATIONS["rr128"]).train_ratio == 128
    with pytest.raises(KeyError):
        dn.build_config({"no_such_key": 1})


def test_replay_ratio_budgets():
    assert runner.native_steps("native") == 320_137 == round(13_919 * 46 * 512 / 1024)
    assert runner.native_steps("lr3e-4") == 320_137 and runner.native_steps("rr128") == 80_034
    assert dn.updates_for(2_336_736 / 5, fid.pusht_config()) == 233_674


def test_launch_sits_at_t0_and_noops_follow():
    launch = torch.tensor([[-0.2, 0.1, 1.0, 0.0, 1.0], [-0.1, 0.05, 1.0, 0.0, 1.0]])
    actions = dn.episode_actions(launch)
    assert actions.shape == (2, 46, 5) and dn.LAUNCH_INDEX == 1
    assert torch.equal(actions[:, 1], launch) and actions[:, 0].abs().sum() == 0 and actions[:, 2:].abs().sum() == 0


def test_sample_segments_follow_upstream_sample_episodes():
    rng = np.random.default_rng(3)
    rows = dn.sample_segments(rng, np.full(10, 46), 64, 50)
    for segments in rows:
        assert sum(stop - start for _, start, stop in segments) == 64
        assert 0 <= segments[0][1] <= 44 and segments[0][2] == min(segments[0][1] + 64, 46)
        assert all(start == 0 for _, start, _ in segments[1:])
        assert all(stop == 46 for _, _, stop in segments[1:-1])
    assert dn.sample_segments(np.random.default_rng(3), np.full(10, 46), 64, 50) == rows  # deterministic
    lengths = np.array([1, 100, 3])  # length-weighted choice; an episode shorter than 2 steps is never used
    picks = [e for row in dn.sample_segments(np.random.default_rng(0), lengths, 8, 400) for e, _, _ in row]
    assert 0 not in picks and picks.count(1) > 10 * picks.count(2)


def test_training_batch_layout():
    data = tiny_data(n=3)
    rows = [[(0, 0, 46), (1, 0, 18)], [(2, 1, 46), (0, 0, 19)]]
    batch = dn.training_batch(data["carriers"], data["launch5"], rows)
    assert batch["carrier"].shape == (2, 64, 236) and batch["action"].shape == (2, 64, 5)
    assert batch["is_first"][0].nonzero().flatten().tolist() == [0, 46]
    assert batch["is_first"][1].nonzero().flatten().tolist() == [0, 45]
    full = dn.episode_actions(data["launch5"])
    assert torch.equal(batch["action"][0], torch.cat([full[0], full[1, :18]]))
    assert torch.equal(batch["action"][1], torch.cat([full[2, 1:], full[0, :19]]))
    assert torch.equal(batch["carrier"][1, :45], data["carriers"][2, 1:]) and batch["is_terminal"].sum() == 0


class FakeShots:
    """FitShots stand-in: carriers[shot, position] = shot * 1000 + position in every dimension."""

    def __init__(self, ends):
        self.keys = [f"k{i}" for i in range(len(ends))]
        self.ends = np.asarray(ends)
        self.launch5 = torch.zeros(len(ends), 5)

    def carrier_rows(self, shots, positions):
        values = np.asarray(shots, dtype=np.float32) * 1000 + np.asarray(positions, dtype=np.float32)
        return np.repeat(values[:, None], dn.CARRIER_DIM, axis=1)


def test_fit_store_repeats_the_final_carrier_after_the_end():
    store = runner.fit_store(FakeShots([285, 100, 225]), chunk=2)
    c = store["carriers"][:, :, 0]
    assert c.shape == (3, 46)
    assert c[0].tolist() == [float(p) for p in dn.POSITIONS]
    assert c[1].tolist() == [1000.0 + min(p, 100) for p in dn.POSITIONS]
    assert c[2, -1] == 2000.0 + 225


def test_rollout_length_launch_placement_and_determinism(tmp_path):
    model = _tiny_model(tmp_path)
    data = tiny_data(n=4)
    z0, launch = data["carriers"][:, 0], data["launch5"]
    out = dn.rollout(model, z0, launch, keep=range(46))
    assert sorted(out["carriers"]) == list(range(46)) and out["endpoint"].shape == (4, 236)
    assert torch.equal(out["carriers"][45], out["endpoint"]) and out["feat_end"].shape == (4, 4 * 4 + 16)
    other = launch.clone()
    other[:, 0] -= 0.2
    moved = dn.rollout(model, z0, other, keep=[0, 1])
    assert torch.equal(moved["carriers"][0], out["carriers"][0])  # t0 reconstruction ignores the launch
    assert not torch.equal(moved["carriers"][1], out["carriers"][1])  # the launch enters at the first img_step
    again = dn.rollout(model, z0, launch)
    assert torch.equal(again["endpoint"], out["endpoint"])  # deterministic (latent modes)
    short = dn.rollout(model, z0, launch, steps=3, keep=[3])
    assert torch.equal(short["carriers"][3], out["carriers"][3])
    items, actions = runner.grid_actions()
    pred = runner.predict(model, z0[0][None].expand(16, -1), actions, keep=runner.POSITION_INDEX, batch=5)
    assert pred["endpoint"].shape == (16, 236) and sorted(pred["carriers"]) == [3, 6, 12, 30, 45]
    assert torch.equal(pred["carriers"][45], pred["endpoint"])
    rows = runner.level_rows(model, items, z0[0], runner.objective_costs())
    assert len(rows) == 16 and all(eb.PRIMARY_COST in r for r in rows)


def test_posterior_features_and_collapse_inputs(tmp_path):
    model = _tiny_model(tmp_path)
    data = tiny_data(n=5)
    z, episodes = runner.posterior_features(model, data, batch=2)
    assert z.shape == (5 * 16, 4 * 4 + 16) and z.dtype == torch.float64
    assert episodes.tolist() == [i for i in range(5) for _ in range(16)]
    z2, _ = runner.posterior_features(model, data, batch=5)
    assert torch.allclose(z, z2, atol=1e-6)
    onehot = z[:, :16].reshape(-1, 4, 4)
    assert torch.equal(onehot.sum(-1), torch.ones(onehot.shape[:2], dtype=torch.float64))  # stochastic modes
    assert runner.COLLAPSE_INDEX == list(range(0, 46, 3))


# ---------------------------------------------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------------------------------------------

def test_specs_paths_and_queues(tmp_path):
    spec = runner.arm_spec("T1", "rr128")
    assert (spec["steps"], spec["batch"], spec["length"], spec["configuration"]) == (80_034, 16, 64, "rr128")
    assert runner.model_dir(tmp_path, "T1", 20260908, "rr128") == tmp_path / "selection/rr128/seed-20260908"
    assert runner.model_dir(tmp_path, "T1", 20260909, "rr128") == tmp_path / "models/T1/seed-20260909"
    assert sorted(sum(runner.SELECTION_QUEUES.values(), [])) == sorted(runner.CONFIGS)
    assert [(a, s) for a, s, _, _ in runner.remaining_items(tmp_path, "native")] == [("T1", 20260909), ("T1", 20260910)]
    report_of = runner.training_report_of(tmp_path, {"selected": "native"})
    assert report_of("train-native-seed20260908") == runner.selection_dir(tmp_path, "native") / "report.json"
    assert report_of("train-native-seed20260910") == tmp_path / "models/T1/seed-20260910/report.json"
    assert report_of("train-rr128-seed20260908") == runner.selection_dir(tmp_path, "rr128") / "report.json"
    assert report_of("readouts-T1-seed20260908") is None
    projection = runner.child_projection({n: 0.3 for n in runner.CONFIGS}, 20.0)
    assert projection["selection_gpu_h"]["rr128"] == pytest.approx(80_034 * 0.3 / 3600)
    assert projection["training_gpu_h"] == pytest.approx(2 * 320_137 * 0.3 / 3600)


def test_train_run_resume_is_bitwise_identical(tmp_path, monkeypatch):
    data, spec = tiny_data(), tiny_spec()
    full = runner.train_run(tmp_path / "full", spec, 3, data, "cpu")
    assert full["complete"] and full["steps"] == 6
    original, calls = runner.training_batch, []

    def stopping(d, step, seed, s):
        calls.append(1)
        if len(calls) == 4:
            eb.STOP_REQUESTED.append(15)
        return original(d, step, seed, s)

    monkeypatch.setattr(runner, "training_batch", stopping)
    with pytest.raises(eb.TrainingPaused):
        runner.train_run(tmp_path / "split", spec, 3, data, "cpu")
    eb.STOP_REQUESTED.clear()
    assert (tmp_path / "split/resume.pt").is_file() and not (tmp_path / "split/model.pt").exists()
    assert torch.load(tmp_path / "split/resume.pt", weights_only=False)["step"] == 4
    monkeypatch.setattr(runner, "training_batch", original)
    split = runner.train_run(tmp_path / "split", spec, 3, data, "cpu")
    assert split["complete"] and not (tmp_path / "split/resume.pt").exists()
    a = torch.load(tmp_path / "full/model.pt", weights_only=False)["state_dict"]
    b = torch.load(tmp_path / "split/model.pt", weights_only=False)["state_dict"]
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    model, payload = runner.load_trained(tmp_path / "split/model.pt", "cpu")
    assert payload["complete"] and payload["parameters"] == dn.parameter_count(model)
    log = [json.loads(line) for line in (tmp_path / "full/train-log.jsonl").read_text().splitlines()]
    assert log[0]["step"] == 0 and np.isfinite(log[0]["model_loss"]) and {"carrier_loss", "kl"} <= set(log[0])


def test_nonfinite_loss_is_a_typed_terminal_failure(tmp_path):
    data = tiny_data()
    data["carriers"][:, 5] = float("nan")
    report = runner.train_run(tmp_path, tiny_spec(), 3, data, "cpu")
    assert not report["complete"] and report["typed_failure"]["typed_failure"] == "nonfinite_loss"
    model, payload = runner.load_trained(tmp_path / "model.pt", "cpu")
    assert model is None and payload["state_dict"] is None
    before = (tmp_path / "model.pt").read_bytes()
    again = runner.train_run(tmp_path, tiny_spec(), 3, tiny_data(), "cpu")  # never retried
    assert again == eb.read_json(tmp_path / "report.json") and (tmp_path / "model.pt").read_bytes() == before


def test_timing_probe_needs_no_funding_and_keeps_only_its_record(tmp_path, monkeypatch):
    real = fid.OVERRIDES
    monkeypatch.setattr(fid, "OVERRIDES", FID_TINY)
    assert not runner.funded(tmp_path)
    assert runner.timing(tmp_path, updates=3, cem_steps=2, device="cpu", t1_overrides=TINY, data=tiny_data()) == 0
    record = eb.read_json(tmp_path / "timing.json")
    assert record["schema"] == "issue_133_timing_v1" and record["device"] == "cpu"
    for phase in ("t1", "pusht"):
        block = record[phase]
        assert block["updates"] == 3 and block["finite"] and block["seconds_per_update_steady_median"] > 0
        assert block["host_peak_rss_gib"] > 0 and sum(c["updates"] for c in block["chunks"]) == 3
    assert record["t1"]["batch"] == [16, 64] and record["pusht"]["batch"] == [2, 6]
    assert record["cem_case"]["cem_steps"] == 2 and record["cem_case"]["seconds_per_30_steps"] > 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ledger", "timing.json"]  # nothing else kept
    assert sorted(p.name for p in (tmp_path / "ledger").iterdir()) == ["timing.json"]
    gpu_like = {**record, "t1": {**record["t1"], "seconds_per_update_steady_median": 0.25},
                "pusht": {**record["pusht"], "seconds_per_update_steady_median": 0.3},
                "cem_case": {**record["cem_case"], "seconds_per_30_steps": 3.0}}
    monkeypatch.setattr(fid, "OVERRIDES", real)
    measured = runner.measured_projection(gpu_like)
    assert measured["fidelity"]["train_gpu_h"] == pytest.approx(233_674 * 0.3 / 3600)
    assert measured["fidelity"]["plan_gpu_h"] == pytest.approx(600 * 3.0 / 3600 + 0.25)
    assert measured["child"]["selection_gpu_h"]["native"] == pytest.approx(320_137 * 0.25 / 3600)


def test_funding_marker_gates_selection_training_and_fidelity(tmp_path, monkeypatch):
    assert not runner.funded(tmp_path) and runner.readiness(tmp_path) == {
        "funded.json": False, "gport/gport.json": False, "fidelity/gfid.json": False, "pass": False}
    assert runner.selection(tmp_path, 0, "cpu") == 1 and not (tmp_path / "selection").exists()
    assert runner.train(tmp_path, 0, "cpu") == 1 and not (tmp_path / "models").exists()
    eb.write_json(tmp_path / "gport/gport.json", {"pass": True})
    eb.write_json(tmp_path / "fidelity/gfid.json", {"pass": True})
    assert runner.selection(tmp_path, 0, "cpu") == 1 and not (tmp_path / "selection").exists()  # still unfunded
    eb.write_json(tmp_path / "funded.json", {"funded": False})
    assert not runner.funded(tmp_path) and not fid.funded(tmp_path)
    eb.write_json(tmp_path / "funded.json", {"funded": True, "decision": "url"})
    assert runner.readiness(tmp_path)["pass"] and fid.funded(tmp_path)
    eb.write_json(tmp_path / "fidelity/gfid.json", {"pass": False})
    assert not runner.readiness(tmp_path)["pass"] and runner.guard_extras(tmp_path) == {"G_port": True, "G_fid": False}
    # the fidelity stages: every gated stage refuses without the marker and touches nothing
    monkeypatch.setattr(fid, "OUTPUT", tmp_path / "unfunded")
    (tmp_path / "unfunded").mkdir()
    for stage in ("--build-cache", "--train", "--plan"):
        assert fid.main([stage]) == 1
    assert sorted(p.name for p in (tmp_path / "unfunded").iterdir()) == []


def _novelty_setup(tmp_path, monkeypatch):
    plan = {"selected": "native"}
    for arm, seed in runner.readout_items():
        runner.train_run(runner.model_dir(tmp_path, arm, seed, "native"), {**tiny_spec(steps=1), "name": "tiny"}, seed,
                         tiny_data(), "cpu")
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


def test_score_novelty_refuses_then_scores_with_the_primary_rows(tmp_path, monkeypatch):
    from world_model.planning import scoring_harness as harness
    plan, cells, members, fit = _novelty_setup(tmp_path, monkeypatch)
    assert runner.score_novelty(tmp_path, "cpu", cells) == 1  # before the primary records
    assert not runner.novelty_dir(tmp_path).exists()
    for arm, seed in runner.readout_items():
        eb.write_json(runner.record_path(tmp_path, arm, seed), {"arm": arm, "seed": seed, "typed_failure": None,
                                                                 "scored_at": "2026-10-25T00:00:00Z"})
    primary = {p: p.read_bytes() for p in (tmp_path / "records").iterdir()}
    passing = {"G_rank": True, "G_action": True, "G_collapse": True, "G_nonfinite": True, "G_trained": True}
    monkeypatch.setattr(runner, "arm_guards", lambda output, records: {"T1": passing})
    monkeypatch.setattr(runner, "guard_extras", lambda output: {"G_port": True, "G_fid": True})
    assert runner.score_novelty(tmp_path, "cpu", cells) == 0
    assert eb.read_json(runner.novelty_dir(tmp_path) / "arms.json")["arms"]["T1"] == {"guards_pass": True,
                                                                                       "failed": []}
    assert {p: p.read_bytes() for p in (tmp_path / "records").iterdir()} == primary  # primary untouched
    costs = runner.objective_costs()
    for seed in eb.SEEDS:
        record = eb.read_json(runner.novelty_record_path(tmp_path, "T1", seed))
        assert record["schema"] == "issue_133_novelty_record_v1" and sorted(record["inventories"]["grid"]) == members
        model, _ = runner.load_trained(runner.model_dir(tmp_path, "T1", seed, "native") / "model.pt", "cpu")
        for m in members:
            with harness.deterministic_scoring():
                rows = runner.level_rows(model, cells["levels"][m]["inventory"], fit[m], costs)
            assert record["inventories"]["grid"][m]["rows"] == json.loads(eb.json_text(rows))
    monkeypatch.setattr(runner, "arm_guards", lambda output, records: {"T1": {**passing, "G_collapse": False}})
    (runner.novelty_dir(tmp_path) / "arms.json").unlink()
    for seed in eb.SEEDS:
        runner.novelty_record_path(tmp_path, "T1", seed).unlink()
    assert runner.score_novelty(tmp_path, "cpu", cells) == 0
    assert eb.read_json(runner.novelty_dir(tmp_path) / "arms.json")["arms"]["T1"]["failed"] == ["G_collapse"]
    assert not any(runner.novelty_record_path(tmp_path, "T1", s).exists() for s in eb.SEEDS)


# ---------------------------------------------------------------------------------------------------------------
# PushT fidelity harness
# ---------------------------------------------------------------------------------------------------------------

FID_TINY = {**fid.OVERRIDES, "encoder.cnn_depth": 4, "decoder.cnn_depth": 4, "encoder.mlp_units": 16,
            "decoder.mlp_units": 16, "encoder.mlp_layers": 1, "decoder.mlp_layers": 1, "dyn_hidden": 16,
            "dyn_deter": 16, "dyn_stoch": 4, "dyn_discrete": 4, "units": 16, "batch_size": 2, "batch_length": 6}


def tiny_cache(path, lengths=(23, 17, 9)):
    """A synthetic cache: frame f has every pixel = f % 251, proprio = f, action[f] = (f, -f)."""
    total = sum(lengths)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(int).tolist()
    fid.write_json(fid.cache_paths(path)["index"], {"schema": fid.SCHEMA_CACHE, "lengths": list(lengths),
                                                    "offsets": offsets, "frames": total, "source": "synthetic"})
    arrays = fid.open_cache(path, "w+", total)
    f = np.arange(total)
    arrays["frames"][:] = (f % 251).astype(np.uint8)[:, None, None, None]
    arrays["proprio"][:] = f[:, None]
    arrays["action"][:] = np.stack([f, -f], 1)
    for a in arrays.values():
        a.flush()
    fid.write_json(fid.cache_paths(path)["progress"], {"episodes_done": len(lengths), "complete": True})
    fid.write_json(Path(path) / "MANIFEST.json", {"schema": fid.SCHEMA_CACHE})
    return offsets


def test_phase_episodes_and_pusht_batch_windows(tmp_path):
    offsets = tiny_cache(tmp_path)
    base, count = fid.phase_episodes([23, 17, 9], offsets)
    assert count.sum() == 23 + 17 + 9  # every frame belongs to exactly one phase episode
    assert base[:5].tolist() == [0, 1, 2, 3, 4] and count[:5].tolist() == [5, 5, 5, 4, 4]
    arrays = fid.open_cache(tmp_path)
    rows = [[(0, 0, 5), (6, 0, 1)], [(2, 1, 5), (10, 0, 2)]]  # phase episodes: (src 0, p 0), (src 1, p 1), ...
    batch = fid.pusht_batch(arrays, base, rows)
    assert batch["image"].shape == (2, 6, 64, 64, 3) and batch["action"].shape == (2, 6, 10)
    assert batch["proprio"][0, :, 0].tolist() == [0, 5, 10, 15, 20, 24]  # frames 0..20 step 5, then src 1 phase 1
    assert batch["is_first"][0].tolist() == [1, 0, 0, 0, 0, 1] and batch["is_first"][1].tolist() == [1, 0, 0, 0, 1, 0]
    assert batch["action"][0, 0].abs().sum() == 0 and batch["action"][0, 5].abs().sum() == 0  # step 0: reset
    assert batch["action"][0, 1].tolist() == [0, 0, 1, -1, 2, -2, 3, -3, 4, -4]  # frames 0..4 lead into frame 5
    assert batch["action"][1, 0].tolist() == [2, -2, 3, -3, 4, -4, 5, -5, 6, -6]  # start k=1 of phase 2: frames 2..6
    assert batch["image"][0, 1, 0, 0, 0] == 5


def test_pusht_training_resume_is_bitwise_identical(tmp_path, monkeypatch):
    tiny_cache(tmp_path / "cache")
    monkeypatch.setattr(fid, "OVERRIDES", FID_TINY)
    out = tmp_path / "out"
    assert fid.train(tmp_path / "cache", tmp_path / "full", "cpu", steps=5, output=out) == 0
    original, calls = fid.pusht_batch, []

    def stopping(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            fid.STOP.append(15)
        return original(*args, **kwargs)

    monkeypatch.setattr(fid, "pusht_batch", stopping)
    assert fid.train(tmp_path / "cache", tmp_path / "split", "cpu", steps=5, output=out) == 75
    fid.STOP.clear()
    assert torch.load(tmp_path / "split/resume.pt", weights_only=False)["step"] == 3
    monkeypatch.setattr(fid, "pusht_batch", original)
    assert fid.train(tmp_path / "cache", tmp_path / "split", "cpu", steps=5, output=out) == 0
    a = torch.load(tmp_path / "full/model.pt", weights_only=False)
    b = torch.load(tmp_path / "split/model.pt", weights_only=False)
    assert a["complete"] and a["native_updates"] == dn.updates_for(49 / 5, fid.pusht_config())
    assert all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
    assert not (tmp_path / "split/resume.pt").exists() and (out / "ledger/fidelity-train.json").is_file()
    model, payload = fid.load_pusht_model(tmp_path / "split/model.pt", "cpu")
    assert payload["parameters"] == dn.parameter_count(model)


def test_planning_model_follows_dino_wm_interface(monkeypatch):
    monkeypatch.setattr(fid, "OVERRIDES", FID_TINY)
    torch.manual_seed(0)
    wm = fid.PlanningModel(fid.build_pusht_model("cpu").eval())
    assert wm.decoder is None and next(wm.parameters()).device.type == "cpu"
    obs = {"visual": torch.rand(3, 2, 3, 224, 224) * 2 - 1, "proprio": torch.randn(3, 2, 4)}
    z = wm.encode_obs(obs)
    assert z["visual"].shape == (3, 2, 1, 16) and torch.equal(z["proprio"], torch.zeros(3, 2, 1))
    assert torch.allclose(z["visual"].reshape(3, 2, 4, 4).sum(-1), torch.ones(3, 2, 4))  # categorical probabilities
    first = {k: v[:, :1] for k, v in obs.items()}
    z_obses, zs = wm.rollout(first, torch.randn(3, 5, 10))
    assert z_obses["visual"].shape == (3, 6, 1, 16) and torch.equal(zs, z_obses["visual"])
    assert torch.allclose(z_obses["visual"][:, 0], wm.encode_obs(first)["visual"][:, 0])  # z0 = enc(o0)
    image = fid.to_dreamer_image(torch.full((1, 3, 224, 224), 0.5))
    assert image.shape == (1, 64, 64, 3) and image.dtype == torch.uint8 and int(image[0, 0, 0, 0]) == 128


def _harness(*args, timeout=900):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
    return subprocess.run([str(DINO_PY), str(ROOT / "scripts/b127_dreamerv3_fidelity.py"), *args], env=env,
                          capture_output=True, text=True, timeout=timeout, cwd=ROOT)


@pytest.mark.skipif(not DINO_PY.is_file() or not (ROOT / "data/external/dino_wm/datasets/pusht_noise").is_dir(),
                    reason="b127-dino_wm env or the PushT dataset is absent")
def test_planning_resume_equals_uninterrupted(tmp_path):
    """DINO-WM's planner over the DreamerV3 latent (2 cases, 2 MPC iterations, CPU, untrained model): stop after
    iteration 1 (exit 75) + resume == one run."""
    whole, split = tmp_path / "whole", tmp_path / "split"
    assert _harness("--smoke", "--run-dir", str(whole), "--max-iter", "2").returncode == 0
    first = _harness("--smoke", "--run-dir", str(split), "--max-iter", "2", "--stop-after", "1")
    assert first.returncode == 75 and (split / "mpc_checkpoint.pt").is_file() and not (split / "mpc_final.pt").exists()
    second = _harness("--smoke", "--run-dir", str(split), "--max-iter", "2")
    assert second.returncode == 0 and "resumed the MPC loop after iteration 1" in second.stdout
    assert (whole / "logs.json").read_bytes() == (split / "logs.json").read_bytes()
    a = torch.load(whole / "mpc_final.pt", weights_only=False)
    b = torch.load(split / "mpc_final.pt", weights_only=False)
    assert torch.equal(a["planned_actions"], b["planned_actions"]) and a["iterations"] == b["iterations"] == 2


@pytest.mark.skipif(not DINO_PY.is_file() or not (ROOT / "data/external/dino_wm/datasets/pusht_noise").is_dir(),
                    reason="b127-dino_wm env or the PushT dataset is absent")
def test_plan_cfg_and_cache_in_the_dino_wm_env(tmp_path):
    """plan_cfg == #130's composed hydra config (bar max_iter / saved_folder); a 2-episode cache build resumes."""
    code = f"""
import json, sys, yaml
sys.path.insert(0, 'scripts')
import b127_dreamerv3_fidelity as fid
fid.official_imports()
cfg = fid.plan_cfg('/tmp/x', 12)
ref = yaml.safe_load(open('.local-artifacts/issue-130-dino-wm-v1/fidelity/runs/tier-e-2/.hydra/config.yaml'))
same = {{k: cfg.get(k) == ref.get(k) for k in set(cfg) | set(ref) if k not in ('saved_folder', 'wandb_logging')}}
first = fid.build_cache('{tmp_path}/c', limit=2, stop_after=1)
second = fid.build_cache('{tmp_path}/c', limit=2)
print(json.dumps({{'same': same, 'codes': [first, second], 'complete': fid.cache_complete('{tmp_path}/c')}}))
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
    out = subprocess.run([str(DINO_PY), "-c", code], env=env, capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-2000:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert all(result["same"].values()), result["same"]
    assert result["codes"] == [75, 0] and result["complete"]
    index = fid.read_json(Path(tmp_path) / "c/index.json")
    frames = fid.open_cache(Path(tmp_path) / "c")["frames"]
    assert index["episodes"] == 2 and frames.shape == (index["frames"], 64, 64, 3) and frames[-1].any()


# ---------------------------------------------------------------------------------------------------------------
# G-port sensitivity and isolation
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not (GPORT / "reference-t1-native.pt").is_file(), reason="G-port reference not generated")
def test_gport_passes_and_detects_injected_defects(monkeypatch):
    from scripts import b127_dreamerv3_port_reference as gp
    ref = torch.load(GPORT / "reference-t1-native.pt", map_location="cpu", weights_only=False)
    torch.set_num_threads(4)
    assert gp.compare_arm("t1-native", ref)["pass"]
    monkeypatch.setattr(dn, "LAUNCH_INDEX", 2)  # launch one step late
    out = gp.compare_arm("t1-native", ref)
    assert not out["batch_equal"] and not out["pass"]
    monkeypatch.setattr(dn, "LAUNCH_INDEX", 1)
    monkeypatch.setitem(dn.CONFIGURATIONS, "native", {"kl_free": 0.5})  # free bits changed
    out = gp.compare_arm("t1-native", ref)
    assert out["config_differences"] == ["kl_free"] and not out["pass"]
    monkeypatch.setitem(dn.CONFIGURATIONS, "native", {})
    original = dn.WorldModel.preprocess
    monkeypatch.setattr(dn.WorldModel, "preprocess", lambda self, obs: {  # carriers rescaled inside the adapter
        k: (v * 1.001 if k == "carrier" else v) for k, v in original(self, obs).items()})
    assert not gp.compare_arm("t1-native", ref)["pass"]


FORBIDDEN = {"read_evaluation_anchors", "evaluation_cells", "position_cache", "read_novelty_anchors", "novelty_cells"}


@pytest.mark.parametrize("path", ["world_model/baselines/dreamerv3/novphy.py", "scripts/b127_dreamerv3_fidelity.py",
                                  "scripts/b127_dreamerv3_port_reference.py"])
def test_no_evaluation_access(path):
    tree = ast.parse((ROOT / path).read_text())
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not names & FORBIDDEN
    assert not any("issue109-evaluation" in s for s in strings)
