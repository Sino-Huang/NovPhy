"""CPU tests of #136 T4 Diffusion Forcing (adapter, runner gates, G-port record, fidelity band, projection,
not-run publication). Run with CUDA_VISIBLE_DEVICES=""."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts import b127_dforce_fidelity as fid
from scripts import external_baselines as eb
from scripts import run_issue_136_diffusion_forcing as runner
from world_model.baselines.diffusion_forcing import novphy as dn

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".local-artifacts/issue-136-diffusion-forcing-v1"
VENDOR = ROOT / "world_model/baselines/diffusion_forcing/vendor"


@pytest.fixture(autouse=True)
def _one_thread():
    """Tiny models on a shared host: one intra-op thread."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


TINY_EARLY = {"patience": 2, "epoch_batches": 2, "validation_shots": 2, "samples": 20}


def tiny_spec(name="ts-native", steps=6, early=None):
    config = dn.config_of(name)
    config.update(z_dim=16, batch_size=4, warmup_steps=3, steps=steps, early_stopping=early)
    config["diffusion"].update(network_size=4, timesteps=20, sampling_timesteps=4,
                               num_gru_layers=1 if name == "planning" else 2)
    return {"name": "tiny", "arm": "T4", "size": "native", "configuration": name, "config": config, "steps": steps,
            "batch": 4, "early_stopping": early, "samples": dn.SAMPLES, "sample_seed": dn.SAMPLE_SEED}


def tiny_data(n=6, seed=5):
    g = torch.Generator().manual_seed(seed)
    launch = torch.zeros(n, 5)
    launch[:, 0] = -torch.rand(n, generator=g) * 0.3
    launch[:, 1] = torch.rand(n, generator=g) * 0.15
    launch[:, 2], launch[:, 4] = 1.0, 1.0
    carriers = torch.rand(n, dn.FRAMES, dn.CARRIER_DIM, generator=g)
    carriers[:, :, 118:] = 0.0
    mean, std = dn.normalisation(carriers)
    return {"carriers": carriers, "launch5": launch, "mean": mean, "std": std}


def tiny_model(spec=None, seed=3):
    spec = spec or tiny_spec()
    data = tiny_data()
    torch.manual_seed(seed)
    return dn.build_model(spec["config"], data["mean"], data["std"]).eval(), data


class FakeShots:
    """FitShots stand-in: carriers[shot, position] = shot * 1000 + position in every dimension."""

    def __init__(self, ends):
        self.keys = [f"k{i}" for i in range(len(ends))]
        self.ends = np.asarray(ends)
        self.launch5 = torch.zeros(len(ends), 5)

    def carrier_rows(self, shots, positions):
        values = np.asarray(shots, dtype=np.float32) * 1000 + np.asarray(positions, dtype=np.float32)
        return np.repeat(values[:, None], dn.CARRIER_DIM, axis=1)


# ---------------------------------------------------------------------------------------------------------------
# adapter: configurations, sequences, shapes, causality
# ---------------------------------------------------------------------------------------------------------------

def _yaml(rel):
    return yaml.safe_load((VENDOR / "configurations" / rel).read_text())


def test_configurations_are_the_official_recipes():
    base, pred, exp_p = _yaml("algorithm/df_base.yaml"), _yaml("algorithm/df_prediction.yaml"), _yaml(
        "experiment/exp_prediction.yaml")
    plan, exp_m = _yaml("algorithm/df_planning.yaml"), _yaml("experiment/exp_planning.yaml")
    ts = dn.config_of("ts-native")
    assert ts["diffusion"] == {**base["diffusion"], **pred["diffusion"]}
    assert (ts["z_dim"], ts["lr"], ts["weight_decay"], ts["warmup_steps"]) == (  # YAML 1.1: "3e-4" is a string
        pred["z_shape"][0], float(pred["lr"]), float(pred["weight_decay"]), base["warmup_steps"])
    assert ts["batch_size"] == exp_p["training"]["batch_size"] == 32
    assert ts["gradient_clip_val"] == exp_p["training"]["optim"]["gradient_clip_val"]
    assert ts["steps"] == exp_p["training"]["max_steps"] and ts["precision"] == exp_p["training"]["precision"]
    assert ts["early_stopping"]["patience"] == exp_p["training"]["early_stopping"]["patience"]
    assert list(ts["betas"]) == base["optimizer_beta"] and ts["uncertainty_scale"] == base["uncertainty_scale"]
    assert dn.config_of("ts-100k")["steps"] == 100_000 and dn.config_of("ts-100k")["early_stopping"] is None
    pl = dn.config_of("planning")
    assert pl["diffusion"] == {**base["diffusion"], **plan["diffusion"]}
    assert (pl["z_dim"], pl["warmup_steps"], pl["lr"], pl["weight_decay"]) == (
        plan["z_shape"][0], plan["warmup_steps"], float(exp_m["training"]["lr"]), float(base["weight_decay"]))
    assert pl["steps"] == exp_m["training"]["max_steps"] and pl["batch_size"] == 2048  # paper App. D.10
    assert pl["gradient_clip_val"] is None and pl["early_stopping"] is None
    assert list(dn.CONFIGURATIONS) == ["ts-native", "ts-100k", "planning"] and dn.SAMPLES == 32
    torch.manual_seed(0)
    mean, std = torch.zeros(236), torch.ones(236)
    assert dn.parameter_count(dn.build_model(ts, mean, std)) == 7_367_468
    assert dn.parameter_count(dn.build_model(pl, mean, std)) == 12_034_060


def test_condition_sequence_places_the_launch_out_of_t0_and_noops_after():
    launch = torch.tensor([[-0.2, 0.1, 1.0, 0.0, 1.0], [-0.1, 0.05, 1.0, 0.0, 1.0]])
    c = dn.condition_sequence(launch)
    assert c.shape == (2, 46, 5) and torch.equal(c[:, 1], launch)
    assert c[:, 0].abs().sum() == 0 and c[:, 2:].abs().sum() == 0
    model, data = tiny_model()
    xs, conditions, nonterminals = dn.sequence_batch(data["carriers"], data["launch5"], [0, 1])
    _, cond_t, masks, z = dn.preprocess(model, xs, conditions, nonterminals)
    assert cond_t.shape == (46, 2, 5) and torch.equal(cond_t[1], data["launch5"][:2]) and masks.shape == (46, 2)
    assert z.shape == (2, 16) and z.abs().sum() == 0


def test_fit_store_repeats_the_final_carrier_and_normalises():
    store = runner.fit_store(FakeShots([225, 100, 3]), chunk=2)
    c = store["carriers"]
    assert c.shape == (3, 46, 236)
    assert c[0, :, 0].tolist() == [float(p) for p in dn.POSITIONS]
    assert c[1, 20, 0] == 1100 and c[1, 21, 0] == 1100 and c[1, -1, 0] == 1100
    assert c[2, 0, 0] == 2000 and c[2, 1, 0] == 2003
    flat = c.reshape(-1, 236).double()
    assert torch.allclose(store["mean"], flat.mean(0).float()) and torch.allclose(store["std"], flat.std(0).float())


def test_training_batches_are_uniform_rows_from_the_batch_generator():
    data, spec = tiny_data(), tiny_spec()
    xs, conditions, nonterminals = runner.training_batch(data, 7, 11, spec)
    rows = eb.batch_rng(11, 7).integers(0, 6, size=4)
    assert torch.equal(xs, data["carriers"][rows]) and torch.equal(conditions[:, 1], data["launch5"][rows])
    assert nonterminals.shape == (4, 46) and bool(nonterminals.all())
    again = runner.training_batch(data, 7, 11, spec)
    assert torch.equal(again[0], xs)


def test_training_loss_shapes_and_independent_per_step_noise_levels(monkeypatch):
    model, data = tiny_model()
    batch = dn.sequence_batch(data["carriers"], data["launch5"], [0, 1, 2, 3])
    levels, original = [], model.transition_model.q_sample

    def recording(x_start, t, noise=None):
        levels.append(t.clone())
        return original(x_start=x_start, t=t, noise=noise)

    monkeypatch.setattr(model.transition_model, "q_sample", recording)
    torch.manual_seed(0)
    loss, losses, preds = dn.training_loss(model, *batch)
    assert losses.shape == (46, 4, 236) and preds.shape == (46, 4, 236) and torch.isfinite(loss)
    assert len(levels) == 46 and all(t.shape == (4,) for t in levels)
    grid = torch.stack(levels)  # [frames, batch]: one level per (frame, sequence), drawn independently
    assert len(torch.unique(grid)) > 10 and not all(torch.equal(grid[0], g) for g in grid[1:])
    assert torch.allclose(loss, losses.mean())


def test_training_has_no_future_leakage():
    model, data = tiny_model()
    xs, conditions, nonterminals = dn.sequence_batch(data["carriers"], data["launch5"], [0, 1, 2])
    torch.manual_seed(1)
    _, losses, preds = dn.training_loss(model, xs, conditions, nonterminals)
    k = 12
    changed = xs.clone()
    changed[:, k + 1:] += 5.0
    torch.manual_seed(1)
    _, losses2, preds2 = dn.training_loss(model, changed, conditions, nonterminals)
    assert torch.equal(losses[:k + 1], losses2[:k + 1]) and torch.equal(preds[:k + 1], preds2[:k + 1])
    assert not torch.equal(losses[k + 1], losses2[k + 1])


def _call_frames(model, horizon=dn.HORIZON):
    """Frame of every noise call of sample_paths: context (-1), chunk inits, then the pyramid's stochastic steps."""
    tm = model.transition_model
    steps = tm.sampling_timesteps
    frames = [-1] + list(range(horizon))
    for row in dn.pyramid(horizon, steps, model.config["uncertainty_scale"]):
        frames += [t for t in range(horizon) if min(int(row[t]), steps - 1) < steps - 1]
    return frames


def test_pyramid_sampler_is_causal_and_follows_the_schedule():
    model, data = tiny_model()
    grid = dn.pyramid(45, 4, 1)
    assert grid.shape == (49, 45) and grid[0, 0] == 0 and grid[3, 0] == 3 and grid[4, 0] == 4 and grid[3, 2] == 1
    assert (grid[-1] == 4).all() and (np.diff(grid, axis=1) <= 0).all()  # later frames are always noisier
    frames = _call_frames(model)
    z0, launch = data["carriers"][:3, 0], data["launch5"][:3]

    def noise_fn(perturb_after):
        calls = iter(frames)
        g = torch.Generator().manual_seed(9)

        def draw(shape, device):
            x = torch.randn(shape, generator=g)
            return x + 3.0 if next(calls) > perturb_after else x
        return draw

    base = dn.sample_paths(model, z0, launch, noise_fn(99))
    assert base.shape == (46, 3, 236) and torch.isfinite(base).all()
    k = 20
    moved = dn.sample_paths(model, z0, launch, noise_fn(k))  # randomness of frames > k changed
    assert torch.equal(base[:k + 2], moved[:k + 2])  # path index = frame index + 1 (index 0: context)
    assert not torch.equal(base[k + 2], moved[k + 2])
    other = launch.clone()
    other[:, 0] -= 0.2
    assert not torch.equal(dn.sample_paths(model, z0, other, noise_fn(99))[-1], base[-1])  # the launch conditions


def test_common_noise_is_shared_by_candidates_and_batch_invariant():
    model, data = tiny_model()
    z0 = data["carriers"][:, 0]
    launch = data["launch5"]
    paths = dn.candidate_paths(model, z0, launch, samples=3, keep=(3, dn.HORIZON))
    assert paths[dn.HORIZON].shape == (6, 3, 236) and paths[3].shape == (6, 3, 236)
    alone = dn.candidate_paths(model, z0[4:5], launch[4:5], samples=3)
    assert torch.allclose(alone[dn.HORIZON][0], paths[dn.HORIZON][4], atol=1e-5, rtol=0)
    noise = dn.CommonNoise(3)
    a, b = noise((6, 236), "cpu"), noise((3, 236), "cpu")
    assert torch.equal(a[:3], a[3:]) and not torch.equal(a[:3], b)
    noise.reset()
    assert torch.equal(noise((9, 236), "cpu")[:3], a[:3])


def test_expected_row_is_the_mean_endpoint_cost_over_samples():
    costs = runner.objective_costs()
    g = torch.Generator().manual_seed(2)
    z0 = torch.rand(236, generator=g)
    samples = torch.rand(6, 236, generator=g)
    row = dn.expected_row(7, samples, z0, costs)
    singles = [costs.row(7, x, z0) for x in samples]
    for key in (eb.PRIMARY_COST, "pig_presence", "pig_displacement"):
        assert row[key] == pytest.approx(float(np.mean([r[key] for r in singles])), abs=1e-12)
    assert row["samples"] == 6 and not row["excluded"]
    samples[4, 3] = float("inf")
    bad = dn.expected_row(7, samples, z0, costs)
    assert bad["excluded"] and bad["typed_failure"] == "nonfinite_samples" and bad[eb.PRIMARY_COST] is None


def test_levels_rows_match_single_levels_and_feed_fit_readouts(monkeypatch):
    monkeypatch.setattr(dn, "SAMPLES", 2)
    model, _ = tiny_model()
    costs = runner.objective_costs()
    items, _ = runner.grid_actions()
    fit = eb.fit_anchors()["carriers"]
    members = sorted(fit)[:3]
    batched = runner.levels_rows(model, [(items, fit[m]) for m in members], costs)
    for m, (rows, samples) in zip(members, batched):
        single = runner.level_rows(model, items, fit[m], costs)
        assert [r["ordinal"] for r in rows] == [it["ordinal"] for it in items] and samples.shape == (16, 2, 236)
        assert [r[eb.PRIMARY_COST] for r in rows] == pytest.approx([r[eb.PRIMARY_COST] for r in single], abs=1e-5)
    assert batched[0][0] == runner.levels_rows(model, [(items, fit[members[0]])], costs)[0][0]  # deterministic


# ---------------------------------------------------------------------------------------------------------------
# runner: specs, training, resume, gates
# ---------------------------------------------------------------------------------------------------------------

def test_specs_paths_and_queues(tmp_path):
    spec = runner.arm_spec("T4", "ts-100k")
    assert (spec["steps"], spec["batch"], spec["samples"]) == (100_000, 32, 32)
    assert runner.config_spec("planning")["batch"] == 2048 and runner.config_spec("ts-native")["early_stopping"]
    assert runner.model_dir(tmp_path, "T4", 20260908, "planning") == tmp_path / "selection/planning/seed-20260908"
    assert runner.model_dir(tmp_path, "T4", 20260909, "planning") == tmp_path / "models/T4/seed-20260909"
    assert sorted(sum(runner.SELECTION_QUEUES.values(), [])) == sorted(runner.CONFIGS)
    assert [(a, s) for a, s, _, _ in runner.remaining_items(tmp_path, "ts-native")] == [
        ("T4", 20260909), ("T4", 20260910)]
    report_of = runner.training_report_of(tmp_path, {"selected": "ts-native"})
    assert report_of("train-ts-native-seed20260908") == runner.selection_dir(tmp_path, "ts-native") / "report.json"
    assert report_of("train-ts-native-seed20260910") == tmp_path / "models/T4/seed-20260910/report.json"
    assert report_of("readouts-T4-seed20260908") is None
    assert runner.ARM_CHECKS == ["G_rank", "G_action", "G_nonfinite", "G_trained"]
    assert runner.POSITION_INDEX == [3, 6, 12, 30, 45]


def test_trainer_warmup_follows_optimizer_step(tmp_path):
    spec = tiny_spec()
    data = tiny_data()
    torch.manual_seed(0)
    trainer = dn.Trainer(spec["config"], data["mean"], data["std"], "cpu")
    lrs = [trainer.train_step(*runner.training_batch(data, b, 0, spec))["lr"] for b in range(5)]
    lr = spec["config"]["lr"]
    assert lrs == pytest.approx([lr, lr / 3, 2 * lr / 3, lr, lr])  # step 0 at the full LR (warm-up set after it)


def test_train_run_resume_is_bitwise_identical_with_early_stopping_state(tmp_path, monkeypatch):
    data, spec = tiny_data(), tiny_spec(steps=8, early=TINY_EARLY)
    full = runner.train_run(tmp_path / "full", spec, 3, data, "cpu")
    assert full["complete"] and full["early_stopping"]["validations"] >= 2
    original, calls = runner.training_batch, []

    def stopping(d, step, seed, s, device="cpu"):
        calls.append(1)
        if len(calls) == 3:
            eb.STOP_REQUESTED.append(15)
        return original(d, step, seed, s, device)

    monkeypatch.setattr(runner, "training_batch", stopping)
    with pytest.raises(eb.TrainingPaused):
        runner.train_run(tmp_path / "split", spec, 3, data, "cpu")
    eb.STOP_REQUESTED.clear()
    resume = torch.load(tmp_path / "split/resume.pt", weights_only=False)
    assert resume["step"] == 3 and resume["early"]["history"] and not (tmp_path / "split/model.pt").exists()
    monkeypatch.setattr(runner, "training_batch", original)
    split = runner.train_run(tmp_path / "split", spec, 3, data, "cpu")
    assert split["complete"] and not (tmp_path / "split/resume.pt").exists()
    a = torch.load(tmp_path / "full/model.pt", weights_only=False)
    b = torch.load(tmp_path / "split/model.pt", weights_only=False)
    assert all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
    assert a["early_stopping"]["history"] == b["early_stopping"]["history"]
    best = torch.load(tmp_path / "full/best.pt", weights_only=False)
    assert best["step"] == a["early_stopping"]["best_step"]
    assert all(torch.equal(best["model"][k], a["state_dict"][k]) for k in best["model"])  # best checkpoint kept
    model, payload = runner.load_trained(tmp_path / "split/model.pt", "cpu")
    assert payload["complete"] and payload["parameters"] == dn.parameter_count(model)


def test_early_stopping_rule_is_lightnings():
    state = {"best": None, "wait": 0, "best_step": None, "history": []}
    assert dn.early_stop_update(state, 1.0, 2) == (True, False)
    assert dn.early_stop_update(state, 1.0, 2) == (False, False)  # equal is not an improvement
    assert dn.early_stop_update(state, 0.5, 2) == (True, False) and state["wait"] == 0
    assert dn.early_stop_update(state, 0.7, 2) == (False, False)
    assert dn.early_stop_update(state, 0.6, 2) == (False, True)  # patience exhausted
    assert dn.early_stop_update({"best": None, "wait": 0}, float("nan"), 6) == (False, True)  # check_finite


def test_crps_quantile_sum_is_the_quantile_loss():
    g = torch.Generator().manual_seed(0)
    pred, truth = torch.rand(100, 5, 3, 4, generator=g), torch.rand(5, 3, 4, generator=g)
    value = float(dn.crps_quantile_sum(pred, truth))
    p, y = pred.sum(-1).numpy(), truth.sum(-1).numpy()
    qs = (np.arange(20) / 20.0)[1:]
    total = 0.0
    for q in qs:
        pq = np.sort(p, axis=0)[int(q * 100) - 1]
        total += np.sum(2 * np.abs((pq - y) * ((y <= pq) - q))) / len(qs)
    assert value == pytest.approx(total / np.abs(y).sum(), rel=1e-5)


def test_nonfinite_loss_is_a_typed_terminal_failure(tmp_path):
    data = tiny_data()
    data["carriers"][:, :, 5] = float("nan")
    report = runner.train_run(tmp_path, tiny_spec(), 3, data, "cpu")
    assert not report["complete"] and report["typed_failure"]["typed_failure"] == "nonfinite_loss"
    model, payload = runner.load_trained(tmp_path / "model.pt", "cpu")
    assert model is None and payload["state_dict"] is None
    before = (tmp_path / "model.pt").read_bytes()
    again = runner.train_run(tmp_path, tiny_spec(), 3, tiny_data(), "cpu")  # never retried
    assert again == eb.read_json(tmp_path / "report.json") and (tmp_path / "model.pt").read_bytes() == before


def test_funding_marker_gport_and_gfid_gate_selection_training_and_scoring(tmp_path, monkeypatch):
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
    assert runner.selection(tmp_path, 0, "cpu") == 1 and not (tmp_path / "selection").exists()
    monkeypatch.setattr(runner, "load_plan", lambda output: {"selected": "ts-native"})
    monkeypatch.setattr(runner, "load_manifest", lambda output: {})
    assert runner.score(tmp_path, "cpu") == 1 and not eb.evaluation_anchors_path(tmp_path).exists()


def test_timing_probe_needs_no_funding_and_keeps_only_its_record(tmp_path, monkeypatch):
    monkeypatch.setattr(dn, "SAMPLES", 2)
    specs = {"ts-native": tiny_spec(early=TINY_EARLY), "planning": tiny_spec("planning")}
    monkeypatch.setattr(runner, "config_spec", lambda name: specs[name])
    assert not runner.funded(tmp_path)
    assert runner.timing(tmp_path, updates=2, planning_updates=2, device="cpu", data=tiny_data(), decisions=2,
                         gfid_updates=2, gfid_rows=16) == 0
    record = eb.read_json(tmp_path / "timing.json")
    assert record["schema"] == "issue_136_timing_v1" and record["device"] == "cpu"
    assert set(record["configs"]) == {"ts-native", "planning"}
    assert all(c["finite"] and sum(ch["updates"] for ch in c["chunks"]) == 2 for c in record["configs"].values())
    assert record["configs"]["ts-native"]["validation"]["seconds"] > 0
    assert all(c["scoring"]["decisions"] == 2 for c in record["configs"].values())
    gfid = record["gfid"]
    assert gfid["rows"] == 16 and gfid["seconds_per_sampler_call"] > 0 and gfid["seconds_per_update_steady_median"] > 0
    assert gfid["shape"]["frames"] == 60 and gfid["parameters"] == 6_667_784  # + the 8-weight series embedding
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ledger", "timing.json"]  # nothing else kept


# ---------------------------------------------------------------------------------------------------------------
# projection math
# ---------------------------------------------------------------------------------------------------------------

def test_child_projection_math():
    rates = {"ts-native": 0.02, "ts-100k": 0.02, "planning": 0.1}
    p = runner.child_projection(rates, 1.0, 0.5, 200, 2.0)
    assert p["selection_gpu_h"]["ts-native"] == pytest.approx((200_000 * 0.02 + 2000 * 2.0) / 3600)
    assert p["selection_gpu_h"]["ts-100k"] == pytest.approx(100_000 * 0.02 / 3600)
    assert p["selection_gpu_h"]["planning"] == pytest.approx(120_005 * 0.1 / 3600)
    per_seed = 120_005 * 0.1  # the most expensive configuration
    assert p["native_per_seed_gpu_h"] == pytest.approx(per_seed / 3600)
    assert p["training_gpu_h"] == pytest.approx(2 * per_seed / 3600)
    counts = p["decision_counts"]
    assert counts == {"readouts_per_arm_seed": 680, "selection_readouts": 600, "score_per_seed": 512 + 480 + 311 + 464}
    assert p["readouts_gpu_h"] == pytest.approx(3 * 680 * 0.5 / 3600)
    assert p["score_gpu_h"] == pytest.approx(3 * counts["score_per_seed"] * 0.5 / 3600)
    total = 3600 + 8000 + 2000 + per_seed + 2 * per_seed + 300 + 1020 + 3 * counts["score_per_seed"] * 0.5
    assert p["total_gpu_h"] == pytest.approx(total / 3600)
    wall = 3600 + per_seed + 300 + per_seed + 1020 / 4 + 3 * counts["score_per_seed"] * 0.5
    assert p["wall_h_on_4_gpus_serial_ladder"] == pytest.approx(wall / 3600)
    assert p["training_queue_loads_h"][:2] == pytest.approx([per_seed / 3600] * 2)
    assert p["within_caps"] and not runner.child_projection({k: 1.0 for k in rates}, 1.0, 0.5, 200, 2.0)["within_caps"]
    assert runner.fidelity_seconds(0.02, 2.0) == pytest.approx(200_000 * 0.02 + 2001 * 2.0)
    assert runner.GFID["calls"] == 30 + 30 * 51 and runner.GFID["rows"] == 5 * 100 * 100


def _fake_cpu():
    m = {"cpu_seconds_per_update": 0.4, "batch": 32, "update_ops": 10_000, "update_flops": 6e10,
         "parameters": 7_367_468, "sampler_call_ops": 60, "sampler_call_flops_per_row": 1.5e7}
    return {"ts-native": m, "ts-100k": m, "planning": {**m, "update_ops": 20_000, "update_flops": 7e12,
                                                       "sampler_call_flops_per_row": 2.4e7}}


def test_project_writes_inference_range_then_prefers_the_measured_probe(tmp_path):
    assert runner.project(tmp_path, policy_cells=200, cpu=_fake_cpu()) == 0
    p = eb.read_json(tmp_path / "projection.json")
    assert p["primary"] == "projection_upper" and "[INFERENCE]" in p["label"]
    lo, hi = p["projection_lower"], p["projection_upper"]
    assert lo["total_gpu_h"] < hi["total_gpu_h"]
    up = runner.GPU_MODEL["upper"]
    assert hi["gpu_seconds_per_update"]["planning"] == pytest.approx(
        20_000 * up["launch_us"] * 1e-6 + 7e12 / (up["train_tflops"] * 1e12))
    fid = hi["fidelity_rates"]
    assert fid["update"] == pytest.approx(hi["gpu_seconds_per_update"]["ts-native"] * 60 / 46)
    assert fid["validation"] == pytest.approx(1560 * (60 * up["launch_us"] * 1e-6 + 1.5e7 * 50_000 / 40e12))
    assert hi["fidelity_gpu_h"] == pytest.approx(runner.fidelity_seconds(fid["update"], fid["validation"]) / 3600)
    timing = {"device": "cuda", "configs": {
        "ts-native": {"seconds_per_update_steady_median": 0.03, "validation": {"seconds": 1.5},
                      "scoring": {"seconds_per_decision": 0.4}},
        "planning": {"seconds_per_update_steady_median": 0.08, "scoring": {"seconds_per_decision": 0.6}}},
        "gfid": {"seconds_per_update_steady_median": 0.04, "seconds_per_sampler_call": 0.001, "rows": 10_000}}
    eb.write_json(tmp_path / "timing.json", timing)
    runner.project(tmp_path, policy_cells=200, cpu=_fake_cpu())
    p = eb.read_json(tmp_path / "projection.json")
    assert p["primary"] == "projection_measured"
    m = p["projection_measured"]
    assert m["gpu_seconds_per_update"] == {"ts-native": 0.03, "ts-100k": 0.03, "planning": 0.08}
    assert m["seconds_per_decision"] == 0.6 and m["validation_seconds"] == 1.5
    assert m["fidelity_rates"] == {"update": 0.04, "validation": pytest.approx(0.001 * 5 * 1560)}
    expected = runner.child_projection(m["gpu_seconds_per_update"],
                                       runner.fidelity_seconds(0.04, m["fidelity_rates"]["validation"]) / 3600, 0.6,
                                       200, 1.5)
    assert m["child"] == json.loads(json.dumps(expected))


# ---------------------------------------------------------------------------------------------------------------
# not-run publication, novelty mode
# ---------------------------------------------------------------------------------------------------------------

def test_publish_not_run_lists_the_typed_reason(tmp_path):
    with pytest.raises(SystemExit):
        runner.publish(tmp_path)  # no plan.json and no typed reason
    eb.write_json(tmp_path / "gport/gport.json", {"pass": True})
    eb.write_json(tmp_path / "projection.json", {"primary": "projection_measured",
                                                 "projection_measured": {"child": {"total_gpu_h": 150.0}}})
    eb.write_json(tmp_path / "not-run.json", {"typed_reason": "cap", "decision": "url"})
    assert runner.publish(tmp_path) == 0
    summary = eb.read_json(tmp_path / "summary.json")
    assert summary["arms"] == {} and summary["extra"]["pending_arms"] == {"T4": "cap"}
    assert summary["extra"]["G_port"] is True and summary["gfid"]["T4"]["pass"] is False
    assert summary["extra"]["projection"]["total_gpu_h"] == 150.0 and summary["frozen_at"] is None
    assert "`cap`" in (tmp_path / "findings.md").read_text()


def _novelty_setup(tmp_path, monkeypatch):
    plan = {"selected": "ts-native"}
    for arm, seed in runner.readout_items():
        runner.train_run(runner.model_dir(tmp_path, arm, seed, "ts-native"), tiny_spec(steps=1), seed, tiny_data(),
                         "cpu")
    monkeypatch.setattr(runner, "load_plan", lambda output: plan)
    monkeypatch.setattr(runner, "load_manifest", lambda output: {})
    monkeypatch.setattr(dn, "SAMPLES", 2)
    fit = eb.fit_anchors()["carriers"]
    members = sorted(fit)[:2]
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
                                                                 "scored_at": "2026-10-26T00:00:00Z"})
    primary = {p: p.read_bytes() for p in (tmp_path / "records").iterdir()}
    passing = {"G_rank": True, "G_action": True, "G_nonfinite": True, "G_trained": True}
    monkeypatch.setattr(runner, "arm_guards", lambda output, records: {"T4": passing})
    monkeypatch.setattr(runner, "guard_extras", lambda output: {"G_port": True, "G_fid": True})
    assert runner.score_novelty(tmp_path, "cpu", cells) == 0
    assert eb.read_json(runner.novelty_dir(tmp_path) / "arms.json")["arms"]["T4"] == {"guards_pass": True,
                                                                                       "failed": []}
    assert {p: p.read_bytes() for p in (tmp_path / "records").iterdir()} == primary  # primary untouched
    costs = runner.objective_costs()
    seed = eb.SEEDS[0]
    record = eb.read_json(runner.novelty_record_path(tmp_path, "T4", seed))
    assert record["schema"] == "issue_136_novelty_record_v1" and sorted(record["inventories"]["grid"]) == members
    model, _ = runner.load_trained(runner.model_dir(tmp_path, "T4", seed, "ts-native") / "model.pt", "cpu")
    with harness.deterministic_scoring():
        levels = runner.levels_rows(model, [(cells["levels"][m]["inventory"], fit[m]) for m in members], costs)
    for m, (rows, _) in zip(members, levels):
        assert record["inventories"]["grid"][m]["rows"] == json.loads(eb.json_text(rows))
    monkeypatch.setattr(runner, "arm_guards", lambda output, records: {"T4": {**passing, "G_action": False}})
    (runner.novelty_dir(tmp_path) / "arms.json").unlink()
    for s in eb.SEEDS:
        runner.novelty_record_path(tmp_path, "T4", s).unlink()
    assert runner.score_novelty(tmp_path, "cpu", cells) == 0
    assert eb.read_json(runner.novelty_dir(tmp_path) / "arms.json")["arms"]["T4"]["failed"] == ["G_action"]
    assert not any(runner.novelty_record_path(tmp_path, "T4", s).exists() for s in eb.SEEDS)


# ---------------------------------------------------------------------------------------------------------------
# G-port record, vendoring, fidelity band and runner
# ---------------------------------------------------------------------------------------------------------------

def test_gport_record_passes_exactly_on_the_current_adapter():
    gport = eb.read_json(OUTPUT / "gport/gport.json")
    assert gport["pass"] and gport["exact"] and gport["same_torch"] and gport["schema"] == "issue_127_gport_v1"
    assert gport["adapter_sha256"] == eb.sha256_of(ROOT / "world_model/baselines/diffusion_forcing/novphy.py")
    assert set(gport["arms"]) == {"ts", "planning"} and all(a["pass"] for a in gport["arms"].values())
    for arm in gport["arms"].values():
        assert [s["lr_equal"] for s in arm["steps"]] == [True, True] and arm["sampling"]["pyramid_rows"] == 95
    assert all(d["detected"] for d in gport["injected_defects"].values()) and len(gport["injected_defects"]) == 3


def test_vendor_is_verbatim_with_license_and_credit():
    vendor = json.loads((VENDOR / "VENDOR.json").read_text())
    assert vendor["commit"] == "dd65516a470b0eda08dcc0f9cb55c87d6527eb9e" and vendor["branch"] == "paper"
    for rel, entry in vendor["files"].items():
        assert eb.sha256_of(VENDOR / rel) == entry["vendored_sha256"] == entry["upstream_sha256"]
        assert entry["edits"] == []
    credit = "This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research template"
    assert credit in (VENDOR / "LICENSE").read_text() and "forked from [Boyuan Chen]" in (VENDOR / "README.md").read_text()
    for path in ("world_model/baselines/diffusion_forcing/novphy.py", "scripts/b127_dforce_fidelity.py",
                 "scripts/b127_dforce_port_reference.py"):
        assert "This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research template" in (
            ROOT / path).read_text()


def test_band_frozen_and_recorded_before_any_run():
    band_path = OUTPUT / "fidelity/band.json"
    band = eb.read_json(band_path)
    assert eb.sha256_of(band_path).split(":")[-1] in (OUTPUT / "decisions.md").read_text()
    assert (band["low"], band["high"]) == (0.0005, 0.0055) and band["reported"] == {
        "mean": 0.003, "sd": 0.001, "runs": 5, "statistic": "mean ± SD over five seeds", "decimals": 3}
    run_dir = OUTPUT / "fidelity/run"
    if run_dir.exists():  # any fidelity run started after the freeze
        progress = [eb.read_json(p) for p in run_dir.glob("seed-*/progress.json")]
        assert all(p["started_at"] >= band["frozen_at"] for p in progress)


def test_fidelity_run_refuses_without_band_and_parses_the_test_table(tmp_path):
    with pytest.raises(SystemExit):
        fid.run("cpu", 1, tmp_path)
    assert not (tmp_path / "fidelity").exists()
    rich = "│      test/crps_sum      │  0.0031234567   │\n"
    plain = "       test/crps_sum             0.0031234567\n"
    assert fid.TEST_METRIC.findall(rich) == ["0.0031234567"] == fid.TEST_METRIC.findall(plain)
    args = fid.overrides(1, "cpu", ["test"], Path("/x/best.ckpt"), tmp_path)
    assert "experiment.training.precision=32" in args and "load='/x/best.ckpt'" in args
    assert "experiment.tasks=[test]" in args and "wandb.mode=disabled" in args
    gpu = fid.overrides(1, "cuda", ["save_mean_std_metadata", "training"], None, tmp_path)
    assert not any("precision" in a for a in gpu) and not any(a.startswith("load=") for a in gpu)


FORBIDDEN = {"read_evaluation_anchors", "evaluation_cells", "position_cache", "read_novelty_anchors", "novelty_cells"}


@pytest.mark.parametrize("path", ["world_model/baselines/diffusion_forcing/novphy.py", "scripts/b127_dforce_fidelity.py",
                                  "scripts/b127_dforce_port_reference.py"])
def test_no_evaluation_access(path):
    tree = ast.parse((ROOT / path).read_text())
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not names & FORBIDDEN
    assert not any("issue109-evaluation" in s for s in strings)
