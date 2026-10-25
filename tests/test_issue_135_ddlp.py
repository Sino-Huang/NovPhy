"""CPU tests of #135 (T2 DDLP): adapter (episodes, launch particle, rollout, readout), runner (windows, resumable
training, typed failures, funding / readiness gates, scoring-path equality, the novelty mode), G-port defect
detection, the MEGA fetcher's key handling, and the absence of evaluation access."""
from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[1]

from scripts import external_baselines as eb  # noqa: E402
from scripts import run_issue_135_ddlp as runner  # noqa: E402
from world_model.baselines.ddlp import novphy as dd  # noqa: E402

torch.set_num_threads(4)
CLONE = Path("/mnt/array/sukaih/novphy-baselines-src/ddlp@5686275")
GPORT_REF = Path(os.environ.get("B135_GPORT_REF", Path.home() / ".cache/novphy-issue-135-gport"))
TINY = {**dd.NATIVE, "enc_channels": [4, 4, 8, 8, 8], "prior_channels": [4, 4, 8], "n_kp_enc": 3, "n_kp_prior": 6,
        "pint_layers": 1, "pint_heads": 2, "pint_dim": 16, "learned_feature_dim": 2}
MODEL_KEYS = [k for k in dd.NATIVE if k not in ("batch_size",)]


@pytest.fixture(autouse=True)
def tiny_models(monkeypatch):
    monkeypatch.setitem(runner.MODEL_CONFIGS, "tiny", TINY)


def tiny_spec(steps=4, epoch_steps=2, static=False):
    return {"name": "tiny", "arm": "T2", "configuration": "tiny", "model": "tiny", "batch": runner.BATCH,
            "epochs": math.ceil(steps / epoch_steps), "epoch_steps": epoch_steps, "steps": steps, "static": static,
            "launch": not static}


class FakeShots:
    def __init__(self, ends, keys=None):
        self.ends = np.asarray(ends)
        self.keys = keys or [f"s{i}" for i in range(len(ends))]

    def __len__(self):
        return len(self.ends)

    def carrier_rows(self, shots, positions):
        return np.stack([np.full(236, 0.01 * s + 0.001 * p, dtype=np.float32) for s, p in zip(shots, positions)])


def tiny_data(n=3, seed=0, ends=(285, 140, 230)):
    g = np.random.default_rng(seed)
    return {"frames": g.integers(0, 256, (n, len(dd.POSITIONS), 3, dd.IMAGE, dd.IMAGE), dtype=np.uint8),
            "launch5": torch.as_tensor(g.uniform(-1, 1, (n, 5)), dtype=torch.float32), "shots": FakeShots(ends[:n])}


def tiny_model(launch=True, seed=0):
    return dd.build_model(TINY, launch=launch, seed=seed)


# ---------------------------------------------------------------------------------------------------------------
# configuration and layout
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not dd.RELEASED_PHYRE_HPARAMS.is_file(), reason="released checkpoint not downloaded")
def test_native_is_the_released_phyre_checkpoint_config():
    released = dd.released_hparams()
    assert {k: released[k] for k in MODEL_KEYS} == {k: dd.NATIVE[k] for k in MODEL_KEYS}
    assert dd.BATCH == 12 and dd.NATIVE["batch_size"] == released["batch_size"] == 3


@pytest.mark.skipif(not (CLONE / "configs/phyre.json").is_file(), reason="official clone missing")
def test_repo_config_is_the_repository_phyre_json():
    repo = json.loads((CLONE / "configs/phyre.json").read_text())
    assert {k: repo[k] for k in MODEL_KEYS} == {k: dd.REPO_PHYRE[k] for k in MODEL_KEYS}


def test_episode_layout_window_counts_and_budgets():
    assert dd.EPISODE_FRAMES[:4] == (0, 0, 0, 0) and dd.EPISODE_FRAMES[4] == 1 and len(dd.EPISODE_FRAMES) == 49
    assert dd.EPISODE_FRAMES[-1] == 45 and dd.LAUNCH_INDEX == 3 and dd.STEPS == 45
    assert dd.windows_per_episode(dd.NATIVE) == 34
    assert dd.frame_positions(140)[:30] == list(range(0, 141, 5)) + [140]
    assert runner.config_spec("native")["steps"] == 40 * (13_919 * 34 // 12) == 1_577_480
    assert runner.config_spec("paper-schedule")["steps"] == 40 * 18_232 == 729_280
    assert runner.config_spec("repo-config")["model"] == "repo-phyre"
    r0 = runner.config_spec(runner.R0["name"])
    assert r0["steps"] == 30_000 and r0["static"] and not r0["launch"]
    window = dd.window(torch.arange(46), 0, dd.NATIVE)
    assert window.tolist() == [0, 0, 0, 0] + list(range(1, 13))


def test_resize_is_cv2_inter_area():
    import cv2
    frames = torch.randint(0, 256, (2, 3, 240, 320), dtype=torch.uint8)
    out = dd.resize_frames(frames)
    assert out.shape == (2, 3, 128, 128) and out.dtype == torch.uint8
    ref = cv2.resize(frames[1].permute(1, 2, 0).numpy(), (128, 128), interpolation=cv2.INTER_AREA)
    assert np.array_equal(out[1].permute(1, 2, 0).numpy(), ref)


def test_dynamics_discount_and_phases_follow_train_ddlp():
    cfg = dd.NATIVE
    for iteration, per_epoch in ((0, 18_232), (5000, 18_232), (90_000, 39_437), (7, 100)):
        warmup_iters = max(cfg["warmup_epoch"], 1) * max(10_000, per_epoch)
        per_step = warmup_iters // cfg["timestep_horizon"]
        ref = (iteration / torch.tensor([per_step * (i + 1) for i in range(cfg["timestep_horizon"])])).clamp(0, 1)
        assert torch.equal(dd.dynamics_discount(iteration, per_epoch, cfg), ref)
    assert dd.phase(0, cfg) == {"warmup": True, "noisy": True}
    assert dd.phase(1, cfg) == {"warmup": False, "noisy": True}
    assert dd.phase(2, cfg) == {"warmup": False, "noisy": False}


# ---------------------------------------------------------------------------------------------------------------
# the launch particle and rollouts
# ---------------------------------------------------------------------------------------------------------------

def test_launch_particle_is_reset_every_step_and_stripped(monkeypatch):
    model = tiny_model().eval()
    frames = torch.rand(2, 3, 128, 128)
    launch5 = torch.tensor([[0.1, 0.2, 1.0, 0.0, 1.0], [-0.3, 0.5, 1.0, 0.0, 1.0]])
    calls = []
    original = model.dyn_module.dyn.sample

    def spy(z, s, o, d, f, bg, steps=10, deterministic=False):
        calls.append((z[:, -1, -1].clone(), o[:, -1, -1].clone(), f[:, -1, -1].clone()))
        return original(z, s, o, d, f, bg, steps=steps, deterministic=deterministic)

    monkeypatch.setattr(model.dyn_module.dyn, "sample", spy)
    out = dd.rollout(model, frames, launch5, steps=5)
    model.set_launch(launch5)
    xy, _, obj_on, _, feat = model.dyn_module.launch_particle()
    model.set_launch(None)
    assert len(calls) == 5
    for z_last, o_last, f_last in calls:
        assert torch.equal(z_last, xy[:, 0, 0]) and torch.equal(o_last, obj_on[:, 0, 0])
        assert torch.equal(f_last, feat[:, 0, 0])
    assert out["z"].shape == (2, dd.STATIC + 5, TINY["n_kp_enc"], 2) and out["z_obj_on"].shape[-1] == TINY["n_kp_enc"]
    assert model.dyn_module.launch5 is None
    # training forward: every per-particle dynamics output has exactly K particles
    x = torch.rand(1, 16, 3, 128, 128)
    o = model.forward_launch(x, launch5[:1], forward_dyn=True, x_prior=x, warmup=False, noisy=False)
    assert o["mu_dyn"].shape[2] == TINY["n_kp_enc"] and o["obj_on_a_dyn"].shape[2] == TINY["n_kp_enc"]


def test_rollout_without_launch_equals_the_official_sample():
    model = tiny_model(launch=False).eval()
    frames0 = torch.rand(2, 3, 128, 128)
    x = torch.cat([dd.context(frames0), torch.zeros(2, 4, 3, 128, 128)], 1)
    with torch.no_grad():
        _, z_out = model.sample(x, num_steps=4, deterministic=True, cond_steps=dd.STATIC, return_z=True)
    ours = dd.rollout(model, frames0, None, steps=4)
    for k, v in {"z": "z_pos", "z_scale": "z_scale", "z_obj_on": "z_obj_on", "z_depth": "z_depth",
                 "z_features": "z_features", "z_bg": "z_bg_features"}.items():
        assert torch.equal(ours[k], z_out[v]), k


def test_launch_changes_the_rollout_and_rollouts_are_deterministic():
    model = tiny_model().eval()
    frame = torch.rand(1, 3, 128, 128).expand(2, -1, -1, -1)
    launches = torch.tensor([[0.1, 0.2, 1.0, 0.0, 1.0], [-0.9, -0.5, 1.0, 0.0, 1.0]])
    a = dd.rollout(model, frame, launches, steps=6)
    b = dd.rollout(model, frame, launches, steps=6)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not torch.allclose(a["z"][0, -1], a["z"][1, -1])
    assert torch.equal(a["z"][0, :dd.STATIC], a["z"][1, :dd.STATIC])  # the context does not see the launch


def test_predict_batching_matches_level_rows():
    """The scoring path (level_rows: one decision per forward) is shared by every scored pass; the batched path (G-rank /
    G-action over 480 fit anchors) agrees with it up to float round-off of batch-size-dependent CPU kernels."""
    from world_model.planning import scoring_harness as harness
    model = tiny_model().eval()
    probe = dd.ParticleReadout(6 + TINY["learned_feature_dim"], TINY["learned_feature_dim"]).eval()
    costs = runner.objective_costs()
    items, actions = runner.grid_actions()
    frames = torch.randint(0, 256, (3, 3, 128, 128), dtype=torch.uint8)
    with harness.deterministic_scoring():
        out = runner.predict(model, probe, frames, actions[None].expand(3, -1, -1), chunk=32)
        for i in range(3):
            rows = runner.level_rows(model, probe, items, frames[i], costs)
            assert rows == runner.level_rows(model, probe, items, frames[i], costs)  # deterministic
            batched = eb.endpoint_rows(items, out["carriers"][dd.STEPS][i], out["anchor"][i], costs)
            for a, b in zip(rows, batched):
                assert a["excluded"] == b["excluded"] and a["ordinal"] == b["ordinal"]
                assert a[eb.PRIMARY_COST] == pytest.approx(b[eb.PRIMARY_COST], rel=1e-4, abs=1e-6)


# ---------------------------------------------------------------------------------------------------------------
# readout
# ---------------------------------------------------------------------------------------------------------------

def test_readout_items_clamp_to_the_shot_end():
    items = runner.readout_items(FakeShots([285, 140, 226]))
    first = [(j, q) for s, j, q in items if s == 0]
    assert first == [(p // 5, p) for p in range(0, 226, 15)]
    second = [(j, q) for s, j, q in items if s == 1]
    assert second[-1] == (28, 140) and len(second) == 11  # 0..135 by 15, then the end 140 at cache index 28
    assert [(j, q) for s, j, q in items if s == 2][-1] == (45, 225)
    assert all(min(5 * j, end) == q for s, j, q in items for end in [[285, 140, 226][s]])


def test_readout_fits_and_inverts_the_z_score():
    g = torch.Generator().manual_seed(0)
    tokens, bg = torch.randn(512, 3, 8, generator=g), torch.randn(512, 2, generator=g)
    targets = torch.randn(512, 236, generator=g) * 3 + 5
    probe, report = dd.fit_readout(tokens, bg, targets, seed=1, epochs=2)
    assert report["steps"] == 2 * (512 // 128) and probe.mean.shape == (236,)
    out = dd.apply_readout(probe, tokens, bg)
    assert torch.allclose(out, probe.forward_z(tokens, bg) * probe.std + probe.mean)


def test_g_action_calibration_reports_the_engine_truth():
    cal = runner.g_action_calibration(0.02)
    if cal["source"] is None:
        pytest.skip("#129 calibration file absent")
    assert cal["engine_truth"] == pytest.approx(0.03946, abs=1e-4) and cal["fraction_of_engine_truth"] > 0.5


# ---------------------------------------------------------------------------------------------------------------
# training plumbing
# ---------------------------------------------------------------------------------------------------------------

def test_windows_cover_each_epoch_once_and_are_deterministic():
    data = tiny_data()
    w = runner.Windows(data["frames"], data["launch5"], 7, TINY, batch=12)
    assert w.n == 3 * 34 and w.steps_per_epoch == 8
    seen = np.concatenate([w.indices(s) for s in range(8)])
    assert len(set(seen.tolist())) == 96 and seen.max() < 102
    x, l5, shots, offsets = w.batch_of(3)
    x2, _, _, _ = runner.Windows(data["frames"], data["launch5"], 7, TINY, batch=12).batch_of(3)
    assert torch.equal(x, x2) and x.shape == (12, 16, 3, 128, 128)
    s, o = int(shots[0]), int(offsets[0])
    expect = data["frames"][s][list(dd.EPISODE_FRAMES[o:o + 16])]
    assert torch.equal(x[0], torch.from_numpy(expect).float() / 255) and torch.equal(l5[0], data["launch5"][s])
    assert not np.array_equal(w.indices(8), w.indices(0))  # a new permutation in data-epoch 1


def test_train_run_resume_is_bitwise_identical(tmp_path):
    data = tiny_data()
    full = runner.train_run(tmp_path / "full", tiny_spec(), 3, data, "cpu")
    part = runner.train_run(tmp_path / "split", tiny_spec(), 3, data, "cpu", max_steps=3)
    assert part["partial"] and (tmp_path / "split/resume.pt").is_file()
    resumed = runner.train_run(tmp_path / "split", tiny_spec(), 3, data, "cpu")
    a = torch.load(tmp_path / "full/model.pt", weights_only=False)
    b = torch.load(tmp_path / "split/model.pt", weights_only=False)
    assert full["complete"] and resumed["complete"] and not (tmp_path / "split/resume.pt").exists()
    assert all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
    model, payload = runner.load_trained(tmp_path / "full/model.pt", "cpu")
    assert payload["spec"]["model"] == "tiny" and not model.training


def test_static_r0_step_trains_the_particle_model_only(tmp_path):
    data = tiny_data()
    report = runner.train_run(tmp_path / "r0", tiny_spec(steps=2, static=True), 5, data, "cpu")
    assert report["complete"]
    model = dd.build_model(TINY, launch=False, seed=0)
    opt, _ = dd.build_optimizer(model, TINY)
    dd.train_step_static(model, opt, torch.rand(2, 16, 3, 128, 128), 2, TINY)
    assert all(p.grad is None for p in model.dyn_module.parameters())
    assert any(p.grad is not None for p in model.fg_module.parameters())


def test_nonfinite_loss_is_a_typed_terminal_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "train_step", lambda *a, **k: {"loss": float("nan")})
    report = runner.train_run(tmp_path / "nan", tiny_spec(), 3, tiny_data(), "cpu")
    assert not report["complete"] and report["typed_failure"]["typed_failure"] == "nonfinite_loss"
    model, payload = runner.load_trained(tmp_path / "nan/model.pt")
    assert model is None and payload["state_dict"] is None


# ---------------------------------------------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------------------------------------------

def test_funding_marker_and_readiness_gate_every_gpu_mode(tmp_path):
    out = tmp_path
    for fn in (lambda: runner.readiness(out, "cpu"), lambda: runner.selection(out, 0, "cpu"),
               lambda: runner.selection_readouts(out, "cpu"), lambda: runner.train(out, 0, "cpu"),
               lambda: runner.readouts(out, 0, "cpu"), lambda: runner.score(out, "cpu"),
               lambda: runner.score_novelty(out, "cpu")):
        assert fn() == 1
    assert sorted(p.name for p in out.iterdir()) == []
    eb.write_json(out / "funded.json", {"funded": False})
    assert not runner.funded(out) and runner.selection(out, 0, "cpu") == 1
    eb.write_json(out / "funded.json", {"funded": True, "decision": "url"})
    assert runner.selection_gate(out) == {"funded.json": True, "gport/gport.json": False,
                                          "fidelity/gfid.json": False, "readiness/readiness.json": False,
                                          "pass": False}
    assert runner.readiness(out, "cpu") == 1  # G-port not passed
    eb.write_json(out / "gport/gport.json", {"pass": True})
    eb.write_json(out / "fidelity/gfid.json", {"pass": True})
    eb.write_json(out / "readiness/readiness.json", {"pass": False, "typed_reason": runner.TYPED_STOP})
    assert runner.selection(out, 0, "cpu") == 1 and not (out / "selection").exists()  # the readiness stop holds
    eb.write_json(out / "readiness/readiness.json", {"pass": True})
    assert runner.selection_gate(out)["pass"]


def _novelty_setup(tmp_path, monkeypatch):
    plan = {"selected": "native"}
    data = tiny_data()
    for arm, seed in runner.arm_seeds():
        mdir = runner.model_dir(tmp_path, arm, seed, "native")
        runner.train_run(mdir, tiny_spec(steps=1), seed, data, "cpu")
        torch.manual_seed(seed)
        probe = dd.ParticleReadout(6 + TINY["learned_feature_dim"], TINY["learned_feature_dim"])
        runner.save_readout(runner.readouts_dir(tmp_path, arm, seed) / "readout.pt", probe, {}, "sha256:x")
    monkeypatch.setattr(runner, "load_plan", lambda output: plan)
    monkeypatch.setattr(runner, "load_manifest", lambda output: {})
    members = ["lvA", "lvB"]
    g = torch.Generator().manual_seed(1)
    frames = {m: torch.randint(0, 256, (3, 240, 320), dtype=torch.uint8, generator=g) for m in members}
    inventory = [{**item, "key": f"k{item['ordinal']}"} for item in eb.grid_items()]
    cells = {"levels": {m: {"inventory": inventory} for m in members}, "mixed": members}

    def fake_anchors(output, policy, cells_=None):
        payload = {"carriers": {m: torch.zeros(236) for m in members}, "frames": frames, "read_at": eb.utc_now(),
                   "scoring_policy": policy}
        eb.atomic_torch(eb.novelty_anchors_path(output), payload)
        return payload

    monkeypatch.setattr(eb, "read_novelty_anchors", fake_anchors)
    monkeypatch.setattr(eb, "novelty_reference_records", lambda *a, **k: {})
    return plan, cells, members, frames


def test_score_novelty_refuses_then_scores_with_the_primary_rows(tmp_path, monkeypatch):
    from world_model.planning import scoring_harness as harness
    eb.write_json(tmp_path / "funded.json", {"funded": True})
    plan, cells, members, frames = _novelty_setup(tmp_path, monkeypatch)
    assert runner.score_novelty(tmp_path, "cpu", cells) == 1  # before the primary records
    assert not runner.novelty_dir(tmp_path).exists()
    for arm, seed in runner.arm_seeds():
        eb.write_json(runner.record_path(tmp_path, arm, seed), {"arm": arm, "seed": seed, "typed_failure": None,
                                                                 "scored_at": "2026-10-25T00:00:00Z"})
    primary = {p: p.read_bytes() for p in (tmp_path / "records").iterdir()}
    passing = {"G_rank": True, "G_action": True, "G_readout": True, "G_nonfinite": True, "G_trained": True}
    monkeypatch.setattr(runner, "arm_guards", lambda output, records: {"T2": passing})
    monkeypatch.setattr(runner, "guard_extras", lambda output: {"G_port": True, "G_fid": True})
    assert runner.score_novelty(tmp_path, "cpu", cells) == 0
    assert eb.read_json(runner.novelty_dir(tmp_path) / "arms.json")["arms"]["T2"] == {"guards_pass": True, "failed": []}
    assert {p: p.read_bytes() for p in (tmp_path / "records").iterdir()} == primary
    costs = runner.objective_costs()
    for seed in eb.SEEDS:
        record = eb.read_json(runner.novelty_record_path(tmp_path, "T2", seed))
        assert record["schema"] == "issue_135_novelty_record_v1" and sorted(record["inventories"]["grid"]) == members
        model, probe = runner.frozen_predictor(tmp_path, plan, "T2", seed, "cpu")
        for m in members:
            with harness.deterministic_scoring():
                rows = runner.level_rows(model, probe, cells["levels"][m]["inventory"], dd.resize_frames(frames[m]),
                                         costs)
            assert record["inventories"]["grid"][m]["rows"] == json.loads(eb.json_text(rows))
    monkeypatch.setattr(runner, "arm_guards", lambda output, records: {"T2": {**passing, "G_readout": False}})
    (runner.novelty_dir(tmp_path) / "arms.json").unlink()
    for seed in eb.SEEDS:
        runner.novelty_record_path(tmp_path, "T2", seed).unlink()
    assert runner.score_novelty(tmp_path, "cpu", cells) == 0
    assert eb.read_json(runner.novelty_dir(tmp_path) / "arms.json")["arms"]["T2"]["failed"] == ["G_readout"]
    assert not any(runner.novelty_record_path(tmp_path, "T2", s).exists() for s in eb.SEEDS)


@pytest.mark.skipif(not runner.frame_paths()["manifest"].is_file(), reason="frame cache not built (--data)")
def test_cpu_ladder_readiness_selection_readouts_and_scoring_path(tmp_path, monkeypatch):
    """Readiness -> selection -> selection readouts -> readouts -> freeze -> the scoring path, on the real fit / policy
    frame products with a tiny model and subsets (no evaluation data: the scored cells are fabricated policy cells)."""
    from world_model.planning import scoring_harness as harness
    (tmp_path / "data").symlink_to(runner.OUTPUT / "data")
    eb.write_json(tmp_path / "funded.json", {"funded": True})
    eb.write_json(tmp_path / "gport/gport.json", {"pass": True})
    eb.write_json(tmp_path / "fidelity/gfid.json", {"pass": True})
    tiny = {"model": "tiny", "epochs": 1, "epoch_steps": 2}
    monkeypatch.setattr(runner, "R0", {"name": "r0-static", "model": "tiny", "epochs": 1, "epoch_steps": 2,
                                       "static": True})
    monkeypatch.setattr(runner, "CONFIGS", {"native": tiny, "paper-schedule": tiny, "repo-config": tiny})
    original_items = runner.readout_items
    monkeypatch.setattr(runner, "readout_items", lambda shots: original_items(shots)[:300])
    cells = eb.policy_cells("grid")
    few = dict(sorted(cells.items())[:3])
    monkeypatch.setattr(eb, "policy_cells", lambda inventory="grid": few)
    fit_frames = runner.load_fit_anchor_frames()
    anchors4 = dict(sorted(fit_frames.items())[:4])
    monkeypatch.setattr(runner, "load_fit_anchor_frames", lambda output=None: anchors4)
    monkeypatch.setitem(dd.READOUT, "epochs", 2)
    assert runner.readiness(tmp_path, "cpu") == 0
    ready = eb.read_json(tmp_path / "readiness/readiness.json")
    assert ready["schema"] == runner.SCHEMA_READINESS and ready["policy_grid_cells"] == 3
    assert ready["pass"] == (ready["ceiling_auc"] is not None and ready["ceiling_auc"] >= 0.65)
    assert ready["typed_reason"] in (None, runner.TYPED_STOP) and ready["r0"]["spec"]["static"]
    eb.write_json(tmp_path / "readiness/readiness.json", {**ready, "pass": True})  # continue the ladder in the test
    for q in range(3):
        assert runner.selection(tmp_path, q, "cpu") == 0
    assert runner.selection_readouts(tmp_path, "cpu") == 0
    sel = eb.read_json(tmp_path / "selection.json")
    assert sel["selected"] in runner.CONFIGS and all("ceiling_auc" in d for d in sel["details"].values())
    plan = {"selected": sel["selected"], "queues": {"readouts": {"0": [f"T2--seed{s}" for s in eb.SEEDS]}},
            "frozen_at": eb.utc_now()}
    monkeypatch.setattr(runner, "load_plan", lambda output: plan)
    data = runner.load_data(tmp_path)
    for seed in eb.SEEDS[1:]:
        runner.train_run(runner.model_dir(tmp_path, "T2", seed), runner.arm_spec("T2", sel["selected"]), seed, data,
                         "cpu")
    assert runner.readouts(tmp_path, 0, "cpu") == 0
    for seed in eb.SEEDS:
        rd = eb.read_json(runner.readouts_dir(tmp_path, "T2", seed) / "readouts.json")
        assert rd["typed_failure"] is None and rd["G_rank"]["levels"] == 4 and rd["G_action"] >= 0
        assert rd["G_action_calibration"]["arm"] == rd["G_action"] and rd["macs_per_candidate"] > 0
        assert rd["policy_auc"] is None or 0 <= rd["policy_auc"] <= 1
    reused = runner.readouts_dir(tmp_path, "T2", eb.SELECTION_SEED) / "readout.pt"
    assert reused.read_bytes() == (runner.selection_dir(tmp_path, sel["selected"]) / "readout.pt").read_bytes()
    assert runner.freeze_readouts(tmp_path) == 0 and runner.load_manifest(tmp_path)["entries"]
    # the scoring path on fabricated (policy) cells; position helpers faked (no evaluation content is read)
    pa = eb.policy_anchor_frames()
    fake_cells = {inv: {"mixed": few} for inv in eb.INVENTORIES}
    fake_anchors = {"frames": {m: pa[m] for m in few}}
    branches = [(m, few[m]["inventory"][0]) for m in sorted(few)]
    monkeypatch.setattr(eb, "position_branches", lambda cells_: branches)
    monkeypatch.setattr(eb, "position_cache", lambda: {"entries": {it["key"]: {
        "presence": torch.ones(5, 18, dtype=torch.bool), "centers": torch.zeros(5, 18, 2)} for _, it in branches}})
    with harness.deterministic_scoring():
        a = runner.score_arm(tmp_path, plan, "T2", eb.SELECTION_SEED, fake_anchors, fake_cells, "cpu")
        b = runner.score_arm(tmp_path, plan, "T2", eb.SELECTION_SEED, fake_anchors, fake_cells, "cpu")
    assert json.loads(eb.json_text(a["inventories"])) == json.loads(eb.json_text(b["inventories"]))
    assert sorted(a["wrong_anchor"]) == sorted(anchors4) and set(a["positions"]) == {it["key"] for _, it in branches}
    assert all(len(cell["rows"]) == 16 for cell in a["inventories"]["grid"].values())


def test_publish_not_run_lists_the_typed_reason(tmp_path):
    with pytest.raises(SystemExit):
        runner.publish(tmp_path)  # neither a plan nor a typed reason
    eb.write_json(tmp_path / "gport/gport.json", {"pass": True})
    eb.write_json(tmp_path / "not-run.json", {"typed_reason": "cap", "decision": "url"})
    assert runner.publish(tmp_path) == 0
    summary = eb.load_child_summary(tmp_path / "summary.json")
    assert summary["arms"] == {} and summary["extra"]["pending_arms"] == {"T2": "cap"}
    assert summary["extra"]["G_port"] is True and summary["gfid"]["T2"]["pass"] is False
    eb.write_json(tmp_path / "readiness/readiness.json", {"pass": False, "typed_reason": runner.TYPED_STOP})
    assert runner.publish(tmp_path) == 0
    assert eb.read_json(tmp_path / "summary.json")["extra"]["pending_arms"] == {"T2": runner.TYPED_STOP}


def test_measured_projection_uses_the_timing_record():
    record = {"configs": {"native": {"seconds_per_step_steady_median": 0.1},
                          "paper-schedule": {"seconds_per_step_steady_median": 0.1},
                          "repo-config": {"seconds_per_step_steady_median": 0.15},
                          "r0-static": {"seconds_per_step_steady_median": 0.05}},
              "encode_seconds_per_frame": 0.001, "rollout_seconds_per_decision": 0.5, "readout_fit_frames": 200_000}
    m = runner.measured_projection(record)
    assert m["three"]["selection_gpu_h"]["native"] == pytest.approx(1_577_480 * 0.1 / 3600)
    assert m["three"]["selection_gpu_h"]["repo-config"] == pytest.approx(1_577_480 * 0.15 / 3600)
    assert m["two_after_cut"]["configurations"] == ["native", "paper-schedule"]
    assert m["r0_gpu_h"] == pytest.approx(30_000 * 0.05 / 3600)


# ---------------------------------------------------------------------------------------------------------------
# G-port defect detection, the MEGA fetcher, no evaluation access
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not (GPORT_REF / "novphy-launch.pt").is_file(), reason="G-port reference not generated")
def test_gport_passes_and_detects_injected_defects(monkeypatch):
    from scripts import b127_ddlp_port_reference as gp
    assert gp.compare_case("novphy-launch")["pass"]
    original = dd.LaunchDynamics.launch_particle

    def shifted(self):  # the launch particle one unit off in x
        xy, s, o, d, f = original(self)
        return xy + torch.tensor([1.0, 0.0]), s, o, d, f

    monkeypatch.setattr(dd.LaunchDynamics, "launch_particle", shifted)
    assert not gp.compare_case("novphy-launch")["pass"]
    monkeypatch.setattr(dd.LaunchDynamics, "launch_particle", original)
    monkeypatch.setattr(dd, "dynamics_discount", lambda it, per, cfg: torch.ones(cfg["timestep_horizon"]))
    out = gp.compare_case("phyre-released")
    assert not out["pass"] and not out["steps"][0]["checks"]["loss_terms"]


def test_mega_key_split_and_chunk_bounds():
    from scripts import b127_ddlp_mega as mega
    fid, raw = mega.parse_link("https://mega.nz/file/UBcl2LgQ#cA2mS1mfSoAbNa8VmjeSkrRHDlXlupjVNNXnhO8zoPs")
    assert fid == "UBcl2LgQ" and len(raw) == 32
    aes, nonce, meta = mega.split_key(raw)
    assert len(aes) == 16 and len(nonce) == 8 and len(meta) == 8
    bounds = list(mega.chunk_bounds(3 * 2 ** 20))
    assert bounds[0] == (0, 0x20000) and bounds[1] == (0x20000, 0x60000)
    assert all(b - a == 0x100000 for a, b in bounds[8:-1]) and bounds[-1][1] == 3 * 2 ** 20
    assert all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1))


FORBIDDEN = {"read_evaluation_anchors", "evaluation_cells", "position_cache", "read_novelty_anchors", "novelty_cells"}


@pytest.mark.parametrize("path", ["world_model/baselines/ddlp/novphy.py", "scripts/b127_ddlp_fidelity.py",
                                  "scripts/b127_ddlp_port_reference.py", "scripts/b127_ddlp_mega.py"])
def test_no_evaluation_access(path):
    tree = ast.parse((ROOT / path).read_text())
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not names & FORBIDDEN
    assert not any("issue109-evaluation" in s for s in strings)
