"""#137 T5 DIAMOND: CPU tests (adapter, G-port record, gates, readiness / not-run publication, resume, projection,
G-fid band logic). Run with CUDA_VISIBLE_DEVICES=""."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

from scripts import external_baselines as eb  # noqa: E402
from scripts import run_issue_137_diamond as runner  # noqa: E402
from world_model.baselines.diamond import novphy as dn  # noqa: E402

TINY_AGENT = {"sigma_data": 0.5, "sigma_offset_noise": 0.3,
              "inner_model": {"img_channels": 3, "num_steps_conditioning": 4, "cond_channels": 16,
                              "depths": [1, 1, 1, 1], "channels": [8, 8, 8, 8], "attn_depths": [0, 0, 0, 0]}}


@pytest.fixture
def tiny(monkeypatch):
    """Tiny U-Net, 2 paths, 4-update configurations (the official code paths, small sizes)."""
    monkeypatch.setattr(dn, "AGENT", TINY_AGENT)
    monkeypatch.setattr(runner, "PATHS", 2)
    monkeypatch.setattr(runner, "CONFIGS", {name: {**c, "updates": 4, "batch": 2}
                                            for name, c in runner.CONFIGS.items()})


def tiny_data(n=3, seed=0):
    g = np.random.default_rng(seed)
    return {"frames": g.integers(0, 256, (n, len(dn.POSITIONS), 3, dn.IMAGE, dn.IMAGE), dtype=np.uint8),
            "launch5": torch.as_tensor(g.normal(size=(n, 5)), dtype=torch.float32)}


def live_model(seed=0):
    """A tiny denoiser whose zero-initialised output convolutions are randomised (an untrained official model ignores
    its conditioning: conv_out and every conv2 start at zero)."""
    torch.manual_seed(seed)
    model = dn.build_denoiser(TINY_AGENT)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("conv_out.weight") or name.endswith("conv2.weight"):
                p.normal_(0, 0.2)
    return model.eval().requires_grad_(False)


# ---------------------------------------------------------------------------------------------------------------
# vendoring and the adapter
# ---------------------------------------------------------------------------------------------------------------

def test_vendor_is_verbatim_and_mit():
    meta = json.loads((dn.VENDOR / "VENDOR.json").read_text())
    assert meta["schema"] == "issue_127_vendor_v1" and meta["commit"].startswith("5bcd159")
    for rel, entry in meta["files"].items():
        digest = "sha256:" + hashlib.sha256((dn.VENDOR / rel).read_bytes()).hexdigest()
        assert digest == entry["vendored_sha256"] == entry["upstream_sha256"] and entry["edits"] == []
    assert (dn.VENDOR / "LICENSE").read_text().startswith("MIT License")


def test_denoiser_is_the_official_config_with_the_launch_table():
    torch.manual_seed(0)
    model = dn.build_denoiser()
    assert dn.trainable_parameters(model) == 4_406_019
    table = model.inner_model.act_emb[0]
    assert isinstance(table, dn.LaunchEmbedding) and isinstance(table, torch.nn.Embedding)
    assert tuple(table.weight.shape) == (5, 64)
    launch = torch.randn(2, 4, 5)
    assert torch.allclose(model.inner_model.act_emb(launch), (launch @ table.weight).flatten(1))
    assert torch.equal(model.inner_model.act_emb(torch.zeros(2, 4, 5)), torch.zeros(2, 256))
    assert "data" not in sys.modules or not hasattr(sys.modules["data"], "Batch") or \
        sys.modules["data"].__name__ != "data" or "vendor" in str(getattr(sys.modules["data"], "__file__", ""))
    assert "utils" not in sys.modules or getattr(sys.modules["utils"], "LossAndLogs", None) is None
    names_decay = {n for g in dn.configure_opt(model, **dn.OPTIMIZER).param_groups if g["weight_decay"] > 0
                   for n in [k for k, p in model.named_parameters() if any(p is q for q in g["params"])]}
    assert "inner_model.act_emb.0.weight" not in names_decay and "inner_model.conv_in.weight" in names_decay
    assert dn.NATIVE_UPDATES == 409_600 and dn.seq_length() == 6


def test_resize_is_exact_area_averaging():
    flat = torch.full((3, 240, 320), 77, dtype=torch.uint8)
    assert torch.equal(dn.resize_frames(flat), torch.full((3, 64, 64), 77, dtype=torch.uint8))
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 256, (2, 3, 128, 128), dtype=torch.uint8, generator=g)
    box = x.double().reshape(2, 3, 64, 2, 64, 2).mean((3, 5))
    assert torch.equal(dn.resize_frames(x), torch.round(box).to(torch.uint8))  # 2x: the 2 x 2 box mean
    w = dn._area_weights(240, 64)
    assert torch.allclose(w.sum(1), torch.ones(64, dtype=torch.float64)) and w.shape == (64, 240)


def test_episode_windows_padding_and_launch_slot():
    data = tiny_data(2)
    length = dn.seq_length()
    batch = dn.make_batch(data["frames"], data["launch5"], [0, 1, 1], [-2, 0, 43], length)
    assert batch.obs.shape == (3, length, 3, 64, 64) and batch.act.shape == (3, length, 5)
    assert batch.mask_padding[0].tolist() == [False, False, True, True, True, True]
    assert torch.equal(batch.obs[0, :2], torch.zeros(2, 3, 64, 64))
    for t in range(2, length):  # episode index t - 2 -> stored frame EPISODE_FRAMES[t - 2]
        assert torch.equal(batch.obs[0, t], dn.to_unit(data["frames"][0][dn.EPISODE_FRAMES[t - 2]]))
    assert torch.equal(batch.act[0, 2 + dn.LAUNCH_INDEX], data["launch5"][0])  # the launch at the decision frame
    assert batch.act[0].abs().sum() == data["launch5"][0].abs().sum()
    assert torch.equal(batch.act[1, dn.LAUNCH_INDEX], data["launch5"][1]) and batch.act[2].abs().sum() == 0
    assert torch.equal(batch.obs[2, -1], dn.to_unit(data["frames"][1][45]))  # window ends at the frame-225 slot
    starts_all = []
    for b in range(50):
        shots, starts = dn.sample_segments(eb.batch_rng(1, b), 7, 32, length)
        assert shots.min() >= 0 and shots.max() < 7
        assert starts.min() >= -(length - 1) and (starts + length).max() <= len(dn.EPISODE_FRAMES)
        starts_all.append(starts)
    a = dn.sample_segments(eb.batch_rng(1, 3), 7, 32, length)
    assert all(np.array_equal(x, y) for x, y in zip(a, dn.sample_segments(eb.batch_rng(1, 3), 7, 32, length)))
    assert np.concatenate(starts_all).min() < 0  # left padding occurs, as in the official sampler


def test_rollout_is_causal_and_launch_conditioned():
    model = live_model()
    frames0 = dn.to_unit(torch.as_tensor(tiny_data(1)["frames"][:, 0])).expand(2, -1, -1, -1)
    launch = torch.stack([torch.zeros(5), torch.tensor([0.2, -0.1, 1.0, 0.0, 1.0])])
    g = torch.Generator().manual_seed(3)
    noise = torch.randn(5, 1, 3, 64, 64, generator=g).expand(-1, 2, -1, -1, -1).clone()
    out = dn.rollout(model, frames0, launch, noise, keep=(1, 3, 5))
    assert not torch.equal(out[1][0], out[1][1])  # the launch changes the first predicted frame
    changed = noise.clone()
    changed[3:] = torch.randn(2, 2, 3, 64, 64, generator=g)
    again = dn.rollout(model, frames0, launch, changed, keep=(1, 3, 5))
    assert torch.equal(again[1], out[1]) and torch.equal(again[3], out[3])  # step k ignores later noise
    assert not torch.equal(again[5], out[5])
    assert out[5].abs().max() <= 1 and torch.isfinite(out[5]).all()


def test_predict_rows_do_not_depend_on_batchmates(tiny):
    """Common random numbers: a candidate's paths see the same noise whatever its batchmates (rows agree up to CPU
    convolution blocking, amplified by the official 8-bit quantisation of each denoised frame); the same call is
    bitwise reproducible (scoring always batches one level's 16 x PATHS rows)."""
    model = live_model()
    probe = dn.FrameReadout().eval().requires_grad_(False)
    frames = torch.as_tensor(tiny_data(2)["frames"][:, 0])
    _, actions = runner.grid_actions()
    both = runner.predict(model, probe, frames, actions[None, :2].expand(2, -1, -1), keep=(2,))
    one = runner.predict(model, probe, frames[:1], actions[None, :1], keep=(2,))
    assert both["carriers"][2].shape == (2, 2, 236)
    assert torch.allclose(both["carriers"][2][0, 0], one["carriers"][2][0, 0], atol=1e-2)
    assert torch.allclose(both["anchor"][0], one["anchor"][0], atol=1e-5)
    again = runner.predict(model, probe, frames, actions[None, :2].expand(2, -1, -1), keep=(2,))
    assert torch.equal(again["carriers"][2], both["carriers"][2])


@pytest.mark.skipif(not runner.frame_paths()["manifest"].is_file(), reason="frame cache not built (--data)")
def test_cpu_ladder_readiness_selection_readouts_and_scoring_path(tmp_path, tiny, monkeypatch):
    """Readiness -> selection -> selection readouts -> readouts -> freeze -> the scoring path, on the real fit / policy
    frame products with a tiny U-Net and subsets (no evaluation data: the scored cells are fabricated policy cells)."""
    from world_model.planning import scoring_harness as harness
    (tmp_path / "data").symlink_to(runner.OUTPUT / "data")
    eb.write_json(tmp_path / "funded.json", {"funded": True})
    eb.write_json(tmp_path / "gport/gport.json", {"pass": True})
    eb.write_json(tmp_path / "fidelity/gfid.json", {"pass": True})
    original_items = runner.readout_items
    monkeypatch.setattr(runner, "readout_items", lambda shots: original_items(shots)[:300])
    few = dict(sorted(eb.policy_cells("grid").items())[:3])
    monkeypatch.setattr(eb, "policy_cells", lambda inventory="grid": few)
    anchors4 = dict(sorted(runner.load_fit_anchor_frames().items())[:4])
    monkeypatch.setattr(runner, "load_fit_anchor_frames", lambda output=None: anchors4)
    monkeypatch.setitem(dn.READOUT, "epochs", 1)
    assert runner.readiness(tmp_path, "cpu") == 0
    ready = eb.read_json(tmp_path / "readiness/readiness.json")
    assert ready["schema"] == runner.SCHEMA_READINESS and ready["policy_grid_cells"] == 3
    assert ready["readout"]["fit_frames"] == 300 and ready["e99_true_carrier_ceiling_auc"] is not None
    eb.write_json(tmp_path / "readiness/readiness.json", {**ready, "pass": True})  # continue the ladder in the test
    for q in range(3):
        assert runner.selection(tmp_path, q, "cpu") == 0
    assert runner.selection_readouts(tmp_path, "cpu") == 0
    sel = eb.read_json(tmp_path / "selection.json")
    assert sel["selected"] in runner.CONFIGS and sel["policy_cells"] == 3
    plan = {"selected": sel["selected"], "queues": {"readouts": {"0": [f"T5--seed{s}" for s in eb.SEEDS]}},
            "frozen_at": eb.utc_now()}
    monkeypatch.setattr(runner, "load_plan", lambda output: plan)
    data = runner.load_data(tmp_path)
    for seed in eb.SEEDS[1:]:
        runner.train_run(runner.model_dir(tmp_path, "T5", seed), runner.arm_spec("T5", sel["selected"]), seed, data,
                         "cpu")
    assert runner.readouts(tmp_path, 0, "cpu") == 0
    for seed in eb.SEEDS:
        rd = eb.read_json(runner.readouts_dir(tmp_path, "T5", seed) / "readouts.json")
        assert rd["typed_failure"] is None and rd["G_rank"]["levels"] == 4 and rd["G_action"] >= 0
        assert rd["policy_ceiling_auc"] == ready["ceiling_auc"] and rd["macs_per_candidate"] > 0
        assert rd["readout_sha256"] == eb.sha256_of(tmp_path / "readiness/readout.pt")
    assert runner.freeze_readouts(tmp_path) == 0 and runner.load_manifest(tmp_path)["entries"]
    pa = eb.policy_anchor_frames()
    fake_cells = {inv: {"mixed": few} for inv in eb.INVENTORIES}
    fake_anchors = {"frames": {m: pa[m] for m in few}}
    branches = [(m, few[m]["inventory"][0]) for m in sorted(few)]
    monkeypatch.setattr(eb, "position_branches", lambda cells_: branches)
    monkeypatch.setattr(eb, "position_cache", lambda: {"entries": {it["key"]: {
        "presence": torch.ones(5, 18, dtype=torch.bool), "centers": torch.zeros(5, 18, 2)} for _, it in branches}})
    with harness.deterministic_scoring():
        a = runner.score_arm(tmp_path, plan, "T5", eb.SELECTION_SEED, fake_anchors, fake_cells, "cpu")
        b = runner.score_arm(tmp_path, plan, "T5", eb.SELECTION_SEED, fake_anchors, fake_cells, "cpu")
    assert json.loads(eb.json_text(a["inventories"])) == json.loads(eb.json_text(b["inventories"]))
    assert sorted(a["wrong_anchor"]) == sorted(anchors4) and set(a["positions"]) == {it["key"] for _, it in branches}
    assert all(len(cell["rows"]) == 16 for cell in a["inventories"]["grid"].values())


# ---------------------------------------------------------------------------------------------------------------
# G-port
# ---------------------------------------------------------------------------------------------------------------

GPORT = runner.OUTPUT / "gport"


@pytest.mark.skipif(not (GPORT / "reference.pt").is_file(), reason="G-port reference not generated")
def test_gport_record_passes_and_detects_injected_defects(tmp_path, monkeypatch):
    record = eb.read_json(GPORT / "gport.json")
    assert record["schema"] == "issue_127_gport_v1" and record["pass"]
    assert set(record["cases"]) == {"atari-discrete", "launch", "launch-ar4"}
    assert all(c["exact"] for c in record["cases"].values()) and record["cases"]["launch"]["rollout"]["steps"] == 45
    assert record["adapter_sha256"] == "sha256:" + hashlib.sha256(
        (ROOT / "world_model/baselines/diamond/novphy.py").read_bytes()).hexdigest()
    from scripts import b127_diamond_port_reference as gp
    (tmp_path / "gport").mkdir()
    for name in ("inputs.pt", "reference.pt"):
        shutil.copy(GPORT / name, tmp_path / "gport" / name)
    monkeypatch.setattr(gp, "CASES", {"launch": gp.CASES["launch"]})

    def failed(defect):
        with monkeypatch.context() as m:
            defect(m)
            assert gp.compare(tmp_path) == 1
        return eb.read_json(tmp_path / "gport/gport.json")["cases"]["launch"]

    case = failed(lambda m: m.setattr(dn, "LAUNCH_INDEX", 2))  # launch one frame early
    assert not case["batch0_equal"]
    case = failed(lambda m: m.setattr(dn, "get_lr_sched", lambda opt, n: torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 1.0)))  # no warm-up
    assert not case["steps"][0]["lr_equal"]
    original = dn.rollout

    def no_roll(denoiser, frames0, launch5, noise, keep=(dn.STEPS,), sampler=None):
        return original(denoiser, frames0, launch5, noise[:1].expand(len(noise), -1, -1, -1, -1), keep, sampler)
    case = failed(lambda m: m.setattr(dn, "rollout", no_roll))  # one noise draw reused at every step
    assert not case["rollout"]["pass"]


# ---------------------------------------------------------------------------------------------------------------
# gates, readiness, not-run publication
# ---------------------------------------------------------------------------------------------------------------

def test_funding_and_gates_refuse_every_gpu_mode(tmp_path):
    out = tmp_path
    for fn in (lambda: runner.selection(out, 0, "cpu"), lambda: runner.selection_readouts(out, "cpu"),
               lambda: runner.train(out, 0, "cpu"), lambda: runner.readouts(out, 0, "cpu"),
               lambda: runner.score(out, "cpu"), lambda: runner.score_novelty(out, "cpu")):
        assert fn() == 1
    assert sorted(p.name for p in out.iterdir()) == []
    eb.write_json(out / "funded.json", {"funded": False})
    assert not runner.funded(out) and runner.selection(out, 0, "cpu") == 1
    eb.write_json(out / "funded.json", {"funded": True, "decision": "url"})
    assert runner.selection_gate(out) == {"funded.json": True, "gport/gport.json": False,
                                          "fidelity/gfid.json": False, "readiness/readiness.json": False,
                                          "pass": False}
    assert runner.score(out, "cpu") == 1  # G-fid not passed: refused before any plan / evaluation access
    eb.write_json(out / "gport/gport.json", {"pass": True})
    eb.write_json(out / "fidelity/gfid.json", {"pass": True})
    eb.write_json(out / "readiness/readiness.json", {"pass": False, "typed_reason": runner.TYPED_STOP})
    assert runner.selection(out, 0, "cpu") == 1 and not (out / "selection").exists()  # the readiness stop holds
    eb.write_json(out / "fidelity/gfid.json", {"pass": False})
    eb.write_json(out / "readiness/readiness.json", {"pass": True})
    assert not runner.selection_gate(out)["pass"] and runner.selection(out, 0, "cpu") == 1
    eb.write_json(out / "fidelity/gfid.json", {"pass": True})
    assert runner.selection_gate(out)["pass"]
    assert not eb.evaluation_anchors_path(out).exists()


class FakeShots:
    def __init__(self, n=6, seed=0):
        g = np.random.default_rng(seed)
        self.ends = np.array([225, 120, 300, 225, 64, 230][:n])
        self._carriers = torch.as_tensor(g.normal(size=(n, 301, 236)), dtype=torch.float32)

    def __len__(self):
        return len(self.ends)

    def carrier_rows(self, shots, positions):
        return self._carriers[shots, positions]


def _readiness_setup(tmp_path, monkeypatch, ceiling):
    data = {**tiny_data(6), "shots": FakeShots()}
    monkeypatch.setattr(runner, "load_data", lambda output: data)
    monkeypatch.setattr(runner, "load_policy_frames", lambda output: {})
    monkeypatch.setattr(runner, "objective_costs", lambda: None)
    monkeypatch.setattr(runner, "policy_ceiling", lambda probe, costs, pframes: {
        "ceiling_auc": ceiling, "policy_grid_cells": 94, "policy_candidates": 1504})
    monkeypatch.setattr(runner, "e99_true_ceiling", lambda costs: 0.7223)
    monkeypatch.setitem(dn.READOUT, "epochs", 2)
    monkeypatch.setitem(dn.READOUT, "batch", 16)


def test_readiness_stop_publishes_the_not_run_summary(tmp_path, monkeypatch):
    _readiness_setup(tmp_path, monkeypatch, 0.5948)
    assert runner.readiness(tmp_path, "cpu") == 0
    record = eb.read_json(tmp_path / "readiness/readiness.json")
    assert record["pass"] is False and record["typed_reason"] == runner.TYPED_STOP and record["threshold"] == 0.65
    assert record["readout"]["fit_frames"] == len(runner.readout_items(FakeShots())) and record["readout"]["epochs"] == 2
    assert record["data"]["evaluation"] == "not read"
    assert runner.readiness(tmp_path, "cpu") == 0  # one run, never re-run
    assert runner.not_run_reason(tmp_path) == runner.TYPED_STOP
    assert runner.selection(tmp_path, 0, "cpu") == 1
    assert runner.publish(tmp_path) == 0
    summary = eb.load_child_summary(tmp_path / "summary.json")
    assert summary["arms"] == {} and summary["extra"]["pending_arms"] == {"T5": runner.TYPED_STOP}
    assert summary["extra"]["readiness"]["ceiling_auc"] == 0.5948 and summary["frozen_at"] is None
    assert "readiness_ceiling_below_threshold" in (tmp_path / "findings.md").read_text()


def test_readiness_pass_continues_and_cap_marker_publishes(tmp_path, monkeypatch):
    _readiness_setup(tmp_path, monkeypatch, 0.70)
    assert runner.readiness(tmp_path, "cpu") == 0
    assert runner.readiness_pass(tmp_path) and runner.not_run_reason(tmp_path) is None
    with pytest.raises(SystemExit):
        runner.publish(tmp_path)  # neither a plan nor a typed reason
    eb.write_json(tmp_path / "not-run.json", {"typed_reason": "cap", "decision": "url"})
    assert runner.publish(tmp_path) == 0
    assert eb.read_json(tmp_path / "summary.json")["extra"]["pending_arms"] == {"T5": "cap"}


def test_readout_fit_resume_is_identical(monkeypatch):
    monkeypatch.setitem(dn.READOUT, "batch", 8)
    g = torch.Generator().manual_seed(0)
    frames = torch.randint(0, 256, (40, 3, 64, 64), dtype=torch.uint8, generator=g)
    targets = torch.randn(40, 236, generator=g)
    full, _ = dn.fit_readout(frames, targets, 5, epochs=3)
    saved = []

    class Stop(Exception):
        pass

    def checkpoint(state):
        saved.append(state)
        raise Stop

    with pytest.raises(Stop):
        dn.fit_readout(frames, targets, 5, epochs=3, checkpoint=checkpoint)
    resumed, report = dn.fit_readout(frames, targets, 5, epochs=3, state=saved[0])
    assert report["steps"] == 15
    assert all(torch.equal(a, b) for a, b in zip(full.state_dict().values(), resumed.state_dict().values()))


# ---------------------------------------------------------------------------------------------------------------
# training: resume determinism, stop signal, typed failure
# ---------------------------------------------------------------------------------------------------------------

def test_train_run_resume_is_bitwise_identical(tmp_path, tiny):
    data = tiny_data(3)
    spec = runner.config_spec("ar4")
    whole = runner.train_run(tmp_path / "whole", spec, 7, data, "cpu")
    part = runner.train_run(tmp_path / "split", spec, 7, data, "cpu", max_steps=2)
    assert part["partial"] and (tmp_path / "split/resume.pt").is_file()
    runner.train_run(tmp_path / "split", spec, 7, data, "cpu")
    a = torch.load(tmp_path / "whole/model.pt", weights_only=False)
    b = torch.load(tmp_path / "split/model.pt", weights_only=False)
    assert whole["complete"] and a["complete"] and b["complete"] and not (tmp_path / "split/resume.pt").exists()
    assert all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
    assert a["spec"]["training"]["num_autoregressive_steps"] == 4


def test_stop_signal_pauses_with_exit_75(tmp_path, tiny, monkeypatch):
    monkeypatch.setattr(eb, "STOP_REQUESTED", [15])
    monkeypatch.setattr(eb, "install_stop_handlers", lambda: None)
    item = (runner.config_spec("native"), 7, tmp_path / "run")
    assert runner.run_queue(tmp_path, [item], "cpu", tiny_data(2)) == eb.EXIT_PAUSED
    assert (tmp_path / "run/resume.pt").is_file() and not (tmp_path / "run/model.pt").exists()
    assert (tmp_path / "ledger").is_dir()


def test_nonfinite_loss_is_a_typed_terminal_failure(tmp_path, tiny, monkeypatch):
    monkeypatch.setattr(dn, "train_step", lambda *a, **k: {"loss": math.nan, "grad_norm": math.nan, "lr": 0.0})
    report = runner.train_run(tmp_path, runner.config_spec("native"), 7, tiny_data(2), "cpu")
    assert report["complete"] is False and report["typed_failure"]["typed_failure"] == "nonfinite_loss"
    model, payload = runner.load_trained(tmp_path / "model.pt")
    assert model is None and payload["state_dict"] is None


# ---------------------------------------------------------------------------------------------------------------
# projection math, G-fid band logic, no evaluation access
# ---------------------------------------------------------------------------------------------------------------

def test_projection_math_and_measured_projection():
    rates = {"native": 0.05, "ar4": 0.12, "native-2x": 0.05}
    p = runner.child_projection(rates, 1.0)
    n = dn.NATIVE_UPDATES
    assert p["selection_gpu_h"] == pytest.approx({"native": n * 0.05 / 3600, "ar4": n * 0.12 / 3600,
                                                  "native-2x": 2 * n * 0.05 / 3600})
    assert p["training_gpu_h"] == pytest.approx(2 * n * 0.12 / 3600)  # two seeds at the most expensive config
    assert p["score_gpu_h"] == pytest.approx(3 * runner.SCORING_DECISIONS / 3600)
    assert p["readouts_gpu_h"] == pytest.approx(3 * runner.READOUT_DECISIONS / 3600)
    other = (3 * runner.POLICY_DECISIONS + 3 * runner.READOUT_DECISIONS + 3 * runner.SCORING_DECISIONS) / 3600
    assert p["total_gpu_h"] == pytest.approx(sum(p["selection_gpu_h"].values()) + p["training_gpu_h"] + other)
    assert p["training_queues"] == {0: ["T5--seed20260909"], 1: ["T5--seed20260910"], 2: [], 3: []}
    assert p["within_caps"] == (p["total_gpu_h"] <= 136 and p["wall_h_on_4_gpus_serial_ladder"] <= 48
                                and p["native_per_seed_gpu_h_max"] <= 24)
    record = {"configs": {k: {"seconds_per_update_steady_median": v} for k, v in rates.items()},
              "scoring": {"seconds_per_decision": 1.0}}
    m = runner.measured_projection(record)
    assert m["total_gpu_h"] == pytest.approx(p["total_gpu_h"]) and m["source"].startswith("timing.json")
    queues, loads = runner.lpt([("a", 3.0), ("b", 2.0), ("c", 2.0), ("d", 1.0), ("e", 1.0)])
    assert sorted(loads) == [1.0, 2.0, 2.0, 4.0] or sorted(loads) == [2.0, 2.0, 2.0, 3.0]


def test_gfid_finish_applies_the_frozen_band(tmp_path):
    from scripts import b127_diamond_fidelity as fid
    band = {"tier": "E", "target": "t", "protocol": "p", "rule": "r", "episodes": 3, "seed": 1,
            "games": {"Pong": {"band": [18.963, 21.757], "paper_mean": 20.36, "paper_table1": 20.4,
                               "paper_seed_returns": [19.3, 20.9, 20.9, 20.7, 20.0]},
                      "Boxing": {"band": [63.676, 110.124], "paper_mean": 86.9, "paper_table1": 86.9,
                                 "paper_seed_returns": [99.4, 84.0, 97.5, 71.4, 82.2]}}}
    out = ROOT / ".local-artifacts" / "_test-issue-137-gfid"
    shutil.rmtree(out, ignore_errors=True)
    try:
        fid.write_json(out / "fidelity/band.json", band)
        for game, returns in (("Pong", [21.0, 20.0, 19.0]), ("Boxing", [100.0, 90.0, 95.0])):
            fid.write_json(out / f"fidelity/run/{game}.json", {"episodes": [
                {"index": i, "return": r, "length": 100, "seconds": 1.0} for i, r in enumerate(returns)]})
        assert fid.finish(out) == 0
        gfid = json.loads((out / "fidelity/gfid.json").read_text())
        assert gfid["schema"] == "issue_127_gfid_v1" and gfid["pass"] and gfid["games"]["Pong"]["ours_mean_return"] == 20
        fid.write_json(out / "fidelity/run/Pong.json", {"episodes": [
            {"index": i, "return": r, "length": 100, "seconds": 1.0} for i, r in enumerate([18.0, 19.0, 19.0])]})
        fid.finish(out)
        gfid = json.loads((out / "fidelity/gfid.json").read_text())
        assert not gfid["pass"] and not gfid["games"]["Pong"]["pass"] and gfid["games"]["Boxing"]["pass"]
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_frozen_band_is_the_paper_seed_spread():
    path = runner.OUTPUT / "fidelity/band.json"
    if not path.is_file():
        pytest.skip("band not frozen")
    band = json.loads(path.read_text())
    seeds = json.loads((dn.VENDOR / "results/data/DIAMOND.json").read_text())
    for game, entry in band["games"].items():
        values = seeds[game]
        mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
        assert entry["band"] == pytest.approx([mean - 2 * sd, mean + 2 * sd]) and entry["paper_seed_returns"] == values
    assert band["episodes"] == 100 and sorted(band["games"]) == ["Boxing", "Freeway", "Pong"]


FORBIDDEN = {"read_evaluation_anchors", "evaluation_cells", "position_cache", "read_novelty_anchors", "novelty_cells"}


@pytest.mark.parametrize("path", ["world_model/baselines/diamond/novphy.py", "scripts/b127_diamond_fidelity.py",
                                  "scripts/b127_diamond_port_reference.py"])
def test_no_evaluation_access(path):
    tree = ast.parse((ROOT / path).read_text())
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not names & FORBIDDEN
    assert not any("issue109-evaluation" in s for s in strings)
