"""#138 T6 V-JEPA 2-AC: CPU tests (vendoring, adapter shapes / causality, G-port record, gates, readiness logic and
the not-run publication, training resume / stop / typed failure, projection math, G-fid band logic, no evaluation
access). Run with CUDA_VISIBLE_DEVICES=""."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

from scripts import external_baselines as eb  # noqa: E402
from scripts import run_issue_138_vjepa2_ac as runner  # noqa: E402
from world_model.baselines.vjepa2 import novphy as vn  # noqa: E402

CLONE = Path("/mnt/array/sukaih/novphy-baselines-src/vjepa2@204698b")
GPORT = runner.OUTPUT / "gport"
TINY_PREDICTOR = {**vn.PREDICTOR, "depth": 2, "embed_dim": 32, "num_heads": 2}
TINY_DIM = 16


@pytest.fixture
def tiny(monkeypatch):
    """The official predictor class at a tiny width / depth, 16-d tokens, 3-update configurations of 4 clips."""
    monkeypatch.setattr(vn, "PREDICTOR", TINY_PREDICTOR)
    monkeypatch.setattr(vn, "EMBED_DIM", TINY_DIM)
    monkeypatch.setattr(vn, "MICRO_BATCH", 2)
    monkeypatch.setattr(runner, "CONFIGS", {name: {**c, "updates": 3, "batch": 4}
                                            for name, c in runner.CONFIGS.items()})


def tiny_h(b, frames, seed=0):
    g = torch.Generator().manual_seed(seed)
    return vn.normalize(torch.randn(b, frames * vn.TOKENS, TINY_DIM, generator=g))


class TinyClips(runner.ClipData):
    """ClipData over an in-memory fp16 store of 3 shots x 46 frames."""

    def __init__(self, seed=0):
        g = np.random.default_rng(seed)
        store = g.normal(size=(3 * len(vn.POSITIONS), vn.TOKENS, TINY_DIM)).astype(np.float16)
        table = np.arange(3 * len(vn.POSITIONS)).reshape(3, len(vn.POSITIONS))
        super().__init__(store, table, g.normal(size=(3, 5)))


# ---------------------------------------------------------------------------------------------------------------
# vendoring and the adapter
# ---------------------------------------------------------------------------------------------------------------

def test_vendor_is_verbatim_up_to_relative_imports():
    meta = json.loads((vn.VENDOR / "VENDOR.json").read_text())
    assert meta["schema"] == "issue_127_vendor_v1" and meta["commit"].startswith("204698b")
    assert (vn.VENDOR / "LICENSE").read_text().startswith("MIT License")
    for rel, entry in meta["files"].items():
        data = (vn.VENDOR / rel).read_bytes()
        assert entry["vendored_sha256"] == "sha256:" + hashlib.sha256(data).hexdigest(), rel
        if entry["upstream_path"] is None:
            assert entry["edits"] == ["package marker (new)"]
            continue
        assert all("package-relative import" in e for e in entry["edits"]), rel
        if not entry["edits"]:
            assert entry["upstream_sha256"] == entry["vendored_sha256"], rel
        if CLONE.is_dir():
            upstream = (CLONE / entry["upstream_path"]).read_text().splitlines()
            vendored = data.decode().splitlines()
            changed = [(a, b) for a, b in zip(upstream, vendored) if a != b]
            assert len(upstream) == len(vendored) and len(changed) == len(entry["edits"]), rel
            assert all(a.lstrip().startswith("from src.") and b.lstrip().startswith("from .") for a, b in changed)
    assert not any("randaugment" in r or "randerase" in r or "worker_init" in r for r in meta["files"])


def test_predictor_is_the_official_config():
    model = vn.build_predictor()
    assert vn.trainable_parameters(model) == 305_214_848  # 305 220 992 with the DROID 7-d action / state encoders
    assert len(model.predictor_blocks) == 24 and model.predictor_embed.in_features == vn.EMBED_DIM == 1408
    assert model.action_encoder.in_features == vn.LAUNCH_DIM and model.is_frame_causal and model.use_rope
    assert model.attn_mask.shape == (32 * (vn.TOKENS + 2),) * 2  # hub construction: 32-frame mask
    assert vn.NATIVE_UPDATES == 94_500 and vn.GLOBAL_BATCH == 256 and vn.GLOBAL_BATCH % vn.MICRO_BATCH == 0


def test_preprocess_is_the_official_fallback_transform():
    frames = torch.full((2, 3, 240, 320), 255, dtype=torch.uint8)
    frames[1] = 0
    x = vn.preprocess(frames)
    assert x.shape == (2, 3, 256, 256)
    mean, std = torch.tensor(vn.NORMALIZE[0]), torch.tensor(vn.NORMALIZE[1])
    assert torch.allclose(x[0, :, 0, 0], (1 - mean) / std, atol=1e-5)
    assert torch.allclose(x[1, :, 100, 100], -mean / std, atol=1e-5)


def test_clip_actions_launch_only_at_step_zero():
    launch = torch.arange(10, dtype=torch.float32).view(2, 5) + 1
    a = vn.clip_actions(launch, [0, 3])
    assert a.shape == (2, vn.CLIP_FRAMES - 1, 5)
    assert torch.equal(a[0, 0], launch[0]) and a[0, 1:].abs().sum() == 0 and a[1].abs().sum() == 0
    assert vn.clip_states(2).abs().sum() == 0 and vn.clip_states(2).shape == (2, vn.CLIP_FRAMES, 5)


def test_teacher_forcing_shapes_and_frame_causality(tiny):
    torch.manual_seed(0)
    model = vn.build_predictor().eval()
    h = tiny_h(2, 5)
    a, s = vn.clip_actions(torch.randn(2, 5), [0, 0], 5), vn.clip_states(2, 5)
    z_tf, z_ar = vn.forward_predictions(model, h, a, s, auto_steps=3)
    assert z_tf.shape == (2, 4 * vn.TOKENS, TINY_DIM) and z_ar.shape == (2, 3 * vn.TOKENS, TINY_DIM)
    h2 = h.clone()
    h2[:, 3 * vn.TOKENS:4 * vn.TOKENS] += 1.0  # perturb frame 3
    z2, _ = vn.forward_predictions(model, h2, a, s, auto_steps=3)
    t = vn.TOKENS
    assert torch.allclose(z_tf[:, :3 * t], z2[:, :3 * t], atol=1e-6)  # predictions of frames 1..3 ignore frame 3
    assert not torch.allclose(z_tf[:, 3 * t:], z2[:, 3 * t:])  # the prediction of frame 4 uses it
    a2 = a.clone()
    a2[:, 0] += 1.0
    z3, _ = vn.forward_predictions(model, h, a2, s, auto_steps=3)
    assert not torch.allclose(z_tf[:, :t], z3[:, :t])  # the launch conditions the first prediction
    loss, jloss, sloss = vn.clip_loss(model, h, a, s, 2)
    assert float(loss) == pytest.approx(float(jloss) + float(sloss))


def test_rollout_window_launch_and_normalisation(tiny):
    torch.manual_seed(0)
    model = vn.build_predictor().eval()
    seen = []
    forward = model.forward

    def spy(x, actions, states, extrinsics=None):
        seen.append((x.shape[1] // vn.TOKENS, actions.clone()))
        return forward(x, actions, states)
    model.forward = spy
    z0 = tiny_h(2, 1)
    launch = torch.randn(2, 5)
    out = vn.rollout(model, z0, launch, steps=10, keep=(1, 10))
    assert set(out) == {1, 10} and out[10].shape == (2, vn.TOKENS, TINY_DIM)
    assert [n for n, _ in seen] == [1, 2, 3, 4, 5, 6, 7, 7, 7, 7]  # sliding context of <= 7 frames
    for t, (n, acts) in enumerate(seen):
        first = t + 1 - n
        assert torch.equal(acts[:, 0], launch) if first == 0 else acts.abs().sum() == 0
        assert acts[:, 1:].abs().sum() == 0
    assert torch.allclose(out[10].mean(-1), torch.zeros(2, vn.TOKENS), atol=1e-5)  # layer-normed
    model.forward = forward
    other = vn.rollout(model, z0, launch * 2, steps=3, keep=(3,))
    assert not torch.allclose(other[3], vn.rollout(model, z0, launch, steps=3, keep=(3,))[3])


def test_accumulated_update_equals_the_full_batch_gradient(tiny):
    h, a, s = tiny_h(4, 3), vn.clip_actions(torch.randn(4, 5), [0, 1, 0, 2], 3), vn.clip_states(4, 3)
    grads = []
    for micro in ([(h, a, s)], [(h[:2], a[:2], s[:2]), (h[2:], a[2:], s[2:])]):
        torch.manual_seed(0)
        model = vn.build_predictor()
        opt, sch, wsch = vn.build_optimizer(model, {})
        captured = []
        step = opt.step

        def wrapped(*args, **kwargs):
            captured.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
            return step(*args, **kwargs)
        opt.step = wrapped
        terms = vn.train_step(model, opt, sch, wsch, micro, 2)
        grads.append(captured[0])
        assert terms["lr"] == pytest.approx(7.5e-5 + (4.25e-4 - 7.5e-5) / 4500) and terms["wd"] == 0.04
    assert set(grads[0]) == set(grads[1]) and not any(n.startswith("extrinsics") for n in grads[0])
    assert all(torch.allclose(grads[0][n], grads[1][n], rtol=1e-4, atol=1e-7) for n in grads[0])


def test_optimizer_groups_follow_init_opt(tiny):
    model = vn.build_predictor()
    opt, sch, wsch = vn.build_optimizer(model, {"optimization": {"lr": 2.125e-4, "start_lr": 3.75e-5}})
    decay, no_decay = opt.param_groups
    assert no_decay["WD_exclude"] and no_decay["weight_decay"] == 0 and "WD_exclude" not in decay
    assert sum(p.numel() for g in opt.param_groups for p in g["params"]) == vn.trainable_parameters(model)
    assert sch.warmup_steps == 4500 and sch.anneal_steps == 4500 and sch.ref_lr == 2.125e-4


def test_clip_batches_are_deterministic_and_bounded(tiny):
    data = TinyClips()
    spec = runner.config_spec("native")
    a, b = data.batch(7, 3, spec), data.batch(7, 3, spec)
    assert len(a) == spec["batch"] // spec["micro_batch"] == 2
    assert all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in zip(a, b))
    assert a[0][0].shape == (2, vn.CLIP_FRAMES * vn.TOKENS, TINY_DIM)
    shots, starts = data.draw(7, 3, 1000)
    assert starts.min() >= 0 and starts.max() == runner.CLIP_STARTS - 1 == 38 and shots.max() < 3


def test_fit_layout_clamps_to_the_shot_end():
    from scripts import run_hjepa_comparison as rc
    assert runner.training_positions(100) == list(range(0, 100, 5)) + [100]
    assert runner.training_positions(300) == list(vn.POSITIONS)
    first, positions = runner.fit_layout([100, 300])
    fi = runner.FitIndex(first, positions)
    table = fi.clip_table([100, 300])
    assert table.shape == (2, 46) and len(set(table[0, 20:])) == 1  # frames after the end repeat it
    assert set(rc.probe_positions(100)) <= set(positions[:first[1]].tolist())
    with pytest.raises(KeyError):
        fi.rows([0], [7])


# ---------------------------------------------------------------------------------------------------------------
# G-port record
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not (GPORT / "gport.json").is_file(), reason="G-port not run")
def test_gport_record_passes_for_the_current_adapter():
    record = eb.read_json(GPORT / "gport.json")
    assert record["schema"] == "issue_127_gport_v1" and record["pass"]
    assert set(record["cases"]) == {"native", "ar3"} and record["cases"]["ar3"]["exact"]
    for case in record["cases"].values():  # training path bit-exact; the rollout within 1e-5 (rule)
        assert case["forward"]["loss_rel"] == 0 and all(s["param_max_abs"] == 0 for s in case["steps"])
    assert record["cases"]["native"]["rollout"]["max_abs"] <= record["tolerance"]["feature_max_abs"]
    assert record["preprocess"]["equal"] and record["features"]["equal"]
    assert record["released_predictor"]["max_abs"] == 0.0 and record["cases"]["native"]["rollout"]["steps"] == 7
    assert all(s["lr_equal"] and s["wd_equal"] for c in record["cases"].values() for s in c["steps"])
    assert record["torch"][0] == record["torch"][1]
    assert record["adapter_sha256"] == "sha256:" + hashlib.sha256(
        (ROOT / "world_model/baselines/vjepa2/novphy.py").read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# gates, readiness, not-run publication
# ---------------------------------------------------------------------------------------------------------------

def test_funding_and_gates_refuse_every_gpu_mode(tmp_path):
    out = tmp_path
    for fn in (lambda: runner.selection(out, 0, "cpu"), lambda: runner.selection_readouts(out, "cpu"),
               lambda: runner.train(out, 0, "cpu"), lambda: runner.readouts(out, 0, "cpu"),
               lambda: runner.score(out, "cpu"), lambda: runner.score_novelty(out, "cpu"),
               lambda: runner.features(out, "training", "cpu")):
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
    eb.write_json(out / "readiness/readiness.json", {"pass": True})
    eb.write_json(out / "fidelity/gfid.json", {"pass": False})
    assert not runner.selection_gate(out)["pass"]
    eb.write_json(out / "fidelity/gfid.json", {"pass": True})
    assert runner.selection_gate(out)["pass"]
    assert not eb.evaluation_anchors_path(out).exists()


def test_readiness_threshold_rule():
    assert runner.readiness_decision(0.6499)["typed_reason"] == "readiness_ceiling_below_threshold"
    assert runner.readiness_decision(None)["pass"] is False
    assert runner.readiness_decision(0.65) == {"pass": True, "typed_reason": None,
                                               "consequence": "T6 Phase 0 continues"}
    assert runner.THRESHOLD == eb.GUARD["readout_ceiling_auc_min"] == 0.65


class FakeShots:
    def __init__(self):
        g = np.random.default_rng(0)
        self.ends = np.array([225, 100])
        self.keys = ["s0", "s1"]
        self.launch5 = g.normal(size=(2, 5))
        self._carriers = torch.as_tensor(g.normal(size=(2, 301, 236)), dtype=torch.float32)

    def __len__(self):
        return 2

    def carrier_rows(self, shots, positions):
        return self._carriers[np.asarray(shots), np.asarray(positions)]


def _readiness_setup(tmp_path, monkeypatch, ceiling):
    """The real readiness() over a tiny feature cache (2 fit shots, 16-d tokens); the policy ceiling is stubbed."""
    from world_model.baselines.dino_wm import novphy as dn
    monkeypatch.setattr(vn, "EMBED_DIM", TINY_DIM)
    monkeypatch.setattr(runner, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(eb, "FitShots", FakeShots)
    monkeypatch.setattr(runner, "policy_layout", lambda: [("anchor", "m0", 0), ("endpoint", "k0", 225)])
    monkeypatch.setitem(dn.R1, "epochs", 1)
    monkeypatch.setitem(dn.R1, "batch", 4)
    runner.plan_features(tmp_path, FakeShots())
    p = runner.feature_paths(tmp_path)
    p["fit"].parent.mkdir(parents=True)
    rows = int(np.load(p["fit_index"])["first"][-1])
    g = np.random.default_rng(1)
    for kind, n in (("fit", rows), ("policy", 2)):
        mm = np.memmap(p[kind], dtype=np.float16, mode="w+", shape=(n, vn.TOKENS, TINY_DIM))
        mm[:] = g.normal(size=mm.shape)
        mm.flush()
    eb.write_json(p["manifest"], {"schema": runner.SCHEMA_FEATURES, "phases": {"readiness": {
        "files": {}, "indexes": {k: {"path": str(p[k]), "sha256": eb.sha256_of(p[k])}
                                 for k in ("fit_index", "policy_index", "plan")}}}})
    monkeypatch.setattr(runner, "objective_costs", lambda: None)
    monkeypatch.setattr(runner, "ceiling_from", lambda probe, costs, store, items: {
        "ceiling_auc": ceiling, "policy_grid_cells": 94, "policy_candidates": 1504})
    monkeypatch.setattr(runner, "e99_true_ceiling", lambda costs: 0.7223)


def test_readiness_stop_publishes_the_not_run_summary(tmp_path, monkeypatch):
    _readiness_setup(tmp_path, monkeypatch, 0.5948)
    assert runner.readiness(tmp_path, "cpu") == 0
    record = eb.read_json(tmp_path / "readiness/readiness.json")
    assert record["pass"] is False and record["typed_reason"] == runner.TYPED_STOP and record["threshold"] == 0.65
    assert record["r1"]["fit_frames"] == len(runner.r1_items(FakeShots().ends)) and record["r1"]["epochs"] == 1
    assert record["data"]["evaluation"] == "not read"
    assert runner.readiness(tmp_path, "cpu") == 0  # one run, never re-run
    probe, payload = runner.load_r1(tmp_path)
    assert payload["fit"]["samples"] == record["r1"]["fit_frames"]
    assert runner.not_run_reason(tmp_path) == runner.TYPED_STOP
    assert runner.selection(tmp_path, 0, "cpu") == 1
    assert runner.publish(tmp_path) == 0
    summary = eb.load_child_summary(tmp_path / "summary.json")
    assert summary["arms"] == {} and summary["extra"]["pending_arms"] == {"T6": runner.TYPED_STOP}
    assert summary["extra"]["readiness"]["ceiling_auc"] == 0.5948 and summary["frozen_at"] is None
    assert "readiness_ceiling_below_threshold" in (tmp_path / "findings.md").read_text()


def test_readiness_pass_continues_and_cap_marker_publishes(tmp_path, monkeypatch):
    _readiness_setup(tmp_path, monkeypatch, 0.70)
    assert runner.readiness(tmp_path, "cpu") == 0
    assert runner.readiness_pass(tmp_path) and runner.not_run_reason(tmp_path) is None
    with pytest.raises(SystemExit):
        runner.publish(tmp_path)  # neither a plan nor a typed reason
    eb.write_json(tmp_path / "not-run.json", {"typed_reason": "cap", "decision": "url"})
    projection = {"primary": "projection_upper", "projection_upper": {"total_gpu_h": 1234.0}}
    eb.write_json(tmp_path / "projection.json", projection)
    assert runner.publish(tmp_path) == 0
    summary = eb.read_json(tmp_path / "summary.json")
    assert summary["extra"]["pending_arms"] == {"T6": "cap"}
    assert summary["extra"]["projection"]["total_gpu_h"] == 1234.0 and summary["extra"]["projection"]["primary"] == \
        "projection_upper" and summary["extra"]["projection"]["remainder_gpu_h"] == 73.0
    assert "1234 GPU-h" in (tmp_path / "findings.md").read_text()


# ---------------------------------------------------------------------------------------------------------------
# training: resume determinism, stop signal, typed failure
# ---------------------------------------------------------------------------------------------------------------

def test_train_run_resume_is_bitwise_identical(tmp_path, tiny):
    data = TinyClips()
    spec = runner.config_spec("ar4")
    whole = runner.train_run(tmp_path / "whole", spec, 7, data, "cpu")
    part = runner.train_run(tmp_path / "split", spec, 7, data, "cpu", max_steps=1)
    assert part["partial"] and (tmp_path / "split/resume.pt").is_file()
    runner.train_run(tmp_path / "split", spec, 7, data, "cpu")
    a = torch.load(tmp_path / "whole/model.pt", weights_only=False)
    b = torch.load(tmp_path / "split/model.pt", weights_only=False)
    assert whole["complete"] and a["complete"] and b["complete"] and not (tmp_path / "split/resume.pt").exists()
    assert all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
    assert a["spec"]["auto_steps"] == 4 and whole["steps"] == 3


def test_stop_signal_pauses_with_exit_75(tmp_path, tiny, monkeypatch):
    monkeypatch.setattr(eb, "STOP_REQUESTED", [15])
    monkeypatch.setattr(eb, "install_stop_handlers", lambda: None)
    item = (runner.config_spec("native"), 7, tmp_path / "run")
    assert runner.run_queue(tmp_path, [item], "cpu", TinyClips()) == eb.EXIT_PAUSED
    assert (tmp_path / "run/resume.pt").is_file() and not (tmp_path / "run/model.pt").exists()
    assert (tmp_path / "ledger").is_dir()


def test_nonfinite_loss_is_a_typed_terminal_failure(tmp_path, tiny, monkeypatch):
    monkeypatch.setattr(vn, "train_step", lambda *a, **k: {"loss": math.nan, "jloss": math.nan, "sloss": math.nan,
                                                           "lr": 0.0, "wd": 0.0})
    report = runner.train_run(tmp_path, runner.config_spec("native"), 7, TinyClips(), "cpu")
    assert report["complete"] is False and report["typed_failure"]["typed_failure"] == "nonfinite_loss"
    model, payload = runner.load_trained(tmp_path / "model.pt")
    assert model is None and payload["state_dict"] is None


# ---------------------------------------------------------------------------------------------------------------
# projection math, G-fid band logic, no evaluation access
# ---------------------------------------------------------------------------------------------------------------

def test_projection_math_and_measured_projection():
    rates = {"native": 2.0, "ar4": 3.0, "lr-half": 2.0}
    rows = {"readiness": 1000, "training_only": 3000}
    p = runner.child_projection(rates, 5.0, 0.01, 0.1, 0.5, rows)
    n = vn.NATIVE_UPDATES
    assert p["selection_gpu_h"] == pytest.approx({"native": n * 2 / 3600, "ar4": n * 3 / 3600, "lr-half": n * 2 / 3600})
    assert p["training_gpu_h"] == pytest.approx(2 * n * 3 / 3600)  # two seeds at the most expensive config
    assert p["features_gpu_h"] == pytest.approx(4000 * 0.01 / 3600)
    assert p["readiness_gpu_h"] == pytest.approx(runner.R1_STEPS * 0.1 / 3600)
    other = (3 * runner.POLICY_DECISIONS + 3 * runner.READOUT_DECISIONS + 3 * runner.SCORING_DECISIONS) * 5 / 3600
    assert p["total_gpu_h"] == pytest.approx(sum(p["selection_gpu_h"].values()) + p["training_gpu_h"] + other
                                             + p["features_gpu_h"] + p["readiness_gpu_h"] + 0.5 + 1.0)
    assert p["training_queues"] == {0: ["T6--seed20260909"], 1: ["T6--seed20260910"], 2: [], 3: []}
    assert p["fits_remainder"] is (p["total_gpu_h"] <= 73) and p["within_caps"] is False  # 78.75 GPU-h per seed
    record = {"configs": {k: {"seconds_per_update_steady_median": v} for k, v in rates.items()},
              "scoring": {"seconds_per_decision": 5.0}, "encoder": {"seconds_per_frame": 0.01},
              "r1": {"seconds_per_step_median": 0.1}}
    m = runner.measured_projection(record, rows)
    assert m["source"].startswith("timing.json") and m["seconds_per_update"]["native"] == 2.0
    slow_read = runner.measured_projection({**record, "configs": {k: {"seconds_per_update_steady_median": 0.01}
                                                                 for k in rates}}, rows)
    read_bound = vn.GLOBAL_BATCH * vn.CLIP_FRAMES * runner.FRAME_BYTES / 3e9
    assert slow_read["seconds_per_update"]["native"] == pytest.approx(read_bound)  # the cache read bound binds
    queues, loads = runner.lpt([("a", 3.0), ("b", 2.0), ("c", 2.0), ("d", 1.0), ("e", 1.0)])
    assert sorted(loads) == [2.0, 2.0, 2.0, 3.0]


def test_gfid_band_and_finish(tmp_path):
    from scripts import b127_vjepa2_fidelity as fid
    band = fid.band_record()
    se = 100 * math.sqrt(0.89 * 0.11 / 1970)
    assert band["band_pct"] == pytest.approx([89.0 - 2 * se - 0.05, 89.0 + 2 * se + 0.05])
    frozen = runner.OUTPUT / "fidelity/band.json"
    if frozen.is_file():
        assert json.loads(frozen.read_text())["band_pct"] == pytest.approx(band["band_pct"])
    fid.write_json(tmp_path / "fidelity/band.json", band)
    log = tmp_path / "fidelity/run/video_classification_frozen" / fid.TAG / "log_r0.csv"
    log.parent.mkdir(parents=True)
    fid.write_json(tmp_path / "fidelity/run/run.json", {"exit_code": 0})
    for acc, expected in ((88.9, True), (87.0, False)):
        log.write_text(f"epoch,train_acc,val_acc\n1,-1.00000,{acc:.5f}\n")
        assert fid.finish(tmp_path) == 0
        gfid = json.loads((tmp_path / "fidelity/gfid.json").read_text())
        assert gfid["schema"] == "issue_127_gfid_v1" and gfid["pass"] is expected and gfid["runs"][0]["top1_pct"] == acc


FORBIDDEN = {"read_evaluation_anchors", "evaluation_cells", "position_cache", "read_novelty_anchors", "novelty_cells"}


@pytest.mark.parametrize("path", ["world_model/baselines/vjepa2/novphy.py", "scripts/b127_vjepa2_fidelity.py",
                                  "scripts/b127_vjepa2_port_reference.py"])
def test_no_evaluation_access(path):
    tree = ast.parse((ROOT / path).read_text())
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not names & FORBIDDEN
    assert not any("issue109-evaluation" in s for s in strings)
