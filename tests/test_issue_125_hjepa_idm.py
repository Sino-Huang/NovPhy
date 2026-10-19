"""#125 (H-JEPA + inverse dynamics), CPU only: the IDM / action-SIGReg port, the matched-parameter rule, bitwise
SIGTERM resume, the G-port-IDM record, the trigger's guard code, the not-triggered exits, the gamma_1 rule, the
dispositions and the scheduler jobs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from scripts import gpu_window as gw
from scripts import run_hjepa_comparison as rc
from scripts import run_hjepa_idm as ri
from world_model.hjepa import idm
from world_model.hjepa import novphy as hn

ROOT = Path(__file__).resolve().parents[1]
ARM, SEED, LAM, GAMMA1, BATCH, STEPS = "HJ-car-m-IDM", 20260908, 0.18, 100.0, 4, 8


def forward(arm, crop_starts, seed=0, launch_clips=(0, 2)):
    torch.manual_seed(seed)
    model, losses, cfg = idm.build_model(arm, LAM, GAMMA1, checkpointing=False)
    geometry = hn.Geometry(arm)
    starts = {level: s for level, s in enumerate(crop_starts, start=1)}
    needed = geometry.needed(starts)
    model.plan = (geometry.level1_steps, starts, needed)
    wrapper = hn.Wrapper(model, losses).train()
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(BATCH, len(needed), hn.CARRIER_DIM, generator=g)
    action = torch.zeros(BATCH, geometry.level1_steps, hn.ACTION_DIM)
    for clip in launch_clips:
        action[clip, 0] = torch.tensor([-0.25, 0.1, 1.0, 0.0, 1.0])
    out = hn.hjepa_utils.hjepa_forward(wrapper, {"pixels_level1": x, "action_level1": action}, "fit", cfg,
                                       normalize_batch=hn.normalize_uint8_images_on_device)
    return model, cfg, out, action


def test_idm_heads_coefficients_and_action_sigreg_follow_the_official_layout():
    for arm in idm.ARMS:
        model, losses, cfg = idm.build_model(arm, LAM, GAMMA1, checkpointing=False)
        levels = int(cfg.num_levels)
        assert levels == hn.num_levels(idm.BASES[arm][0])
        for level in range(1, levels + 1):
            level_cfg = cfg[f"level{level}"]
            assert level_cfg.idm_coeff == (GAMMA1 if level == 1 else 50.0)
            assert level_cfg.get("action_sigreg_coeff", 0.0) == (0.0 if level == 1 else 0.005)
            head = model.get_level(level).idm_loss_fn.idm.model
            D = model.embed_dims[level]
            assert [type(m).__name__ for m in head] == ["Linear", "ReLU", "Linear", "ReLU", "Linear"]
            assert (head[0].in_features, head[0].out_features, head[2].out_features, head[4].out_features) == (2 * D, 256, 256, D)
        assert sorted(k for k in losses if k.startswith("sigreg_action")) == [f"sigreg_action_level{l}" for l in range(2, levels + 1)]
    # the backbone starts exactly where the #123 arm starts with the same seed
    torch.manual_seed(SEED)
    base, _ = hn.build_model("HJ-px", LAM, checkpointing=False)
    torch.manual_seed(SEED)
    model, _, _ = idm.build_model("HJ-px-IDM", LAM, GAMMA1, checkpointing=False)
    backbone, heads = idm.split_state(model.state_dict())
    assert heads and backbone.keys() == base.state_dict().keys()
    assert all(torch.equal(backbone[k], v) for k, v in base.state_dict().items())


def test_matched_parameter_rule_counts_the_heads_and_is_nearest():
    for arm, size in (("HJ-car-m-IDM", "matched-idm"), ("LeWM-car-m-IDM", "matched-lewm-idm")):
        counts = idm.parameter_counts(arm)
        assert counts["total"] == counts["backbone"] + counts["idm_heads"] and counts["idm_heads"] > 0
        assert idm.matched(counts["total"]), counts
        # the next predictor MLP widths (+-16) are farther from the unit
        width = hn.SIZES[size]["level1_predictor"]["mlp_dim"]
        error = abs(counts["total"] - idm.GATE_B_UNIT_PARAMETERS)
        for other in (width - 16, width + 16):
            probe = {**hn.SIZES[size], "level1_predictor": {**hn.SIZES[size]["level1_predictor"], "mlp_dim": other}}
            if hn.SIZES[size]["upper_predictor"] is not None:
                probe["upper_predictor"] = {**hn.SIZES[size]["upper_predictor"], "mlp_dim": other}
            hn.SIZES["_probe"] = probe
            hn.ARMS["_probe"] = {**hn.ARMS[arm], "size": "_probe"}
            try:
                assert abs(idm.parameter_counts("_probe")["total"] - idm.GATE_B_UNIT_PARAMETERS) > error
            finally:
                del hn.SIZES["_probe"], hn.ARMS["_probe"]
    # the #123 matched arm without heads is unchanged
    assert hn.parameter_count("HJ-car-m") == 1855172


def test_idm_loss_is_the_definition_on_embedded_launch_targets():
    """L_idm = mean ||I(z_t, z_t+1) - sg[a_t]||^2 per level; the level-1 target of the launch transition is the
    embedded launch, elsewhere the embedded no-op; levels >= 2 use the pooled macro-action embedding."""
    model, cfg, out, action = forward(ARM, (0, 0, None))
    for level in range(1, int(cfg.num_levels) + 1):
        emb, act = out[f"embed_{level}"], out[f"action_{level}"]
        head = model.get_level(level).idm_loss_fn.idm
        pred = head(emb[:, :-1].flatten(0, 1), emb[:, 1:].flatten(0, 1))
        expected = (pred - act[:, :emb.size(1) - 1].flatten(0, 1)).square().mean()
        assert torch.allclose(out[f"idm_loss_level{level}"], expected, rtol=0, atol=1e-7)
        weighted = out[f"embed_loss_level{level}"] + cfg[f"level{level}"].idm_coeff * out[f"idm_loss_level{level}"]
        if level > 1:
            weighted = weighted + 0.005 * out[f"sigreg_loss_action_level{level}"]
        assert torch.allclose(out[f"loss_level{level}"], weighted, rtol=1e-6)
    embedder = model.get_level(1).action_embed
    with torch.no_grad():
        launch = embedder(action[0:1, 0:1])[0, 0]
        noop = embedder(torch.zeros(1, 1, hn.ACTION_DIM))[0, 0]
    close = lambda a, b: torch.allclose(a, b, rtol=0, atol=1e-6)  # noqa: E731  (batched vs single: float round-off)
    assert close(out["action_1"][0, 0], launch) and close(out["action_1"][1, 0], noop)
    assert close(out["action_1"][0, 1], noop) and not close(launch, noop)


def test_idm_target_is_detached():
    """The IDM target carries no gradient: the IDM loss alone leaves the action embedder untouched."""
    model, cfg, out, _ = forward(ARM, (0, 0, None))
    model.zero_grad(set_to_none=True)
    out["idm_loss_level1"].backward(retain_graph=True)
    embedder = list(model.get_level(1).action_embed.parameters())
    assert all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in embedder)
    encoder = list(model.get_level(1).encoder.parameters())
    assert any(p.grad is not None and float(p.grad.abs().max()) > 0 for p in encoder)
    head = list(model.get_level(1).idm_loss_fn.parameters())
    assert all(p.grad is not None for p in head)


class SyntheticFit:
    """FitData stand-in: fixed seeded carriers and launches for 6 shots."""

    def __init__(self):
        generator = torch.Generator().manual_seed(7)
        self.ends = np.array([285, 200, 150, 285, 240, 285], dtype=np.int64)
        self.table = torch.randn(len(self.ends), 286, 236, generator=generator)
        self.actions = torch.rand(len(self.ends), hn.ACTION_DIM, generator=generator)

    def carrier_rows(self, shots, positions):
        return self.table[torch.as_tensor(shots), torch.as_tensor(positions)]


class SyntheticSource(rc.BatchSource):
    def __init__(self):
        self.hn = hn
        self.output, self.arm, self.seed, self.steps, self.batch = None, ARM, SEED, STEPS, BATCH
        self.spec = hn.ARMS[ARM]
        self.geometry = hn.Geometry(ARM)
        self._data = SyntheticFit()
        self.ends = self._data.ends
        valid = np.array([self.geometry.valid_starts(int(e)) for e in self.ends], dtype=np.int64)
        self.cumulative = np.concatenate([[0], np.cumsum(valid)])


@pytest.fixture
def stop_after(monkeypatch):
    """stop_after(n): SIGTERM to this process after the n-th optimizer step of the next train_run call."""
    previous = {s: signal.signal(s, rc.request_stop) for s in (signal.SIGTERM, signal.SIGUSR1)}
    official = hn.train_step
    budget = {"calls": None}

    def counted(*args, **kwargs):
        stats = official(*args, **kwargs)
        if budget["calls"] is not None:
            budget["calls"] -= 1
            if budget["calls"] == 0:
                os.kill(os.getpid(), signal.SIGTERM)
        return stats

    monkeypatch.setattr(hn, "train_step", counted)

    def arm(n):
        rc.STOP_REQUESTED.clear()
        budget["calls"] = n

    yield arm
    rc.STOP_REQUESTED.clear()
    for s, handler in previous.items():
        signal.signal(s, handler)


def run(out_dir):
    return ri.train_run(None, ARM, SEED, LAM, GAMMA1, STEPS, BATCH, out_dir, device="cpu", source=SyntheticSource())


def assert_identical(a, b, path="state"):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape, path
        assert torch.equal(a, b), f"{path} differs"
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            assert_identical(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_identical(x, y, f"{path}[{i}]")
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def test_sigterm_stop_and_resume_is_bitwise_identical(tmp_path, stop_after):
    n, k = 6, 2
    stop_after(n)
    with pytest.raises(rc.TrainingPaused):
        run(tmp_path / "continuous")
    continuous = torch.load(tmp_path / "continuous" / "resume.pt", weights_only=False)
    stop_after(k)
    with pytest.raises(rc.TrainingPaused):
        run(tmp_path / "interrupted")
    assert torch.load(tmp_path / "interrupted" / "resume.pt", weights_only=False)["step"] == k
    stop_after(n - k)
    with pytest.raises(rc.TrainingPaused):
        run(tmp_path / "interrupted")
    interrupted = torch.load(tmp_path / "interrupted" / "resume.pt", weights_only=False)
    assert continuous["step"] == interrupted["step"] == n
    assert any(idm.is_head(key) for key in continuous["model"])
    for key in ("model", "optimizer", "scheduler", "cpu_rng"):
        assert_identical(continuous[key], interrupted[key], key)
    stop_after(None)
    reports = [run(tmp_path / name) for name in ("continuous", "interrupted")]
    assert all(r["complete"] for r in reports)
    final = [torch.load(tmp_path / name / "model.pt", weights_only=False) for name in ("continuous", "interrupted")]
    for key in ("state_dict", "idm_heads"):
        assert_identical(final[0][key], final[1][key], f"model.pt {key}")
    assert not (tmp_path / "continuous" / "resume.pt").exists()
    # the backbone loads through #123's loader; the full model (heads included) through this ticket's
    model, payload = rc.load_trained(tmp_path / "continuous" / "model.pt", device="cpu")
    assert payload["gamma1"] == GAMMA1 and not any(idm.is_head(k) for k in model.state_dict())
    full, _ = ri.load_trained_idm(tmp_path / "continuous" / "model.pt", device="cpu")
    assert_identical(idm.split_state(full.state_dict())[1], final[0]["idm_heads"], "heads")


def test_stop_between_runs_exits_paused(monkeypatch, tmp_path):
    plan = {"training": {"queues": [[f"{ARM}/20260908", f"{ARM}/20260909"]]},
            "arms": {ARM: {"lambda": LAM, "gamma1": GAMMA1, "steps": STEPS, "batch": BATCH}}}
    calls = []
    monkeypatch.setattr(ri, "load_plan", lambda output: plan)
    monkeypatch.setattr(ri, "mark", lambda *a, **k: None)

    def fake_run(output, arm, seed, *args):
        calls.append(seed)
        os.kill(os.getpid(), signal.SIGTERM)
        return {"wall_seconds": 1.0, "gpu_seconds": 1.0}

    monkeypatch.setattr(ri, "train_run", fake_run)
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGUSR1)}
    rc.STOP_REQUESTED.clear()
    try:
        assert ri.train(tmp_path, 0) == rc.EXIT_PAUSED
    finally:
        rc.STOP_REQUESTED.clear()
        for s, handler in previous.items():
            signal.signal(s, handler)
    assert calls == [20260908]


GPORT = ri.OUTPUT / "gport-idm.json"


def test_gport_idm_passed():
    """IDM loss and head, action-SIGReg and the positive-LR step equal the official code (CPU float32) within
    #123's declared tolerances, for all four IDM arms."""
    if not GPORT.is_file():
        pytest.skip("gport-idm.json not produced yet")
    result = json.loads(GPORT.read_text())
    assert result["pass"] and result["tolerance"] == rc.GPORT_TOLERANCE and all(result["vendor_hashes_match"].values())
    assert set(result["arms"]) == set(idm.ARMS)
    for arm, r in result["arms"].items():
        assert r["pass"] and r["step2"]["pass"] and r["idm_terms_present"] and r["head_grad_tensors"] > 0, arm
        assert min(r["step2"]["lrs_at_step"]) > 0 and r["step2"]["parameters_moved"], arm


def test_gport_idm_detects_a_wrong_idm_coefficient(monkeypatch):
    path = ri.gport_dir(ri.OUTPUT) / f"reference-{ARM}.pt"
    if not path.is_file():
        pytest.skip("G-port-IDM reference not produced yet")
    reference = torch.load(path, map_location="cpu", weights_only=False)
    assert ri.gport_arm(reference)["pass"]
    monkeypatch.setattr(ri, "GPORT_GAMMA1", 225.0)
    assert not ri.gport_arm(reference)["pass"]
    monkeypatch.setattr(ri, "GPORT_GAMMA1", 100.0)
    monkeypatch.setattr(idm, "ACTION_SIGREG", 0.01)
    assert not ri.gport_arm(reference)["pass"]


def readout_report(rank, chi, sensitivity, dim=112):
    return {"typed_failure": None, "collapse": [{"level": 1, "effective_rank": 50.0, "within_episode_share": 0.5, "dim": dim},
                                                {"level": 3, "effective_rank": rank, "within_episode_share": chi, "dim": dim}],
            "action_sensitivity": sensitivity, "policy_auc": {"ceiling": 0.7, "R1": 0.5}}


CASES = [(11.2, 0.05, 0.05), (11.19, 0.2, 0.2), (40.0, 0.0499, 0.2), (40.0, 0.2, 0.0499), (14.2, 0.017, 1.7e-7),
         (60.0, 0.3, 0.3)]


def test_trigger_guards_are_123_guard_code(tmp_path):
    """seed_guards equals rc.arm_guards seed by seed, at and around every threshold."""
    for i, (rank, chi, sens) in enumerate(CASES):
        out = tmp_path / str(i)
        for seed in rc.SEEDS:
            run = rc.run_dir(out, "HJ-car-m", seed)
            run.mkdir(parents=True)
            rc.write_json(run / "readouts.json", readout_report(rank, chi, sens))
        records = {("HJ-car-m", s): {"typed_failure": None, "inventories": {"grid": {}}} for s in rc.SEEDS}
        theirs = rc.arm_guards(out, {"arms": {"HJ-car-m": {"parameters": 1}}}, records)["HJ-car-m"]["seeds"]
        mine = ri.seed_guards(readout_report(rank, chi, sens))
        for seed in rc.SEEDS:
            assert (mine["G_collapse"], mine["G_action"]) == (theirs[str(seed)]["G_collapse"], theirs[str(seed)]["G_action"])


def test_trigger_decision_any_failing_seed_and_typed_failure():
    good = {**readout_report(60.0, 0.3, 0.3)}
    parts = {(arm, s): good for arm in ri.TRIGGER_ARMS for s in rc.SEEDS}
    arms, failing, fired = ri.trigger_decision(parts)
    assert not fired and not failing and arms["HJ-px"][str(rc.SEEDS[0])]["pass"]
    _, failing, fired = ri.trigger_decision({**parts, ("HJ-px", rc.SEEDS[2]): readout_report(60.0, 0.3, 4.7e-7, dim=192)})
    assert fired and failing == [f"HJ-px/{rc.SEEDS[2]}"]
    _, failing, fired = ri.trigger_decision({**parts, ("HJ-car-m", rc.SEEDS[1]): {"typed_failure": {"typed_failure": "nonfinite_loss"}}})
    assert fired and failing == [f"HJ-car-m/{rc.SEEDS[1]}"]


def test_trigger_never_uses_resume_pt(tmp_path, monkeypatch):
    run = rc.run_dir(tmp_path, "HJ-px", rc.SEEDS[0])
    run.mkdir(parents=True)
    torch.save({"step": 5}, run / "resume.pt")
    monkeypatch.setattr(ri, "SOURCE", tmp_path)
    with pytest.raises(ri.TriggerNotReady):
        ri.trigger_part("HJ-px", rc.SEEDS[0], data=None, device="cpu")


def test_not_triggered_stages_exit_zero_with_a_record(tmp_path):
    rc.write_json(tmp_path / "trigger.json", {"triggered": False})
    assert ri.selection(tmp_path, 1, 3) == 0
    assert ri.smoke(tmp_path) == 0
    for stage in ("selection-slot1", "smoke"):
        assert json.loads((tmp_path / f"{stage}.json").read_text())["status"] == ri.NOT_TRIGGERED
    assert not (tmp_path / "selection").exists()
    with pytest.raises(ValueError, match="not triggered"):
        ri.prepare(tmp_path)


def test_gamma1_selection_rule(tmp_path):
    """Largest policy grid AUC under R1; a tie at 4 d.p. goes to 100 (DROID)."""

    def write(arm, gamma1, auc):
        run = ri.selection_dir(tmp_path, arm, gamma1)
        run.mkdir(parents=True, exist_ok=True)
        rc.write_json(run / "readouts.json", {"typed_failure": None, "policy_auc": {"R1": auc}})

    write("HJ-px-IDM", 100.0, 0.51231)
    write("HJ-px-IDM", 225.0, 0.51229)
    write("HJ-car-m-IDM", 100.0, 0.50), write("HJ-car-m-IDM", 225.0, 0.53)
    result = ri.selection_result(tmp_path)
    assert result["pixels"]["selected_gamma1"] == 100.0 and result["carriers"]["selected_gamma1"] == 225.0
    assert {r for slot in range(3) for i, r in enumerate(ri.selection_runs()) if i % 3 == slot} == set(ri.selection_runs())


def grid(tokens, reverse, wrong):
    return {"contrasts": {k: {"token": tokens.get(k, rc.STOP_TOKEN), "reverse_token": reverse.get(k, rc.STOP_TOKEN)}
                          for k in ri.CONTRASTS},
            "SI8_wrong_anchor": {arm: {"token": wrong.get(arm, rc.STOP_TOKEN)} for arm in idm.PRIMARY}}


def test_dispositions():
    ok = {arm: {"G_collapse": True, "G_action": True} for arm in idm.PRIMARY}
    d = ri.dispositions(grid({"PI1": "supported"}, {}, {"HJ-px-IDM": "supported"}), ok)
    assert d["disposition"] == 1 and d["arms"] == ["HJ-px-IDM"]
    assert ri.dispositions(grid({"PI1": "supported"}, {}, {}), ok)["disposition"] == 3
    d = ri.dispositions(grid({}, {"PI1": "supported", "PI2": "supported"}, {}), ok)
    assert d["disposition"] == 2 and "IDM + action-SIGReg remedy" in d["scope"]
    still = {**ok, "HJ-car-m-IDM": {"G_collapse": False, "G_action": True}}
    assert ri.dispositions(grid({}, {}, {}), still)["disposition_4_remedy_does_not_prevent_collapse"]


def test_failed_guard_turns_every_involved_token_into_the_stop_token():
    rows = [{"member": f"m{m}", "seed": s, "views": {name: {"auc": 0.9 if name == "HJ-px-IDM" else 0.5}
                                                     for name in ("U", "C*", "P", "J", "J_dec", *idm.ARMS, "HJ-px", "HJ-car-m")}}
            for m in range(6) for s in rc.SEEDS]
    passing = {arm: {"pass": True} for arm in (*idm.ARMS, *ri.ISSUE_123_ARMS)}
    good = ri.grid_contrasts(rows, passing, True)
    assert good["PI1"]["token"] == "supported"
    bad = ri.grid_contrasts(rows, {**passing, "HJ-px": {"pass": False}}, True)
    assert bad["SI1"]["token"] == rc.STOP_TOKEN and bad["PI1"]["token"] == "supported"
    assert all(c["token"] == rc.STOP_TOKEN for c in ri.grid_contrasts(rows, passing, False).values())


def test_scheduler_jobs_wait_for_123_training_and_avoid_its_gpu():
    spec = json.loads((ROOT / "scripts/gpu_window_issue_125.json").read_text())["jobs"]
    known = {j["id"]: {} for j in json.loads((ROOT / "scripts/gpu_window_issue_123.json").read_text())["jobs"]}
    for job in spec:
        gw.validate_spec(job, known)
        known[job["id"]] = job
    by_id = {j["id"]: j for j in spec}
    assert set(by_id) == {"125-trigger", "125-select-q0", "125-select-q1", "125-select-q2", "125-smoke"}
    assert by_id["125-trigger"]["depends_on"] == [f"123-train-q{g}" for g in range(4)]
    assert by_id["125-trigger"]["command"][-1] == "--trigger"
    for q in range(3):
        job = by_id[f"125-select-q{q}"]
        assert job["depends_on"] == ["125-trigger"]
        assert job["command"][-4:] == ["--gpu", str(q), "--workers", "3"]
    assert by_id["125-smoke"]["depends_on"] == ["125-select-q0", "125-select-q1", "125-select-q2"]
    # #123 scores, publishes and validates on GPU 0: no #125 job there; every job pins its GPU
    for job in spec:
        assert job["gpu"] != 0 and job["env"]["CUDA_VISIBLE_DEVICES"] == str(job["gpu"])
        assert not any(mode in job["command"] for mode in ("--train", "--readouts", "--score", "--prepare"))
