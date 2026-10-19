"""Issue #123: tests of the uncertain code only (clip sampler, launch encoding, readout fit isolation, port) and of
the A2 publication guards (G-records, failed-arm abstention, per-readout G-nonfinite, positive-LR G-port)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from world_model.hjepa import novphy as hn

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "world_model/hjepa/vendor"


@pytest.mark.parametrize("arm", ["HJ-car-m", "LeWM-car-m", "HJ-car-m-fs5"])
def test_needed_frames_match_official_hjepa(arm):
    """The loader's level-1 frame set equals HJEPA._draw_crops_and_level1_frames for the same crop starts."""
    geometry = hn.Geometry(arm)
    model, _ = hn.build_model(arm, 0.09, checkpointing=False)
    rng = np.random.Generator(np.random.PCG64(3))
    for _ in range(20):
        starts = geometry.draw_crops(rng)
        queue = [s for s in (starts[level] for level in sorted(starts)) if s is not None]
        drawn = iter(queue)
        model._draw_crop_start = lambda *a, **k: next(drawn)
        official = hn.HJEPA._draw_crops_and_level1_frames(model, geometry.level1_steps, geometry.levels, "cpu")
        assert official[1] == geometry.needed(starts)
        assert {k: v for k, v in official[0].items()} == starts


def test_h3_geometry_matches_ticket():
    geometry = hn.Geometry("HJ-px")
    assert geometry.level1_steps == 46 and geometry.span == 46 and geometry.lengths == [46, 10, 4]
    assert geometry.top_steps == 15 and geometry.frames_per_top_step == 15
    assert geometry.valid_starts(285) == 241
    fs5 = hn.Geometry("HJ-car-m-fs5")
    assert fs5.level1_steps == 10 and fs5.span == 50 and fs5.top_steps == 15


def test_mixture_sampler_half_anchor_clips_and_valid_starts():
    geometry = hn.Geometry("HJ-car-m")
    ends = np.array([285, 138, 200, 285])
    valid = np.array([geometry.valid_starts(e) for e in ends])
    cumulative = np.concatenate([[0], np.cumsum(valid)])
    for b in range(50):
        rng = np.random.Generator(np.random.PCG64([7, b]))
        clips, starts, needed = hn.plan_batch(rng, ends, cumulative, geometry, "mixture", 64)
        assert len(clips) == 64
        assert all(start == 0 for _, start in clips[:32])
        for shot, start in clips:
            assert 0 <= start <= ends[shot] - geometry.span + 1
        assert needed == geometry.needed(starts) and max(needed) < geometry.level1_steps
    rng = np.random.Generator(np.random.PCG64([7, 0]))
    uniform, _, _ = hn.plan_batch(rng, ends, cumulative, geometry, "uniform", 4096)
    share_zero = np.mean([start == 0 for _, start in uniform])
    assert share_zero < 0.02  # the uniform sampler almost never sees the launch frame


def test_plan_is_deterministic_per_batch_index():
    geometry = hn.Geometry("HJ-car-m")
    ends = np.array([285] * 10)
    cumulative = np.concatenate([[0], np.cumsum([geometry.valid_starts(e) for e in ends])])
    first = hn.plan_batch(np.random.Generator(np.random.PCG64([5, 17])), ends, cumulative, geometry, "mixture", 32)
    second = hn.plan_batch(np.random.Generator(np.random.PCG64([5, 17])), ends, cumulative, geometry, "mixture", 32)
    assert first == second


def test_launch_encoding_sparse_persist_and_frameskip():
    launch = torch.tensor([-0.25, 0.1, 1.0, 0.0, 1.0])
    h3 = hn.Geometry("HJ-car-m")
    a = hn.level1_actions(launch, 0, h3, persist=False)
    assert a.shape == (46, 5) and torch.equal(a[0], launch) and not a[1:].any()
    assert not hn.level1_actions(launch, 3, h3, persist=False).any()
    assert torch.equal(hn.level1_actions(launch, 3, h3, persist=True), launch.repeat(46, 1))
    fs5 = hn.Geometry("HJ-car-m-fs5")
    b = hn.level1_actions(launch, 0, fs5, persist=False)
    assert b.shape == (10, 25) and torch.equal(b[0, :5], launch) and not b[0, 5:].any() and not b[1:].any()


@torch.no_grad()
def test_launch_reaches_only_the_first_macro_action():
    model, _ = hn.build_model("HJ-car-m", 0.09, checkpointing=False)
    model.eval()
    geometry = hn.Geometry("HJ-car-m")
    launches = torch.tensor([[-0.25, 0.1, 1.0, 0.0, 1.0], [-0.1, 0.15, 1.0, 0.0, 1.0]])
    act = hn.top_action_embeddings(model, launches, geometry, persist=False)
    assert act.shape == (2, 15, 112)
    assert not torch.allclose(act[0, 0], act[1, 0])
    assert torch.allclose(act[0, 1:], act[1, 1:])
    persist = hn.top_action_embeddings(model, launches, geometry, persist=True)
    assert not torch.allclose(persist[0, 5], persist[1, 5])


@torch.no_grad()
def test_planned_sparse_encode_equals_official_dense_encode():
    """PlannedHJEPA on the gathered frames reproduces the official dense encode_hierarchical outputs."""
    arm = "HJ-car-m"
    geometry = hn.Geometry(arm)
    model, _ = hn.build_model(arm, 0.09, checkpointing=False)
    model.eval()
    starts = {1: 9, 2: 2, 3: None}
    needed = geometry.needed(starts)
    x = torch.randn(3, geometry.level1_steps, 236)
    action = torch.zeros(3, geometry.level1_steps, 5)
    action[1, 0] = torch.tensor([-0.25, 0.1, 1.0, 0.0, 1.0])
    queue = iter([9, 2])
    model._draw_crop_start = lambda *a, **k: next(queue)
    dense = model.encode_hierarchical({"pixels_level1": x, "action_level1": action})
    del model._draw_crop_start
    model.plan = (geometry.level1_steps, starts, needed)
    sparse = model.encode_hierarchical({"pixels_level1": x[:, needed], "action_level1": action})
    for level in (1, 2, 3):
        assert torch.allclose(dense[f"embed_{level}"], sparse[f"embed_{level}"], atol=1e-6)
        assert torch.allclose(dense[f"action_{level}"], sparse[f"action_{level}"], atol=1e-6)


def test_readout_fit_isolation():
    """Training / readout data are the fit (and policy) streams only; the evaluation stream refuses before scoring."""
    from scripts import run_hjepa_comparison as hc
    with pytest.raises(ValueError):
        hc.stream_rows("evaluation")
    assert hc.probe_positions(285) == list(range(0, 286, 15))
    assert hc.probe_positions(138) == [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 138]
    index = hc.carriers_dir(hc.OUTPUT) / "fit-index.npz"
    if index.is_file():
        keys = np.load(index)["keys"]
        assert all(str(k).startswith("issue-109-sc-fit") for k in keys)


def test_probe_fit_is_seed_deterministic():
    x, y = torch.randn(512, 16), torch.randn(512, 6)
    first, _ = hn.fit_probe(x, y, 11, epochs=2, device="cpu")
    second, _ = hn.fit_probe(x, y, 11, epochs=2, device="cpu")
    for a, b in zip(first.state_dict().values(), second.state_dict().values()):
        assert torch.equal(a, b)


def test_vendored_files_match_their_pinned_hashes():
    import hashlib
    manifest = json.loads((VENDOR / "VENDOR.json").read_text())
    assert manifest["commit"] == "97054ec"
    for rel, entry in manifest["files"].items():
        assert "sha256:" + hashlib.sha256((VENDOR / rel).read_bytes()).hexdigest() == entry["vendored_sha256"], rel


def test_port_check_passed():
    path = ROOT / ".local-artifacts/issue-123-hjepa-comparison-v1/gport/gport.json"
    if not path.is_file():
        pytest.skip("G-port has not been run on this checkout")
    result = json.loads(path.read_text())
    assert result["pass"] and set(result["arms"]) == {"HJ-px", "HWM-px", "LeWM-px", "HJ-car-m", "HJ-car-m-fs5"}


A2_GPORT = ROOT / ".local-artifacts/issue-123-hjepa-comparison-v1/gport/a2"


def test_port_positive_lr_step_passed():
    """A2: the re-run passes both the step-0-LR step and the positive-LR step (parameters move, AdamW state and
    updated parameters within the declared tolerance); the frozen gport.json is kept and referenced by hash."""
    import hashlib
    path = A2_GPORT / "gport.json"
    if not path.is_file():
        pytest.skip("the A2 G-port re-run has not been run on this checkout")
    result = json.loads(path.read_text())
    frozen = ROOT / result["frozen_record"]["artifact"]
    assert result["frozen_record"]["sha256"] == "sha256:" + hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert result["pass"] and set(result["arms"]) == {"HJ-px", "HWM-px", "LeWM-px", "HJ-car-m", "HJ-car-m-fs5"}
    for arm, block in result["arms"].items():
        step2 = block["step2"]
        assert step2["pass"] and step2["parameters_moved"] and min(step2["lrs_at_step"]) > 0, arm
        assert step2["optimizer_steps_equal"] and all(m["pass"] for m in step2["optimizer_moments"].values()), arm


def test_positive_lr_step_detects_what_the_lr_zero_step_cannot():
    """A wrong gradient clip leaves the step-0-LR check passing (no parameter moves) but fails the A2 step through
    the AdamW moments."""
    from scripts import run_hjepa_comparison as hc
    path = A2_GPORT / "reference-HJ-car-m.pt"
    if not path.is_file():
        pytest.skip("the A2 G-port re-run has not been run on this checkout")
    reference = torch.load(path, map_location="cpu", weights_only=False)
    right = hc.gport_arm(reference)
    assert right["pass"]
    clip = hn.GRAD_CLIP
    try:
        hn.GRAD_CLIP = clip / 4
        wrong = hc.gport_arm(reference)
    finally:
        hn.GRAD_CLIP = clip
    for key in ("loss_max_relative_delta", "grad_norm_relative_delta", "grad_tensor_max_relative_l2",
                "param_close_share", "param_max_abs_delta"):  # the step at LR 0 cannot see the clip
        assert wrong[key] == pytest.approx(right[key], rel=1e-3, abs=1e-9), key
    assert not wrong["step2"]["optimizer_moments"]["exp_avg"]["pass"] and not wrong["pass"]


def contrast_rows(missing=(), strong=("HJ-car-m", "HJ-car-m:R2"), members=12):
    """Synthetic grid rows over every view a contrast reads: views of `missing` arms have no AUC (empty estimates),
    `strong` views rank far above U."""
    from scripts import run_hjepa_comparison as hc
    rng = np.random.default_rng(7)
    names = sorted({name for pair in hc.CONTRASTS.values() for name in pair})
    rows = []
    for m in range(members):
        for seed in hc.SEEDS:
            views = {}
            for name in names:
                auc = 0.95 if name in strong else 0.5 + 0.005 * float(rng.standard_normal())
                views[name] = {"auc": None if hc.arm_of(name) in missing else auc}
            rows.append({"member": f"level-{m:02d}", "seed": seed, "views": views})
    return rows


def all_tokens(block):
    if isinstance(block, dict):
        for key, value in block.items():
            if "token" in key:
                yield key, value
            yield from all_tokens(value)


def test_failed_arm_with_empty_estimates_abstains_without_crashing():
    """An arm that fails a guard and has no paired estimate (and an arm that fails a guard with estimates) publishes
    only the stop token, keeps its estimates as numbers, and carries no unguarded token field."""
    from scripts import run_hjepa_comparison as hc
    arms = tuple(hn.ARMS)
    rows = contrast_rows(missing=("HJ-px",))
    passing = {arm: {"pass": True} for arm in arms}
    control = hc.grid_contrasts(rows, arms, passing, True)
    assert control["PC2"]["token"] == "supported" and control["S9_PC2_R2"]["token"] == "supported"
    assert control["PC1"]["estimate"] is None and control["PC1"]["token"] == hc.STOP_TOKEN  # empty, guards pass
    guards = {**passing, "HJ-px": {"pass": False}, "HJ-car-m": {"pass": False}}
    contrasts = hc.grid_contrasts(rows, arms, guards, True)
    for label, (first, second) in hc.CONTRASTS.items():
        involved = {hc.arm_of(first), hc.arm_of(second)}
        if involved & {"HJ-px", "HJ-car-m"}:
            block = contrasts[label]
            assert block["token"] == block["reverse_token"] == hc.STOP_TOKEN and not block["guards_pass"], label
            assert (block["estimate"] is None) == ("HJ-px" in involved), label
    assert contrasts["PC2"]["estimate"]["mean"] > 0.4  # the diagnostic number stays
    records = {(arm, seed): {"typed_failure": "nonfinite_loss", "wrong_anchor": None}
               for arm in hc.WRONG_ANCHOR_ARMS for seed in hc.SEEDS}
    cells = {"grid": {"mixed": {f"level-{m:02d}": {"verdicts": {}} for m in range(3)}}}
    wrong = hc.wrong_anchor_table(cells, records, {arm: {"pass": False} for arm in arms}, True)
    assert all(w["token"] == hc.STOP_TOKEN and w["right_minus_wrong"] is None for w in wrong.values())
    grid = {"contrasts": contrasts, "S10_wrong_anchor": wrong}
    assert not any(key.startswith("raw") for key, _ in all_tokens(grid))
    assert hc.dispositions(grid)["disposition"] == 3
    assert hc.reverse(None) is None


def test_nonfinite_guard_counts_r1_and_r2_costs_and_gates_s9(tmp_path):
    """G-nonfinite counts the actual cost values of R1 and of R2 separately; > 10 % failed R2 costs fail the arm
    (its S9 token included) even when every R1 cost is finite."""
    from scripts import run_hjepa_comparison as hc
    from world_model.planning import scoring_harness as harness
    arm = "HJ-car-m"
    plan = {"arms": {arm: {"parameters": 1855172}}}
    for seed in hc.SEEDS:
        run = hc.run_dir(tmp_path, arm, seed)
        run.mkdir(parents=True)
        (run / "readouts.json").write_text(json.dumps({
            "collapse": [{"dim": 112, "effective_rank": 50.0, "within_episode_share": 0.5}],
            "action_sensitivity": 1.0, "policy_auc": {"ceiling": 0.9, "R1": 0.6}}))

    def records(bad_r2):
        rows = [{"ordinal": i, "excluded": False, harness.PRIMARY_COST: 0.1 * i,
                 f"{harness.PRIMARY_COST}_R2": None if i < bad_r2 else (float("nan") if i == bad_r2 else 0.2)}
                for i in range(20)]
        return {(arm, seed): {"typed_failure": None, "inventories": {"grid": {"level-00": {"rows": rows}}}}
                for seed in hc.SEEDS}

    ok = hc.arm_guards(tmp_path, plan, records(1))[arm]  # 2 of 20 R2 costs fail (None, NaN): 10 %
    assert ok["nonfinite_shares"] == {"R1": 0.0, "R2": 0.1} and ok["G_nonfinite"] and ok["pass"]
    failed = hc.arm_guards(tmp_path, plan, records(2))[arm]  # 3 of 20: 15 %
    assert failed["nonfinite_shares"] == {"R1": 0.0, "R2": 0.15} and not failed["G_nonfinite"] and not failed["pass"]
    arms = tuple(hn.ARMS)
    guards = {**{a: {"pass": True} for a in arms}, arm: failed}
    contrasts = hc.grid_contrasts(contrast_rows(), arms, guards, True)
    assert contrasts["S9_PC2_R2"]["token"] == contrasts["PC2"]["token"] == hc.STOP_TOKEN
    assert contrasts["S9_PC1_R2"]["token"] == "not_supported_by_this_experiment"


def test_publication_tokens_wait_for_a_bound_passing_validation(tmp_path, monkeypatch):
    """G-records: --publish emits only the stop token until --validate's re-score passes and its receipt binds the
    current records manifest; a failed re-score turns every published token back into the stop token."""
    from scripts import run_hjepa_comparison as hc
    manifest = {"schema": "issue_123_records_manifest_v1", "plan_identity": hc.IDENTITY, "records": []}
    spot = {"arm": "HJ-car-m", "seed": hc.SEEDS[0], "cells": 311, "max_abs_delta": {}, "pass": True}

    def tables(output, plan):
        guard = hc.records_guard(output)
        return {"guards": {"global": {"G_records": guard}}, "token": "supported" if guard["pass"] else hc.STOP_TOKEN,
                "disposition": {"disposition": 1 if guard["pass"] else 3, "label": "test"}}

    monkeypatch.setattr(hc, "load_plan", lambda output: {"identity": hc.IDENTITY})
    monkeypatch.setattr(hc, "records_manifest", lambda output: json.loads(json.dumps(manifest)))
    monkeypatch.setattr(hc, "compute_tables", tables)
    monkeypatch.setattr(hc, "rendered", lambda plan, compute: {n: json.dumps(compute["token"]) for n in hc.PUBLISHED})
    monkeypatch.setattr(hc, "spot_check", lambda output, plan: dict(spot))

    def published():
        return {json.loads((tmp_path / n).read_text()) for n in hc.PUBLISHED}

    hc.publish(tmp_path)
    assert published() == {hc.STOP_TOKEN} and not hc.records_guard(tmp_path)["pass"]
    assert hc.validate(tmp_path) == 0
    assert published() == {"supported"} and hc.records_guard(tmp_path)["pass"]
    assert hc.validate(tmp_path) == 0  # re-validation byte-compares the released publication
    manifest["records"] = [{"artifact": "records/x.json", "sha256": "sha256:0", "bytes": 1}]  # records changed
    assert not hc.records_guard(tmp_path)["pass"]
    manifest["records"] = []
    spot["pass"] = False
    with pytest.raises(ValueError, match="re-scoring"):
        hc.validate(tmp_path)
    assert published() == {hc.STOP_TOKEN}
    receipt = json.loads((tmp_path / hc.VALIDATION_RECEIPT).read_text())
    assert receipt["pass"] is False and not hc.records_guard(tmp_path)["pass"]
