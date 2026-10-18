"""Issue #123: tests of the uncertain code only (clip sampler, launch encoding, readout fit isolation, port)."""
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
