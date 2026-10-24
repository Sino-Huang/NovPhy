"""Issue #130 (C1 DINO-WM) CPU tests: patch grid, launch / no-op schedule, rollout geometry, carrier tokens, R1
isolation. Run with CUDA_VISIBLE_DEVICES=""."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from world_model.baselines.dino_wm import novphy as dn

ROOT = Path(__file__).resolve().parents[1]


def test_preprocess_and_patch_grid():
    frames = torch.randint(0, 256, (2, 3, 240, 320), dtype=torch.uint8)
    x = dn.preprocess(frames)
    assert x.shape == (2, 3, 238, 322) and dn.GRID == (17, 23) and dn.NUM_PATCHES == 391
    assert -1.0 - 1e-5 <= float(x.min()) and float(x.max()) <= 1.0 + 1e-5  # Normalize(0.5, 0.5) range
    flat = torch.full((1, 3, 240, 320), 255, dtype=torch.uint8)
    assert torch.allclose(dn.preprocess(flat), torch.ones(1, 3, 238, 322))  # resize preserves constants
    if not dn.DINOV2_REPO.is_dir():
        pytest.skip("pinned DINOv2 hub clone not present")
    encoder = torch.hub.load(str(dn.DINOV2_REPO), dn.DINOV2_NAME, source="local", pretrained=False).eval()
    feats = dn.patch_features(encoder, x)
    assert feats.shape == (2, dn.NUM_PATCHES, dn.FEATURE_DIM)
    assert dn.official_encoder_input(torch.zeros(1, 3, 224, 224)).shape[-2:] == (196, 196)


def test_launch_then_noop_schedule():
    launch = torch.tensor([[-0.1, 0.05, 1.0, 0.0, 1.0], [-0.3, 0.1, 1.0, 0.0, 1.0]])
    acts = dn.step_actions(launch)
    assert acts.shape == (2, dn.STEPS, dn.ACTION_DIM) and dn.STEPS == 45 and dn.ACTION_DIM == 25
    assert torch.equal(acts[:, 0, :5], launch)
    assert float(acts[:, 0, 5:].abs().sum()) == 0 and float(acts[:, 1:].abs().sum()) == 0
    windows = dn.window_actions(launch, torch.tensor([0, 3]))
    assert windows.shape == (2, 4, 25)
    assert torch.equal(windows[0, 0, :5], launch[0]) and float(windows[0, 1:].abs().sum()) == 0
    assert float(windows[1].abs().sum()) == 0  # a window starting at frame 15 holds only no-ops


def small_model(num_patches=6, visual_dim=8, proprio_dim=None):
    torch.manual_seed(0)
    return dn.DinoWM(num_patches, visual_dim, proprio_dim=proprio_dim,
                     predictor={**dn.PREDICTOR, "depth": 2, "heads": 2, "mlp_dim": 16, "dropout": 0.0})


def test_rollout_reaches_frame_225_and_is_causal():
    model = small_model().eval()
    z0 = torch.randn(2, 1, 6, 8)
    act = dn.step_actions(torch.randn(2, 5))
    with torch.no_grad():
        z = model.rollout(z0, act)
    assert z.shape == (2, dn.STEPS + 1, 6, 8 + dn.ACTION_EMB_DIM)  # frames 0, 5, .., 225
    assert torch.equal(z[:, 0, :, :8], z0[:, 0])
    # the first prediction from one frame equals the first block of a 3-frame window (block-causal mask)
    window = torch.randn(2, 3, 6, 8)
    window[:, 0] = z0[:, 0]
    with torch.no_grad():
        full = model.predict(model.encode(window, act[:, :3]))
        single = model.predict(model.encode(z0, act[:, :1]))
    assert torch.allclose(full[:, 0], single[:, 0], atol=1e-5)
    # the launch changes the endpoint
    with torch.no_grad():
        other = model.rollout(z0, dn.step_actions(torch.randn(2, 5)))
    assert float((other[:, -1] - z[:, -1]).abs().max()) > 0


def test_loss_excludes_action_channels_and_proprio_is_optional():
    model = small_model()
    visual = torch.randn(2, 4, 6, 8)
    act = torch.randn(2, 4, dn.ACTION_DIM)
    z_pred, loss, comp = model(visual, act)
    assert model.predictor.transformer.norm.normalized_shape == (8 + dn.ACTION_EMB_DIM,)
    z_tgt = model.encode(visual, act)[:, 1:]
    assert torch.allclose(loss, torch.nn.functional.mse_loss(z_pred[..., :8], z_tgt[..., :8]))
    with_proprio = small_model(proprio_dim=4)
    _, loss_p, comp_p = with_proprio(visual, act, torch.randn(2, 4, 4))
    assert "z_proprio_loss" in comp_p and torch.isfinite(loss_p)
    assert with_proprio.predictor.transformer.norm.normalized_shape == (8 + 10 + 10,)


def test_carrier_token_round_trip():
    rows = torch.randn(64, dn.CARRIER_DIM) * torch.linspace(0.1, 5, dn.CARRIER_DIM) + 3
    mean, std = dn.carrier_stats(rows)
    tokens = dn.carrier_to_tokens(rows[:5], mean, std)
    assert tokens.shape == (5, dn.SLOTS, dn.SLOT_TOKEN_DIM)
    assert torch.allclose(tokens[:, 0, dn.SLOT_WIDTH:], tokens[:, 7, dn.SLOT_WIDTH:])  # header tiled on every slot
    slot3 = (rows[:5, 2 + 3 * 13:2 + 4 * 13] - mean[2 + 39:2 + 52]) / std[2 + 39:2 + 52]
    assert torch.allclose(tokens[:, 3, :13], slot3, atol=1e-5)  # slot k = carrier columns 2 + 13 k ..
    back = dn.tokens_to_carrier(tokens, mean, std)
    assert torch.allclose(back, rows[:5], atol=1e-4)
    assert dn.build("C1-car").predictor.transformer.norm.normalized_shape == (dn.SLOT_TOKEN_DIM + dn.ACTION_EMB_DIM,)


class RecordingStore:
    """An [N, P, D] feature store that records which rows were read."""

    def __init__(self, array):
        self.array, self.read = array, set()
        self.shape = array.shape

    def __getitem__(self, idx):
        self.read.update(np.atleast_1d(np.asarray(idx)).tolist())
        return self.array[idx]


def test_r1_reads_only_its_fit_rows():
    rng = np.random.default_rng(0)
    store = RecordingStore(rng.standard_normal((40, 5, 8)).astype(np.float16))
    rows = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    targets = torch.randn(len(rows), dn.CARRIER_DIM)
    old = dict(dn.R1)
    dn.R1.update(batch=4, epochs=2, heads=2, hidden=8)
    try:
        probe, report = dn.fit_readout(store, rows, targets, seed=1, device="cpu")
    finally:
        dn.R1.clear()
        dn.R1.update(old)
    assert store.read <= set(rows.tolist()) and report["samples"] == len(rows)
    assert dn.apply_readout(probe, store.array[:3], device="cpu").shape == (3, dn.CARRIER_DIM)


def test_fit_index_rejects_uncached_frames_and_r1_set():
    from scripts import b127_dino_wm_features as feat
    ends = [285, 138]
    first, positions = feat.fit_layout(ends)
    assert positions[first[1] - 1] == 285 and 225 in positions[:first[1]].tolist()
    assert 138 in positions[first[1]:].tolist() and 135 in positions[first[1]:].tolist()
    index = feat.FitIndex(first, positions)
    items = feat.r1_items(ends)
    rows = index.rows([s for s, _ in items], [p for _, p in items])
    assert len(set(rows.tolist())) == len(items)
    with pytest.raises(KeyError):
        index.rows([0], [7])  # not a frame-skip-5 or probe position


FORBIDDEN = {"evaluation_cells", "read_evaluation_anchors", "evaluation_set", "eval_attempt", "read_pixel_anchors",
             "build_eval_cache", "evaluation_anchors_path"}


@pytest.mark.parametrize("path", ["scripts/b127_dino_wm_features.py", "world_model/baselines/dino_wm/novphy.py"])
def test_no_evaluation_access(path):
    tree = ast.parse((ROOT / path).read_text())
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not names & FORBIDDEN
    assert not any("issue109-evaluation" in s for s in strings)


DINO_PY = Path("/srv/home/shua0171/miniconda3/envs/b127-dino_wm/bin/python")
FIDELITY = ROOT / "scripts/b127_dino_wm_fidelity.py"


def _smoke(run_dir, *extra):
    import os
    import subprocess
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    return subprocess.run([str(DINO_PY), str(FIDELITY), "--smoke", "--run-dir", str(run_dir), "--max-iter", "2",
                           *extra], env=env, capture_output=True, text=True, timeout=900)


@pytest.mark.skipif(not DINO_PY.is_file() or not (ROOT / "data/external/dino_wm/checkpoints").is_dir(),
                    reason="b127-dino_wm env or the released checkpoint is absent")
def test_tier_e_resume_equals_uninterrupted(tmp_path):
    """Official plan.py (2 cases, 2 MPC iterations, CPU): stop after iteration 1 (exit 75) + resume == one run."""
    whole, split = tmp_path / "whole", tmp_path / "split"
    assert _smoke(whole).returncode == 0
    first = _smoke(split, "--stop-after", "1")
    assert first.returncode == 75 and (split / "mpc_checkpoint.pt").is_file() and not (split / "mpc_final.pt").is_file()
    second = _smoke(split)
    assert second.returncode == 0 and "resumed the MPC loop after iteration 1" in second.stdout
    assert (whole / "logs.json").read_bytes() == (split / "logs.json").read_bytes()
    a = torch.load(whole / "mpc_final.pt", weights_only=False)
    b = torch.load(split / "mpc_final.pt", weights_only=False)
    assert torch.equal(a["planned_actions"], b["planned_actions"]) and a["iterations"] == b["iterations"] == 2
    assert np.array_equal(a["action_len"], b["action_len"]) and np.array_equal(a["is_success"], b["is_success"])


# ---------------------------------------------------------------------------
# the #130 runner (C1-car)
# ---------------------------------------------------------------------------

class TinyShots:
    """Three synthetic fit shots (ends 285, 150, 23) with carriers = f(shot, position)."""

    def __init__(self):
        self.ends = np.array([285, 150, 23])
        self.launch5 = torch.tensor([[-0.1, 0.05, 1.0, 0.0, 1.0], [-0.2, 0.1, 1.0, 0.0, 1.0], [-0.3, 0.15, 1.0, 0.0, 1.0]])
        self.members = ["a", "b", "c"]

    def __len__(self):
        return 3

    def carrier_rows(self, shots, positions):
        shots, positions = torch.as_tensor(np.asarray(shots)), torch.as_tensor(np.asarray(positions))
        base = shots[:, None].float() * 1000 + positions[:, None].float()
        return base + torch.arange(dn.CARRIER_DIM)[None].float() * 0.01


def test_runner_windows_actions_and_targets():
    from scripts import run_issue_130_dino_wm as r
    data = r.FitCarriers(TinyShots())
    # steps: shot 0 -> 0..45, shot 1 -> 0..30, shot 2 -> 0..4; windows of 4 frames: 43 + 28 + 2
    assert len(data.windows) == 43 + 28 + 2 and data.last.tolist() == [45, 30, 4]
    assert torch.isnan(data.carriers[1, 31]).all() and not torch.isnan(data.carriers[1, 30]).any()
    index = torch.arange(len(data.windows))
    tokens, acts = data.batch(index)
    assert tokens.shape == (len(index), 4, dn.SLOTS, dn.SLOT_TOKEN_DIM) and bool(torch.isfinite(tokens).all())
    expected = torch.cat([dn.window_actions(data.launch5[s][None], torch.tensor([t]))
                          for s, t in data.windows.tolist()])
    assert torch.equal(acts, expected)
    # window (shot 0, start 42) ends at frame 225 and its tokens decode back to the stored carriers
    k = [i for i, (s, t) in enumerate(data.windows.tolist()) if (s, t) == (0, 42)][0]
    back = dn.tokens_to_carrier(tokens[k, -1], data.mean, data.std)
    assert torch.allclose(back, TinyShots().carrier_rows([0], [225])[0], rtol=1e-5, atol=1e-2)


def test_fixed_order_batches():
    from scripts import run_issue_130_dino_wm as r
    batches = r.FixedOrderBatches(5, 70, batch=32)
    assert batches.spe == 3 and len(batches(2)) == 6  # drop_last False
    assert torch.equal(batches(0), batches(3)) and torch.equal(batches(1), r.FixedOrderBatches(5, 70, 32)(4))
    assert sorted(torch.cat([batches(i) for i in range(3)]).tolist()) == list(range(70))


def tiny_spec(r, steps):
    return {"name": "tiny", "arm": r.ARM, "batch": 8, "epochs": None, "samples": None, "steps": steps,
            "predictor_lr": 5e-4, "action_encoder_lr": 5e-4, "windows": 0}


def test_train_resume_equals_uninterrupted(tmp_path, monkeypatch):
    from scripts import external_baselines as eb
    from scripts import run_issue_130_dino_wm as r
    monkeypatch.setattr(r, "BATCH", 8)
    monkeypatch.setattr(dn, "PREDICTOR", {**dn.PREDICTOR, "depth": 1, "heads": 2, "mlp_dim": 16})
    torch.set_num_threads(2)
    data = r.FitCarriers(TinyShots())
    spec = tiny_spec(r, 6)
    whole = r.train_run(tmp_path / "whole", spec, 3, data, "cpu")
    assert whole["complete"]
    eb.STOP_REQUESTED.append(15)  # stop after the first step
    try:
        with pytest.raises(eb.TrainingPaused):
            r.train_run(tmp_path / "split", spec, 3, data, "cpu")
    finally:
        eb.STOP_REQUESTED.clear()
    assert (tmp_path / "split/resume.pt").is_file()
    r.train_run(tmp_path / "split", spec, 3, data, "cpu")
    a = torch.load(tmp_path / "whole/model.pt", weights_only=False)["state_dict"]
    b = torch.load(tmp_path / "split/model.pt", weights_only=False)["state_dict"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    wm, _ = r.load_trained(tmp_path / "whole/model.pt", "cpu")
    rolled = wm.rollout(data.carriers[:2, 0], data.launch5[:2])
    assert rolled.shape == (2, dn.STEPS + 1, dn.CARRIER_DIM)
    assert torch.allclose(rolled[:, 0], data.carriers[:2, 0], rtol=1e-5, atol=1e-2)  # frame 0 = the anchor
    assert torch.equal(wm.endpoints(data.carriers[:2, 0], data.launch5[:2]), rolled[:, -1])


def test_nonfinite_loss_is_a_typed_failure(tmp_path, monkeypatch):
    from scripts import run_issue_130_dino_wm as r
    monkeypatch.setattr(dn, "PREDICTOR", {**dn.PREDICTOR, "depth": 1, "heads": 2, "mlp_dim": 16})
    data = r.FitCarriers(TinyShots())
    data.carriers[0, 1] = float("inf")
    data.windows = data.windows[:1]
    report = r.train_run(tmp_path, tiny_spec(r, 3), 3, data, "cpu")
    assert report["complete"] is False and report["typed_failure"]["typed_failure"] == "nonfinite_loss"
    assert r.load_trained(tmp_path / "model.pt", "cpu")[0] is None


def test_selection_mode_on_cpu(tmp_path, monkeypatch):
    """--selection (3 configurations, 2 steps each, real fit split) then --selection-readouts (real policy split)."""
    from scripts import run_issue_130_dino_wm as r
    if not (ROOT / ".local-artifacts/issue-123-hjepa-comparison-v1/carriers/fit-index.npz").is_file():
        pytest.skip("#123 stores absent")
    torch.set_num_threads(8)
    monkeypatch.setattr(dn, "PREDICTOR", {**dn.PREDICTOR, "depth": 1, "heads": 2, "mlp_dim": 16})  # CPU time only
    configs = {name: {**c, "epochs": None, "samples": 64} for name, c in r.CONFIGS.items()}
    monkeypatch.setattr(r, "CONFIGS", configs)
    for queue in r.SELECTION_QUEUES:
        assert r.selection(tmp_path, queue, "cpu") == 0
    for name in configs:
        report = json.loads((r.selection_dir(tmp_path, name) / "report.json").read_text())
        assert report["complete"] and report["steps"] == 2
    assert r.selection_readouts(tmp_path, "cpu") == 0
    result = json.loads((tmp_path / "selection.json").read_text())
    assert result["selected"] in configs and all(v["auc"] is not None for v in result["results"].values())
    assert result["policy_cells"] == 94


# ---------------------------------------------------------------------------
# --score-novelty (P-novelty)
# ---------------------------------------------------------------------------

def _novelty_setup(tmp_path, monkeypatch):
    """A tiny trained C1-car per seed and fit-split stand-ins for the novelty cells / anchors (no evaluation read)."""
    from scripts import external_baselines as eb
    from scripts import run_issue_130_dino_wm as r
    torch.set_num_threads(2)
    monkeypatch.setattr(dn, "PREDICTOR", {**dn.PREDICTOR, "depth": 1, "heads": 2, "mlp_dim": 16})
    monkeypatch.setattr(r, "load_plan", lambda output: {"selected": "tiny"})
    monkeypatch.setattr(r, "load_manifest", lambda output: {})
    data = r.FitCarriers(TinyShots())
    for seed in eb.SEEDS:
        r.train_run(r.model_dir(tmp_path, seed, "tiny"), tiny_spec(r, 2), seed, data, "cpu")
    grid = eb.grid_items()
    carriers = {f"L{i}": data.carriers[i, 0] for i in range(2)}
    levels = {name: {"inventory": [{**it, "key": f"{name}-{it['ordinal']}"} for it in grid],
                     "verdicts": {it["ordinal"]: it["ordinal"] % 2 == 0 for it in grid}, "side": "novel",
                     "novelty_level": 1, "pair": name, "pair_ordinal": 0} for name in carriers}
    cells = {"levels": levels, "mixed": sorted(levels), "in_scope_cells": []}
    monkeypatch.setattr(eb, "novelty_cells", lambda: cells)
    monkeypatch.setattr(eb, "read_novelty_anchors", lambda output, policy, cells=None: {
        "carriers": carriers, "read_at": "2026-11-01T00:00:00Z"})
    monkeypatch.setattr(eb, "novelty_reference_records", lambda *a, **k: {})
    monkeypatch.setattr(r, "car_guards", lambda output, records: {"pass": True, "failed": []})
    return r, eb, cells, carriers


def test_score_novelty_refuses_before_primary_records(tmp_path, monkeypatch):
    r, _, _, _ = _novelty_setup(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="primary records"):
        r.score_novelty(tmp_path, "cpu")
    assert not (tmp_path / "novelty").exists()


def test_score_novelty_rows_equal_primary_path(tmp_path, monkeypatch):
    r, eb, cells, carriers = _novelty_setup(tmp_path, monkeypatch)
    for seed in eb.SEEDS:  # primary records exist (stand-ins; never re-scored)
        eb.write_json(r.record_path(tmp_path, r.ARM, seed), {"arm": r.ARM, "seed": seed, "typed_failure": None,
                                                             "scored_at": "2026-10-31T00:00:00Z"})
    before = {s: r.record_path(tmp_path, r.ARM, s).read_bytes() for s in eb.SEEDS}
    assert r.score_novelty(tmp_path, "cpu") == 0
    assert {s: r.record_path(tmp_path, r.ARM, s).read_bytes() for s in eb.SEEDS} == before
    status = json.loads((tmp_path / "novelty/status.json").read_text())
    assert status["scored"] == [r.ARM] and r.DINO in status["not_scored"]
    # the primary scoring path on the same levels / anchors (Gate-B cell layout), evaluation helpers stubbed
    gate_cells = {inv: {"mixed": {lv: {"inventory": m["inventory"], "verdicts": m["verdicts"]}
                                  for lv, m in cells["levels"].items()}} for inv in eb.INVENTORIES}
    first = cells["levels"]["L0"]["inventory"][0]
    monkeypatch.setattr(eb, "position_branches", lambda c: [("L0", first)])
    monkeypatch.setattr(eb, "position_cache", lambda: {"entries": {first["key"]: {
        "presence": torch.zeros(5, 18, dtype=torch.bool), "centers": torch.zeros(5, 18, 2)}}})
    monkeypatch.setattr(eb, "wall_per_decision", lambda fn, decisions: 0.0)
    monkeypatch.setattr(eb, "fit_anchors", lambda: {"carriers": {"a": carriers["L0"]}})
    for seed in eb.SEEDS:
        record = json.loads(r.novelty_record_path(tmp_path, r.ARM, seed).read_text())
        assert record["schema"] == r.SCHEMA_NOVELTY_RECORD and sorted(record["inventories"]["grid"]) == ["L0", "L1"]
        assert all(len(v["rows"]) == 16 for v in record["inventories"]["grid"].values())
        primary = r.score_arm(tmp_path, {"selected": "tiny"}, seed, {"carriers": carriers}, gate_cells, "cpu")
        fresh = json.loads(eb.json_text(primary["inventories"]["grid"]))
        assert fresh == record["inventories"]["grid"]
