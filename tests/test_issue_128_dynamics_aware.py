"""#128 A4 DynDQN (CPU): the NovPhy trajectory-distance mapping, adapter shapes, the Table-1 command line and the
fidelity runner's SIGTERM -> exit 75 -> resume path (bitwise equal to an uninterrupted run).

The last two tests need the upstream clone / the b127-dynaware env and are skipped without them.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

from world_model.baselines.dynamics_aware import novphy as dn
from world_model.baselines.dynamics_aware.vendor.agents import rollout_dist

ROOT = Path(__file__).resolve().parents[1]
CLONE = Path("/mnt/array/sukaih/novphy-baselines-src/DynamicsAware@eb65576")
ENV_PY = Path("/srv/home/shua0171/miniconda3/envs/b127-dynaware/bin/python")
LABELS = ROOT / ".local-artifacts" / dn.LABELS


def toy_pair():
    """Two rollouts of one level, 3 slots (bird, pig = dynamic; platform = static), 2 and 3 valid frames;
    the pig is destroyed in rollout 1 at t = 1."""
    coords = torch.zeros(2, 3, 3, 2)
    coords[0, :, 0] = torch.tensor([[0.1, 0.5], [0.4, 0.4], [0.7, 0.3]])  # bird, rollout 0
    coords[1, :, 0] = torch.tensor([[0.1, 0.5], [0.4, 0.1], [0.0, 0.0]])  # bird, rollout 1 (padded t=2)
    coords[:, :, 1] = torch.tensor([0.6, 0.5])  # pig at rest
    coords[:, :, 2] = torch.tensor([0.5, 0.6])  # platform
    present = torch.ones(2, 3, 3, dtype=torch.bool)
    present[1, 1:, 1] = False  # pig destroyed in rollout 1 from t = 1
    present[1, 2, :] = False
    frame_valid = torch.tensor([[True, True, True], [True, True, False]])
    moving = torch.tensor([[True, True, False], [True, True, False]])
    return {"coords": coords, "present": present, "frame_valid": frame_valid, "moving": moving}


def test_trajectory_distance_toy_pair():
    alpha = 1.25
    d = dn.trajectory_distances(toy_pair(), alpha)
    # joint frames t = 0, 1 (T = min(3, 2)); objects bird, pig (platform static); 4 (o, t) terms
    bird = 0.0 + math.dist((0.4, 0.4), (0.4, 0.1))
    pig = 0.0 + alpha  # t0 identical, t1 present in exactly one rollout -> alpha
    expected = (bird + pig) / (4 * alpha)
    assert d.shape == (2, 2)
    assert torch.allclose(d, torch.tensor([[0.0, expected], [expected, 0.0]]), atol=1e-7)
    # bins as the official binned loss: round(d * (K - 1))
    assert int(torch.round(d[0, 1] * 19)) == round(expected * 19)
    # both destroyed -> identical (0); clipped at alpha
    traj = toy_pair()
    traj["present"][0, 1:, 1] = False
    far = dn.trajectory_distances(traj, alpha)
    assert torch.isclose(far[0, 1], torch.tensor(bird / (4 * alpha)))
    traj["coords"][1, 1, 0] = torch.tensor([5.0, 5.0])
    assert torch.isclose(dn.trajectory_distances(traj, alpha)[0, 1], torch.tensor(alpha / (4 * alpha)))


@pytest.mark.parametrize("time_slice", [0, 3, -2])
def test_phyre_form_equals_official_measure_all_distances(time_slice):
    g = torch.Generator().manual_seed(3)
    n, t, o = 8, 17, 65
    rollouts = torch.zeros(n, t, o, 14)
    rollouts[..., :2] = torch.rand(n, t, o, 2, generator=g)
    colours = torch.randint(8, 14, (o,), generator=g)
    rollouts[:, :, torch.arange(o), colours] = 1.0
    rollouts[:, :, 50:] = 0.0  # padded object slots
    lengths = torch.randint(3, t + 1, (n,), generator=g)
    masks = (torch.arange(t)[None] >= lengths[:, None]).to(torch.uint8)
    ours = dn.trajectory_distances(dn.phyre_trajectories(rollouts, masks), 1.414, time_slice)
    official = rollout_dist.measure_all_distances(rollouts, masks, 1.414, time_slice=time_slice)
    assert torch.allclose(ours, official, atol=1e-6)


@pytest.mark.skipif(not LABELS.is_dir(), reason="#123 fit labels not present")
def test_novphy_trajectory_from_fit_labels():
    from scripts.run_slot_encoder_fix import vocabulary
    vocab = vocabulary()
    store = torch.load(sorted(LABELS.glob("labels-*.pt"))[0], weights_only=True)["shots"]
    short = next(e for e in store.values() if int(e["positions"][-1]) % 15 != 0)
    tr = dn.novphy_trajectory(short, vocab)
    kept = int(tr["frame_valid"].sum())
    assert kept == int(short["positions"][-1]) // 15 + 1  # the clamped end position is dropped
    assert tr["coords"].shape == (dn.T_MAX, 18, 2)
    assert torch.allclose(tr["coords"][:kept, :, 1], short["centers"][:kept, :, 1] * 0.75)
    static = [i for i, name in enumerate(vocab) if not name.startswith(dn.DYNAMIC_PREFIXES)]
    assert not tr["moving"][static].any()
    assert torch.equal(tr["moving"], short["presence"][0].bool() & torch.tensor(
        [name.startswith(dn.DYNAMIC_PREFIXES) for name in vocab]))
    assert dn.NOVPHY_CUTOFF == 1.25


def test_adapter_shapes_and_film_position():
    torch.manual_seed(0)
    model = dn.A4Net()
    assert model.stem[0].in_channels == 3 and model.action_networks[3] is not None
    assert all(a is None for a in list(model.action_networks)[:3])
    assert model._action_network.net.layers[0].in_features == 4
    assert model._action_network.net.layers[0].out_features == 256
    assert model.distance_regressor.in_features == 256 ** 2 and model.distance_regressor.out_features == 20
    frames = torch.randint(0, 256, (16, 3, 240, 320), dtype=torch.uint8)
    actions = torch.rand(16, 4)
    g = torch.Generator().manual_seed(1)
    traj = {"coords": torch.rand(16, dn.T_MAX, 18, 2, generator=g),
            "present": torch.ones(16, dn.T_MAX, 18, dtype=torch.bool),
            "frame_valid": torch.ones(16, dn.T_MAX, dtype=torch.bool),
            "moving": torch.ones(16, 18, dtype=torch.bool)}
    out = dn.losses(model, frames, actions, torch.tensor([0, 1] * 8).bool(), traj)
    assert out["logits"].shape == (16,) and out["embeddings"].shape == (16, 256)
    assert all(torch.isfinite(out[k]) for k in ("ce", "aux", "loss"))
    opt, sched = dn.make_optimizer(model, updates=10)
    before = model.reason.weight.detach().clone()
    dn.train_step(model, opt, sched, frames, actions, torch.tensor([0, 1] * 8).bool(), traj)
    assert not torch.equal(before, model.reason.weight)
    assert dn.parameter_count(model) == sum(p.numel() for p in model.parameters())


def test_task_balanced_indices_fallbacks():
    rng = np.random.RandomState(0)
    pos = [np.array([0, 1]), np.array([], dtype=np.int64), np.array([4, 5])]
    neg = [np.array([2, 3]), np.array([6, 7]), np.array([], dtype=np.int64)]
    idx, tasks = dn.task_balanced_indices(rng, pos, neg, 3, 8)
    assert len(idx) == 24 and sorted(tasks) == [0, 1, 2]
    for k, task in enumerate(tasks):
        block = idx[8 * k:8 * k + 8]
        allowed = set(pos[task]) | set(neg[task])
        assert set(block) <= allowed
        if len(pos[task]) and len(neg[task]):
            assert set(block[:4]) <= set(neg[task]) and set(block[4:]) <= set(pos[task])


@pytest.mark.skipif(not (CLONE / "agents" / "run_sweep_file.py").is_file(), reason="upstream clone absent")
def test_official_argv_equals_run_sweep_file_command(monkeypatch, tmp_path):
    for name in ("submitit", "phyre"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.syspath_prepend(str(CLONE / "agents"))
    import importlib
    sweep = importlib.import_module("run_sweep_file")
    conf = (CLONE / "agents/experiments/hand_crafted_experiments/06_hand_crafted_test_run_table_1.conf").read_text()
    commands, _ = sweep.build_command(sweep.base_command, *sweep.parse_config_file(conf)[0])
    paths = ("--output-dir", "--dqn-tensorboard-dir", "--dqn-checkpoint-dir")
    sys.path.insert(0, str(ROOT / "scripts"))
    import b127_dynamics_aware_fidelity as fid
    for fold in (0, 1, 2):
        [command] = [c for c in commands if "--eval-setup-name=ball_within_template " in c + " "
                     and f"--fold-id={fold} " in c + " "]
        official = [a for a in command.split()[2:] if a.split("=", 1)[0] not in paths]
        ours = [a for a in fid.official_argv(fold, tmp_path) if a.split("=", 1)[0] not in paths]
        assert ours == official
        assert {a.split("=", 1)[0] for a in fid.official_argv(fold, tmp_path)} >= set(paths)


DRIVER = r"""
import json, os, signal, sys, torch
sys.path.insert(0, "scripts")
import b127_dynamics_aware_fidelity as fid
dtrain, na = fid.import_official()
na.XE_EVAL_SIZE, na.AUX_LOSS_EVAL_TASKS = 16, 2
stop_at = int(sys.argv[2])
calls = []
original = na.sample_distance_loss
def counted(model, embeddings, *args, **kwargs):
    if embeddings.requires_grad:
        calls.append(1)
        if len(calls) == stop_at:
            os.kill(os.getpid(), signal.SIGTERM)
    return original(model, embeddings, *args, **kwargs)
na.sample_distance_loss = counted
torch.manual_seed(0)
torch.set_num_threads(4)
overrides = {"--dqn-updates": 6, "--dqn-train-batch-size": 16, "--dqn-debug": 1, "--dqn-num-auccess-actions": 0,
             "--dqn-eval-every": 1000, "--dqn-save-checkpoints-every": 1000, "--dqn-rank-size": 20,
             "--max-test-attempts-per-task": 5, "--dqn-eval-batch-size": 32}
code = fid.run_fold(0, sys.argv[1], overrides)
print(json.dumps({"code": code, "calls": len(calls)}))
"""


def _run(out, stop_at):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PHYRE_CACHE_DIR": str(ROOT / "data/external/dynamics_aware/phyre_cache")}
    proc = subprocess.run([str(ENV_PY), "-c", DRIVER, str(out), str(stop_at)], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not ENV_PY.is_file(), reason="b127-dynaware env absent")
def test_fidelity_runner_sigterm_resume_is_bitwise(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert _run(a, 0) == {"code": 0, "calls": 6}
    assert _run(b, 4) == {"code": 75, "calls": 4}
    assert sorted(p.name for p in (b / "checkpoints").iterdir()) == ["ckpt.00000004"]
    assert not (b / "output" / "results.json").exists()
    assert _run(b, 0) == {"code": 0, "calls": 2}
    ca = torch.load(a / "checkpoints" / "ckpt.00000006", weights_only=False)
    cb = torch.load(b / "checkpoints" / "ckpt.00000006", weights_only=False)
    assert ca["done_batches"] == cb["done_batches"] == 6
    for k, v in ca["model"].items():
        assert torch.equal(v, cb["model"][k]), k
    for i, s in ca["optim"]["state"].items():
        for k, v in s.items():
            assert torch.equal(v, cb["optim"]["state"][i][k]), (i, k)
    assert ca["scheduler"] == cb["scheduler"]
    assert all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y for x, y in zip(ca["rng"], cb["rng"]))
    ra = json.loads((a / "output" / "results.json").read_text())
    rb = json.loads((b / "output" / "results.json").read_text())
    assert ra["target_metric"] == rb["target_metric"]
    runs = [json.loads(line) for line in (b / "runs.jsonl").read_text().splitlines()]
    assert [r["outcome"] for r in runs] == ["paused", "completed"] and runs[1]["resumed_from"] == "ckpt.00000004"
    assert _run(b, 0) == {"code": 0, "calls": 0}  # finished fold: no work


def test_vendor_json_hashes():
    import hashlib
    vendor = ROOT / "world_model/baselines/dynamics_aware/vendor"
    doc = json.loads((vendor / "VENDOR.json").read_text())
    assert doc["schema"] == "issue_127_vendor_v1" and doc["commit"] == "eb65576d8a9e9b1f74c64518b5dd68af9d0a7eff"
    files = sorted(str(p.relative_to(vendor)) for p in vendor.rglob("*")
                   if p.is_file() and p.name != "VENDOR.json" and "__pycache__" not in p.parts)
    assert files == sorted(doc["files"])
    for rel, entry in doc["files"].items():
        digest = "sha256:" + hashlib.sha256((vendor / rel).read_bytes()).hexdigest()
        assert digest == entry["vendored_sha256"], rel
        if entry["upstream_path"] and "upstream" not in entry and CLONE.is_dir():
            upstream = CLONE / ("agents/" + rel.split("agents/", 1)[1] if rel.startswith("agents/") else rel)
            assert "sha256:" + hashlib.sha256(upstream.read_bytes()).hexdigest() == entry["upstream_sha256"], rel
        if entry["upstream_sha256"] == entry["vendored_sha256"]:
            assert entry["edits"] == [], rel
