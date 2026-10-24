"""#131 D1 THICK G-fid: CPU tests of the env / replay / driver ports, the behaviour helpers and the resumable runner.

Run: CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/test_issue_131_thick_gfid.py
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

from world_model.baselines.thick import agent as ag
from world_model.baselines.thick import envs
from world_model.baselines.thick import fidelity_runner as fr


def test_pinpad_is_the_vendored_env_and_pickles():
    env = envs.make_env("pinpad_four", seed=3)
    assert env.target == ("1", "2", "3", "4") and env.layout.shape == (16, 14)
    obs = env.reset()
    assert obs["image"].shape == (64, 64, 3) and obs["image"].dtype == np.uint8 and obs["is_first"]
    clone = pickle.loads(pickle.dumps(env))
    action = {"action": np.eye(5, dtype=np.float32)[2]}
    assert np.array_equal(env.step(action)["image"], clone.step(action)["image"])
    with pytest.raises(ValueError):
        env.step({"action": np.full(5, 0.2, np.float32)})


def test_pinpad_reward_for_the_target_sequence():
    env = envs.pinpad_class()("four")
    env.reset()
    env.player = (2, 2)  # inside pad 1; walk 1 -> 2 -> 3 -> 4 along the layout
    rewards = []
    for target in ("1", "2", "3", "4"):
        cells = [(x, y) for (x, y), c in np.ndenumerate(env.layout) if c == target]
        env.player = cells[0]
        out = env.step({"action": 0})  # no-op move onto the same tile
        rewards.append(out["reward"])
    assert rewards == [0.0, 0.0, 0.0, 10.0]


def test_time_limit_and_driver_layout():
    env = envs.make_env("pinpad_four", seed=0, time_limit=7)
    driver = envs.Driver([env])
    episodes = []
    driver.on_episode(lambda ep: episodes.append(ep))
    agent = envs.RandomAgent(env.act_space, seed=0)
    steps, eps = driver(agent, episodes=1)
    assert (steps, eps) == (7, 1)
    ep = episodes[0]
    assert len(ep["action"]) == 8 and ep["is_first"][0] and not ep["is_first"][1:].any() and ep["is_last"][-1]
    assert float(np.abs(ep["action"][0]).sum()) == 0.0  # the reset transition carries the zero action
    assert np.allclose(ep["action"][1:].sum(-1), 1.0)


def test_driver_stops_at_a_step_boundary_and_resumes():
    env = envs.make_env("pinpad_four", seed=0, time_limit=10)
    driver = envs.Driver([env])
    agent = envs.RandomAgent(env.act_space, seed=0)
    count = []
    driver.on_step(lambda tran, worker: count.append(1))
    steps, _ = driver(agent, steps=25, stop=lambda: len(count) >= 12)
    assert steps == 12
    steps, _ = driver(agent, steps=25 - steps)
    assert len(count) == 25


def _episodes(tmp_path, n, length, seed=0):
    env = envs.make_env("pinpad_four", seed=seed, time_limit=length)
    replay = envs.Replay(tmp_path, minlen=5, maxlen=5, seed=seed)
    driver = envs.Driver([env])
    driver.on_step(replay.add_step)
    driver.on_reset(replay.add_step)
    driver(envs.RandomAgent(env.act_space, seed), episodes=n)
    return replay


def test_replay_chunks_mark_sequence_starts(tmp_path):
    replay = _episodes(tmp_path / "a", 3, 20)
    assert replay.stats["loaded_episodes"] == 3 and replay.stats["loaded_steps"] == 60
    chunks = [replay.chunk(3) for _ in range(10)]  # 5-step sequences consumed 3 + 2 (+1 from the next)
    firsts = np.concatenate([c["is_first"] for c in chunks])
    assert firsts[::5].all() and firsts.sum() == 6
    batch = replay.batch(4, 5)
    assert batch["image"].shape == (4, 5, 64, 64, 3) and batch["is_first"][:, 0].all()


def test_replay_state_round_trip(tmp_path):
    replay = _episodes(tmp_path / "a", 2, 20)
    replay.chunk(3)
    state = pickle.loads(pickle.dumps(replay.state_dict()))
    expected = [replay.chunk(4) for _ in range(3)]
    other = envs.Replay(tmp_path / "a", minlen=5, maxlen=5, seed=99)
    other.load_state_dict(state)
    got = [other.chunk(4) for _ in range(3)]
    for a, b in zip(expected, got):
        assert all(np.array_equal(a[k], b[k]) for k in a)


def test_loaded_directory_is_memory_mapped(tmp_path):
    _episodes(tmp_path / "src", 2, 20)
    replay = envs.Replay(tmp_path / "own", minlen=5, maxlen=5, load_directory=tmp_path / "src",
                         cache=tmp_path / "cache", seed=0)
    assert replay.stats["loaded_episodes"] == 2
    first = next(iter(replay._complete_eps.values()))
    assert isinstance(first["image"], np.memmap)
    assert replay.batch(2, 5)["image"].shape == (2, 5, 64, 64, 3)


def test_lambda_return_matches_the_recursion():
    torch.manual_seed(0)
    reward, value, pcont = torch.randn(4, 3), torch.randn(4, 3), torch.rand(4, 3)
    bootstrap = torch.randn(3)
    out = ag.lambda_return(reward, value, pcont, bootstrap, 0.95)
    nxt = torch.cat([value[1:], bootstrap[None]], 0)
    expected, last = [], bootstrap
    for t in reversed(range(4)):
        last = reward[t] + pcont[t] * ((1 - 0.95) * nxt[t] + 0.95 * last)
        expected.append(last)
    assert torch.allclose(out, torch.stack(expected[::-1]), atol=1e-6)


def test_stream_norm_with_momentum_one_is_the_identity():
    norm = ag.StreamNorm(momentum=1.0, scale=1.0, eps=1e-8)
    x = torch.randn(5, 7)
    assert torch.equal(norm(x), x) and float(norm.mag) == 1.0


def _fake_exploration(tmp_path, length=60):
    _episodes(tmp_path / "explore" / "train_episodes", 2, length)
    return tmp_path / "explore"


OVERRIDES = {"time_limit": 60, "eval_every": 20, "dataset.batch": 2, "dataset.length": 6, "log_every": 10,
             "replay.minlen": 50, "replay.maxlen": 50}


def _final(run):
    return {k: v.detach().clone() for k, v in run.agent.state_dict().items()}


def test_runner_resume_equals_an_uninterrupted_run(tmp_path, monkeypatch):
    explore = _fake_exploration(tmp_path)
    fr.STOP.clear()
    full = fr.Run("dreamer", 5, tmp_path / "full", load_dir=explore, steps=40, device="cpu",
                  config_overrides=OVERRIDES, timing_updates=0)
    assert full.run() == 0
    reference = _final(full)
    assert full.updates == 2 + 40 // 5
    original = fr.Run._update

    def stopping(self):
        original(self)
        if self.updates == 5 and not getattr(self, "_stopped", False):
            self._stopped = True
            fr.STOP.append("test")

    monkeypatch.setattr(fr.Run, "_update", stopping)
    fr.STOP.clear()
    part = fr.Run("dreamer", 5, tmp_path / "part", load_dir=explore, steps=40, device="cpu",
                  config_overrides=OVERRIDES, timing_updates=0)
    assert part.run() == fr.EXIT_PAUSED
    assert (tmp_path / "part" / "checkpoint.pt").is_file()
    monkeypatch.setattr(fr.Run, "_update", original)
    fr.STOP.clear()
    resumed = fr.Run("dreamer", 5, tmp_path / "part", load_dir=explore, steps=40, device="cpu",
                     config_overrides=OVERRIDES, timing_updates=0)
    assert resumed.run() == 0
    assert resumed.updates == full.updates and resumed.step == full.step
    final = _final(resumed)
    assert all(torch.equal(final[k], reference[k]) for k in reference)
    returns = lambda d: [line for line in (d / "metrics.jsonl").read_text().splitlines() if "eval_return" in line]
    assert returns(tmp_path / "full") == returns(tmp_path / "part")


def test_agent_to_moves_the_optimizer_moments():
    config = ag.port.load_config(["visualpinpad"], {"task": "pinpad_four"}).update({"expl_behavior": "greedy"})
    obs = {"image": (64, 64, 3), "reward": (), "is_first": (), "is_last": (), "is_terminal": ()}
    agent = ag.Agent(config, obs, 5).to("meta")
    for opt in agent.optimizers():
        assert all(p.device.type == "meta" for _, p in opt.named)
        assert all(t.device.type == "meta" for t in opt.m + opt.v)
