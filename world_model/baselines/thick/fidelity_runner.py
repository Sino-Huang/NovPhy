"""#131 D1 THICK G-fid runner: the upstream ``train.py`` (Plan2Explore exploration data) and ``train_from_replay.py``
(THICK Dreamer / DreamerV2) loops on the G-port-verified port, resumable.

Upstream schedule kept: ``prefill`` random steps (train.py only; no update during prefill), the first update plus
``pretrain`` updates, then repeated [one evaluation episode with the eval policy (``eval_eps``), ``eval_every`` env
steps of the train policy], an update every ``train_every`` env steps (``common.Every``), ``train_steps`` updates
each; ``train_from_replay`` pre-fills its replay with the exploration run's episodes and counts env steps from 0.
A run ends after the evaluation at ``steps`` (the band's final window step; upstream's loop would continue to
``config.steps``). Not run (logging only upstream): the video / ``report`` passes and the random evaluation
episodes that fill upstream's evaluation replay (used only for reports).

Resumable: SIGTERM / SIGUSR1 -> the current env step and its update finish -> checkpoint (agent, optimizers,
target-update counters, the torch sampling generator, replay order and RNG, both drivers with their partial
episodes and policy states, the environments, the schedule counters, the loop phase) -> exit 75. A checkpoint is
also written after every loop iteration (1e4 env steps). GPU / wall seconds go to ``ledger/gfid-<run>.json``
(#123 ``mark`` format) and, on completion, ``report.json``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from world_model.baselines.thick import agent as ag
from world_model.baselines.thick import envs
from world_model.baselines.thick import port

EXIT_PAUSED = 75
KINDS = {
    "plan2explore": {"configs": ["visualpinpad", "plan2explore"], "from_replay": False},
    "thick": {"configs": ["thick_pinpad"], "from_replay": True},
    "dreamer": {"configs": ["visualpinpad"], "from_replay": True},
}
STOP = []


def request_stop(signum, frame):
    STOP.append(signum)


class Every:
    """``common.when.Every``."""

    def __init__(self, every, last=None):
        self.every, self.last = every, last

    def __call__(self, step):
        if not self.every:
            return False
        if self.last is None:
            self.last = step
            return True
        if step >= self.last + self.every:
            self.last += self.every
            return True
        return False


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n")
    os.replace(tmp, path)


def mark(ledger_dir, phase, wall, gpu, **extra):
    """The #123 ledger entry: ``ledger/<phase>.json`` accumulating wall / GPU seconds."""
    path = Path(ledger_dir) / f"{phase}.json"
    block = json.loads(path.read_text()) if path.is_file() else {"wall_seconds": 0.0, "gpu_seconds": 0.0}
    _write_json(path, {**block, "wall_seconds": block["wall_seconds"] + wall, "gpu_seconds": block["gpu_seconds"] + gpu,
                       "last_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra})


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(v) for v in value)
    return value


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_device(v, device) for v in value)
    return value


class Run:
    def __init__(self, kind, seed, logdir, task="pinpad_four", load_dir=None, steps=None, device="cuda",
                 tf32=True, timing_updates=1000, stop_after_updates=None, ledger_dir=None, run_name=None,
                 config_overrides=None):
        spec = KINDS[kind]
        self.kind, self.seed, self.logdir = kind, int(seed), Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.from_replay = spec["from_replay"]
        config = port.load_config(spec["configs"], {"task": task, **(config_overrides or {})})
        if config.expl_until <= 0:
            config = config.update({"expl_behavior": "greedy"})
        self.config = config
        self.steps = int(steps if steps is not None else config.steps)
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
        self.tf32 = tf32
        self.timing_updates, self.stop_after_updates = timing_updates, stop_after_updates
        self.ledger_dir = Path(ledger_dir) if ledger_dir else None
        self.run_name = run_name or f"gfid-{kind}-seed{seed}"
        torch.manual_seed(self.seed)
        self.train_env = envs.make_env(task, self.seed, config.time_limit)
        self.eval_env = envs.make_env(task, self.seed + 1000, config.time_limit)
        if self.from_replay and load_dir is None:
            raise ValueError("train_from_replay needs the exploration run's directory")
        load = Path(load_dir) / "train_episodes" if self.from_replay else None
        self.replay = envs.Replay(self.logdir / "train_episodes", **config.replay, load_directory=load,
                                  cache=(Path(load_dir) / "train_episodes_npy_cache") if load else None,
                                  seed=self.seed + 2000)
        obs_space = {k: tuple(v.shape) for k, v in self.train_env.obs_space.items()}
        self.act_dim = self.train_env.act_space["action"].n
        self.agent = ag.Agent(config, obs_space, self.act_dim).to(self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.noise = port.TorchNoise(self.generator)
        self.random_agent = envs.RandomAgent(self.train_env.act_space, self.seed + 3000)
        self.train_driver = envs.Driver([self.train_env])
        self.eval_driver = envs.Driver([self.eval_env])
        self.train_driver.on_step(self._on_train_step)
        self.train_driver.on_reset(lambda tran, worker: self.replay.add_step(tran, worker))
        self.train_driver.on_episode(lambda ep: self._on_episode(ep, "train"))
        self.eval_driver.on_episode(lambda ep: self._on_episode(ep, "eval"))
        self.step = 0
        self.updates = 0
        self.should_train = Every(config.train_every)
        self.should_log = Every(config.log_every)
        self.training_enabled = False  # train.py registers the update callback after the prefill
        self.phase = {"name": "start"}
        self.metric_sums, self.metric_count = {}, 0
        self.seconds = {"wall": 0.0, "update": 0.0}
        self.timing = {"updates": 0, "seconds": 0.0, "wall_start": None, "env_steps_at_start": None}
        self.checkpoint = self.logdir / "checkpoint.pt"
        self.timing_only_done = False
        self.metrics_path = self.logdir / "metrics.jsonl"

    # -- callbacks --
    def _log(self, record):
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps({"step": self.step, **record}) + "\n")

    def _on_episode(self, ep, mode):
        score = float(np.asarray(ep["reward"], np.float64).sum())
        self._log({f"{mode}_return": score, f"{mode}_length": len(ep["reward"]) - 1})

    def _on_train_step(self, tran, worker):
        self.step += 1
        self.replay.add_step(tran, worker)
        if self.training_enabled:
            if self.should_train(self.step):
                for _ in range(self.config.train_steps):
                    self._update()
            if self.should_log(self.step) and self.metric_count:
                self._log({k: float(v) / self.metric_count for k, v in self.metric_sums.items()})
                self.metric_sums, self.metric_count = {}, 0

    def _batch(self):
        batch = self.replay.batch(self.config.dataset.batch, self.config.dataset.length)
        out = {}
        for k, v in batch.items():
            t = torch.as_tensor(v)
            out[k] = t.float() if t.dtype == torch.float64 else t
        return {k: v.to(self.device, non_blocking=True) for k, v in out.items()}

    def _update(self):
        timed = self.timing["updates"] < self.timing_updates and self.device.type == "cuda"
        if timed:
            torch.cuda.synchronize()
        t0 = time.time()
        metrics = self.agent.train_step(self._batch(), self.noise)
        if timed:
            torch.cuda.synchronize()
        if self.timing["updates"] < self.timing_updates:
            self.timing["updates"] += 1
            self.timing["seconds"] += time.time() - t0
            if self.timing["updates"] == self.timing_updates:
                self._write_timing()
        self.updates += 1
        for k, v in metrics.items():
            self.metric_sums[k] = self.metric_sums.get(k, 0.0) + v.detach()
        self.metric_count += 1
        if self.stop_after_updates and self.updates >= self.stop_after_updates:
            self.timing_only_done = True
            STOP.append("updates")

    def _write_timing(self):
        wall = time.time() - self.timing["wall_start"] if self.timing["wall_start"] else None
        env_steps = self.step - self.timing["env_steps_at_start"] if self.timing["env_steps_at_start"] is not None \
            else None
        _write_json(self.logdir / "timing.json", {
            "kind": self.kind, "updates": self.timing["updates"], "tf32": self.tf32,
            "device": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "cpu",
            "seconds_per_update": self.timing["seconds"] / self.timing["updates"],
            "wall_seconds_since_first_train_phase": wall, "env_steps_in_that_wall": env_steps,
            "wall_seconds_per_env_step": (wall / env_steps) if wall and env_steps else None})

    # -- policies --
    def _obs(self, obs):
        out = {}
        for k, v in obs.items():
            arr = np.asarray(v)
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            out[k] = torch.as_tensor(arr).to(self.device)
        return out

    def _policy(self, mode_fn):
        def policy(obs, state):
            if state is not None:
                state = _to_device(state, self.device)
            action, state = self.agent.policy(self._obs(obs), state, mode=mode_fn(), noise=self.noise)
            return {"action": action.cpu().numpy()}, state
        return policy

    def _train_mode(self):
        until = self.config.expl_until // self.config.action_repeat
        explore = (not until) or self.step < until  # common.when.Until
        return "explore" if explore else "train"

    # -- checkpoint --
    def save(self):
        state = {
            "kind": self.kind, "seed": self.seed, "step": self.step, "updates": self.updates, "phase": self.phase,
            "training_enabled": self.training_enabled,
            "should_train": self.should_train.last, "should_log": self.should_log.last,
            "agent": self.agent.state_dict(), "optimizers": _to_cpu(self.agent.optimizer_states()),
            "generator": self.generator.get_state(), "torch_cpu_rng": torch.get_rng_state(),
            "replay": self.replay.state_dict(), "random_agent": self.random_agent._random.get_state(),
            "train_driver": _to_cpu(self.train_driver.state_dict()),
            "eval_driver": _to_cpu(self.eval_driver.state_dict()),
            "train_env": self.train_env, "eval_env": self.eval_env,
            "metric_sums": _to_cpu(self.metric_sums), "metric_count": self.metric_count,
            "seconds": self.seconds, "timing": self.timing,
        }
        tmp = self.checkpoint.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, self.checkpoint)

    def load(self):
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        assert state["kind"] == self.kind and state["seed"] == self.seed
        self.agent.load_state_dict(state["agent"])
        self.agent.load_optimizer_states(_to_device(state["optimizers"], self.device))
        self.generator.set_state(state["generator"])
        torch.set_rng_state(state["torch_cpu_rng"])
        self.replay.load_state_dict(state["replay"])
        self.random_agent._random.set_state(state["random_agent"])
        self.train_env, self.eval_env = state["train_env"], state["eval_env"]
        self.train_driver._envs, self.eval_driver._envs = [self.train_env], [self.eval_env]
        self.train_driver.load_state_dict(state["train_driver"])
        self.eval_driver.load_state_dict(state["eval_driver"])
        self.step, self.updates, self.phase = state["step"], state["updates"], state["phase"]
        self.training_enabled = state["training_enabled"]
        self.should_train.last, self.should_log.last = state["should_train"], state["should_log"]
        self.metric_sums = _to_device(state["metric_sums"], self.device)
        self.metric_count, self.seconds, self.timing = state["metric_count"], state["seconds"], state["timing"]

    # -- the loop --
    def _drive(self, driver, policy, steps=0, episodes=0):
        done_steps, done_episodes = driver(policy, steps=steps, episodes=episodes, stop=lambda: bool(STOP))
        return done_steps, done_episodes

    def run(self):
        """Returns 0 when the run is complete, EXIT_PAUSED after a stop request (checkpoint written)."""
        start = time.time()
        if self.checkpoint.is_file():
            self.load()
        try:
            status = self._loop()
        finally:
            wall = time.time() - start
            self.seconds["wall"] += wall
            if self.ledger_dir is not None:
                gpu = wall if self.device.type == "cuda" else 0.0
                mark(self.ledger_dir, self.run_name, wall, gpu, kind=self.kind, seed=self.seed, step=self.step,
                     updates=self.updates)
        if self.timing_only_done:
            return 0  # a timing-only run (stop_after_updates) ends without a checkpoint
        if status == EXIT_PAUSED:
            self.save()
        return status

    def _loop(self):
        cfg = self.config
        train_policy = self._policy(self._train_mode)
        eval_policy = self._policy(lambda: "eval")
        while True:
            name = self.phase["name"]
            if name == "start":
                prefill = 0 if self.from_replay else max(0, cfg.prefill - self.replay.stats["total_steps"])
                self.phase = {"name": "prefill", "steps": prefill, "episodes": 1 if prefill else 0}
            elif name == "prefill":
                steps, episodes = self._drive(self.train_driver, self.random_agent, self.phase["steps"],
                                              self.phase["episodes"])
                self.phase["steps"] -= steps
                self.phase["episodes"] -= episodes
                if self.phase["steps"] > 0 or self.phase["episodes"] > 0:
                    return EXIT_PAUSED
                self.train_driver.reset()
                self.phase = {"name": "pretrain"}
            elif name == "pretrain":
                self.training_enabled = True
                for _ in range(1 + cfg.pretrain):  # train_agent(next(dataset)), then the pretrain loop
                    self._update()
                self.phase = {"name": "eval", "episodes": cfg.eval_eps}
                self.save()
            elif name == "eval":
                _, episodes = self._drive(self.eval_driver, eval_policy, 0, self.phase["episodes"])
                self.phase["episodes"] -= episodes
                if self.phase["episodes"] > 0:
                    return EXIT_PAUSED
                if self.step >= self.steps:
                    self.phase = {"name": "done"}
                    self.save()
                    continue
                self.phase = {"name": "train", "steps": cfg.eval_every}
                if self.timing["wall_start"] is None:
                    self.timing["wall_start"], self.timing["env_steps_at_start"] = time.time(), self.step
            elif name == "train":
                steps, _ = self._drive(self.train_driver, train_policy, self.phase["steps"], 0)
                self.phase["steps"] -= steps
                if STOP:
                    return EXIT_PAUSED
                self.phase = {"name": "eval", "episodes": cfg.eval_eps}
                self.save()
            elif name == "done":
                _write_json(self.logdir / "report.json", {
                    "kind": self.kind, "seed": self.seed, "steps": self.step, "updates": self.updates,
                    "wall_seconds": self.seconds["wall"], "gpu_seconds": self.seconds["wall"]
                    if self.device.type == "cuda" else 0.0, "tf32": self.tf32,
                    "config": dict(cfg.flat)})
                return 0
            if STOP:
                return EXIT_PAUSED


def main(kind, seed, logdir, **kwargs):
    for signum in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, request_stop)
    run = Run(kind, seed, logdir, **kwargs)
    return run.run()
