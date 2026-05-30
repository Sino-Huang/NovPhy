from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

contracts_module = importlib.import_module("modules.NovPhy.lewm_integration.contracts")
fake_runtime_module = importlib.import_module("modules.NovPhy.lewm_integration.fake_runtime")
runtime_adapter_module = importlib.import_module("modules.NovPhy.lewm_integration.runtime_adapter")

DEFAULT_COORDINATE_CONVENTION = contracts_module.DEFAULT_COORDINATE_CONVENTION
TransitionReason = contracts_module.TransitionReason
ensure_transition_reason = contracts_module.ensure_transition_reason
FakeRuntimeConfig = fake_runtime_module.FakeRuntimeConfig
NovPhyAdapter = runtime_adapter_module.NovPhyAdapter


RANDOM_DX_RANGE = (-100.0, -10.0)
RANDOM_DY_RANGE = (-100.0, 100.0)


def _make_boundary_action(actions: list[npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
    if actions:
        return np.asarray(actions[-1], dtype=np.float32).copy()
    return np.zeros(2, dtype=np.float32)


@dataclass(frozen=True)
class CollectorConfig:
    episodes: int
    seed: int
    output_path: Path
    policy: str
    max_decisions: int = 3
    novelty_level: int = 0
    scenario: str = "unknown"
    task_filter: str = ""
    task_root: str = ""
    replay_path: Path | None = None
    use_fake_runtime: bool = False
    speed: int = 50


def _load_replay_actions(path: Path) -> list[list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Replay JSON must be a list of 2D actions.")
    return [[float(item[0]), float(item[1])] for item in payload]


def _sample_random_action(rng: np.random.Generator) -> list[float]:
    dx = float(rng.uniform(*RANDOM_DX_RANGE))
    dy = float(rng.uniform(*RANDOM_DY_RANGE))
    return [dx, dy]


def _collect_episode(adapter: NovPhyAdapter, *, rng: np.random.Generator, episode_index: int, config: CollectorConfig) -> dict[str, Any]:
    observation, reset_info = adapter.reset(task_id=f"episode-{episode_index}", seed=config.seed + episode_index)
    observations: list[npt.NDArray[np.uint8]] = [observation]
    actions: list[npt.NDArray[np.float32]] = []
    rewards: list[float] = []
    scores: list[float] = []
    terminated_flags: list[int] = []
    truncated_flags: list[int] = []
    reasons: list[str] = []
    waits: list[int] = []

    replay_actions = _load_replay_actions(config.replay_path) if config.policy == "replay_json" and config.replay_path else []

    for step_idx in range(config.max_decisions):
        if config.policy == "random":
            action = _sample_random_action(rng)
        elif config.policy == "replay_json":
            if step_idx >= len(replay_actions):
                break
            action = replay_actions[step_idx]
        else:
            raise ValueError(f"Unsupported policy '{config.policy}'. Supported policies: random, replay_json")

        result = adapter.step(action)
        executed_action = np.asarray(result.info.get("executed_action", action), dtype=np.float32)
        actions.append(executed_action)
        rewards.append(float(result.reward))
        scores.append(float(result.info.get("score", adapter.get_score())))
        terminated_flags.append(int(result.terminated))
        truncated_flags.append(int(result.truncated))
        reasons.append(str(result.info.get("transition_reason", TransitionReason.STATIC.value)))
        waits.append(int(result.info.get("static_wait_steps", 0)))
        observations.append(result.observation)
        if result.terminated or result.truncated:
            break

    final_reason = reasons[-1] if reasons else TransitionReason.STATIC.value
    final_terminated = terminated_flags[-1] if terminated_flags else 0
    final_truncated = truncated_flags[-1] if truncated_flags else 0
    final_score = scores[-1] if scores else 0.0

    actions.append(_make_boundary_action(actions))
    rewards.append(0.0)
    scores.append(final_score)
    terminated_flags.append(final_terminated)
    truncated_flags.append(final_truncated)
    reasons.append(final_reason)
    waits.append(waits[-1] if waits else 0)

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "scores": scores,
        "terminated": terminated_flags,
        "truncated": truncated_flags,
        "reasons": reasons,
        "waits": waits,
        "episode_seed": config.seed + episode_index,
        "task_id": reset_info.get("task_id", f"episode-{episode_index}"),
        "scenario": config.scenario,
        "novelty_level": config.novelty_level,
    }


def _flatten_episodes(episodes: list[dict[str, Any]]) -> dict[str, npt.NDArray[Any]]:
    pixels: list[npt.NDArray[np.uint8]] = []
    actions: list[npt.NDArray[np.float32]] = []
    rewards: list[float] = []
    episode_ids: list[int] = []
    step_ids: list[int] = []
    task_ids: list[int] = []
    scenario_ids: list[int] = []
    novelty_levels: list[int] = []
    seeds: list[int] = []
    scores: list[float] = []
    terminated: list[int] = []
    truncated: list[int] = []
    transition_reasons: list[str] = []
    static_wait_steps: list[int] = []
    ep_len: list[int] = []
    ep_offset: list[int] = []

    scenario_vocab: dict[str, int] = {}
    task_vocab: dict[str, int] = {}
    offset = 0
    for episode_index, episode in enumerate(episodes):
        episode_observations = episode["observations"]
        episode_actions = episode["actions"]
        episode_rewards = episode["rewards"]
        episode_scores = episode["scores"]
        episode_terminated = episode["terminated"]
        episode_truncated = episode["truncated"]
        episode_reasons = episode["reasons"]
        episode_waits = episode["waits"]
        scenario_name = str(episode["scenario"])
        scenario_id = scenario_vocab.setdefault(scenario_name, len(scenario_vocab))
        task_name = str(episode["task_id"])
        task_id = task_vocab.setdefault(task_name, len(task_vocab))

        episode_length = len(episode_observations)
        ep_len.append(episode_length)
        ep_offset.append(offset)
        offset += episode_length

        for step_index in range(episode_length):
            pixels.append(np.asarray(episode_observations[step_index], dtype=np.uint8))
            actions.append(np.asarray(episode_actions[step_index], dtype=np.float32))
            rewards.append(float(episode_rewards[step_index]))
            episode_ids.append(episode_index)
            step_ids.append(step_index)
            task_ids.append(task_id)
            scenario_ids.append(scenario_id)
            novelty_levels.append(int(episode["novelty_level"]))
            seeds.append(int(episode["episode_seed"]))
            scores.append(float(episode_scores[step_index]))
            terminated.append(int(episode_terminated[step_index]))
            truncated.append(int(episode_truncated[step_index]))
            transition_reasons.append(str(ensure_transition_reason(episode_reasons[step_index]).value))
            static_wait_steps.append(int(episode_waits[step_index]))

    return {
        "pixels": np.stack(pixels, axis=0),
        "action": np.stack(actions, axis=0),
        "reward": np.asarray(rewards, dtype=np.float32),
        "ep_len": np.asarray(ep_len, dtype=np.int64),
        "ep_offset": np.asarray(ep_offset, dtype=np.int64),
        "episode_id": np.asarray(episode_ids, dtype=np.int64),
        "step_id": np.asarray(step_ids, dtype=np.int64),
        "task_id": np.asarray(task_ids, dtype=np.int64),
        "scenario_id": np.asarray(scenario_ids, dtype=np.int64),
        "novelty_level": np.asarray(novelty_levels, dtype=np.int64),
        "seed": np.asarray(seeds, dtype=np.int64),
        "score": np.asarray(scores, dtype=np.float32),
        "terminated": np.asarray(terminated, dtype=np.int8),
        "truncated": np.asarray(truncated, dtype=np.int8),
        "transition_reason": np.asarray(transition_reasons, dtype=h5py.string_dtype(encoding="utf-8")),
        "action_coordinate_convention": np.asarray(
            [DEFAULT_COORDINATE_CONVENTION] * len(pixels), dtype=h5py.string_dtype(encoding="utf-8")
        ),
        "static_wait_steps": np.asarray(static_wait_steps, dtype=np.int64),
    }


def write_dataset(path: Path, episodes: list[dict[str, Any]]) -> None:
    flattened = _flatten_episodes(episodes)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for name, values in flattened.items():
            handle.create_dataset(name, data=values)


def collect_dataset(config: CollectorConfig) -> dict[str, Any]:
    adapter = NovPhyAdapter(
        use_fake_runtime=config.use_fake_runtime,
        fake_runtime_config=FakeRuntimeConfig(max_steps=config.max_decisions),
        speed=config.speed,
    )
    rng = np.random.default_rng(config.seed)
    episodes = [_collect_episode(adapter, rng=rng, episode_index=i, config=config) for i in range(config.episodes)]
    write_dataset(config.output_path, episodes)
    adapter.close()
    completed = sum(1 for episode in episodes if episode["terminated"][-1] or episode["truncated"][-1])
    return {
        "episodes_requested": config.episodes,
        "episodes_recorded": len(episodes),
        "episodes_completed_or_truncated": completed,
        "output_path": str(config.output_path),
    }
