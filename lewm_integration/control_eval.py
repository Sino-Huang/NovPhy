from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

collector_module = importlib.import_module("modules.NovPhy.lewm_integration.collector")
runtime_adapter_module = importlib.import_module("modules.NovPhy.lewm_integration.runtime_adapter")

RANDOM_DX_RANGE = collector_module.RANDOM_DX_RANGE
RANDOM_DY_RANGE = collector_module.RANDOM_DY_RANGE
NovPhyAdapter = runtime_adapter_module.NovPhyAdapter


class ControlEvaluationError(ValueError):
    """Raised when control-evaluation inputs are invalid."""


def _ensure_real_planner_supported(checkpoint_path: Path | None) -> None:
    checkpoint_note = (
        f" Checkpoint '{checkpoint_path}' exists but no real planner-backed lewm_planner implementation is available in this MVP."
        if checkpoint_path is not None and checkpoint_path.exists()
        else f" Checkpoint '{checkpoint_path}' is unavailable."
    )
    raise ControlEvaluationError(
        "Real planner loading is not implemented yet; pass --allow-synthetic-model for smoke evaluation or add a real planner."
        + checkpoint_note
    )


def _sample_action(rng: np.random.Generator) -> list[float]:
    return [float(rng.uniform(*RANDOM_DX_RANGE)), float(rng.uniform(*RANDOM_DY_RANGE))]


def _synthetic_cost(action: list[float]) -> float:
    dx, dy = action
    return float((dx / 100.0) ** 2 + (dy / 100.0) ** 2)


def _flatten_action_records(episode_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episode_records:
        for decision_index, action_record in enumerate(episode.get("actions", [])):
            selected_action = action_record.get("selected_action", [0.0, 0.0])
            rows.append(
                {
                    "episode_index": int(episode.get("episode_index", 0)),
                    "decision_index": decision_index,
                    "policy": episode.get("policy", ""),
                    "task_id": episode.get("task_id", ""),
                    "selected_action_dx": float(selected_action[0]),
                    "selected_action_dy": float(selected_action[1]),
                    "selected_action_magnitude": float(np.hypot(selected_action[0], selected_action[1])),
                    "selected_cost": float(action_record.get("selected_cost", _synthetic_cost(selected_action))),
                    "candidate_count": int(len(action_record.get("candidate_actions", []))),
                    "transition_reason": action_record.get("transition_reason", ""),
                    "terminal": bool(episode.get("terminal", False)),
                    "truncated": bool(episode.get("truncated", False)),
                    "final_score": float(episode.get("final_score", 0.0)),
                }
            )
    return rows


def evaluate_control(
    *,
    policy: str,
    episodes: int,
    seed: int,
    output_path: Path,
    use_fake_runtime: bool,
    checkpoint_path: Path | None = None,
    allow_synthetic_model: bool = False,
    candidate_count: int = 8,
    max_decisions: int = 3,
) -> list[dict[str, Any]]:
    if policy not in {"random", "lewm_planner"}:
        raise ControlEvaluationError(f"Unsupported policy '{policy}'.")
    if policy == "lewm_planner" and not allow_synthetic_model:
        _ensure_real_planner_supported(checkpoint_path)

    rng = np.random.default_rng(seed)
    adapter = NovPhyAdapter(use_fake_runtime=use_fake_runtime)
    episode_records: list[dict[str, Any]] = []
    for episode_idx in range(episodes):
        observation, info = adapter.reset(task_id=f"episode-{episode_idx}", seed=seed + episode_idx)
        record: dict[str, Any] = {
            "episode_index": episode_idx,
            "task_id": info.get("task_id", f"episode-{episode_idx}"),
            "seed": seed + episode_idx,
            "policy": policy,
            "actions": [],
            "terminal": False,
            "truncated": False,
            "final_score": 0.0,
        }
        for _ in range(max_decisions):
            if policy == "random":
                action = _sample_action(rng)
                result = adapter.step(action)
                record["actions"].append({"selected_action": action, "transition_reason": result.info.get("transition_reason")})
            else:
                candidates = [_sample_action(rng) for _ in range(candidate_count)]
                costs = [_synthetic_cost(action) for action in candidates]
                best_index = int(np.argmin(costs))
                action = candidates[best_index]
                result = adapter.step(action)
                record["actions"].append(
                    {
                        "candidate_actions": candidates,
                        "candidate_costs": costs,
                        "selected_action": action,
                        "selected_cost": costs[best_index],
                        "transition_reason": result.info.get("transition_reason"),
                    }
                )

            record["terminal"] = bool(result.terminated)
            record["truncated"] = bool(result.truncated)
            record["final_score"] = float(result.info.get("score", 0.0))
            if result.terminated or result.truncated:
                break
        episode_records.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(episode_records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    adapter.close()
    return episode_records


def write_control_monitoring_csv(output_path: Path, episode_records: list[dict[str, Any]]) -> None:
    csv_path = output_path.with_suffix(".csv")
    rows = _flatten_action_records(episode_records)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
