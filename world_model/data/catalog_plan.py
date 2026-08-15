"""Collection-plan provenance and rollout validation-contract parsing."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.rollout_artifacts import EpisodeValidationContract


_DEFAULT_COUNT: Final = 12
_DEFAULT_FPS: Final = 30.0
_DEFAULT_DURATION: Final = 5.0


@dataclass(frozen=True, slots=True)
class PlanResolution:
    source_keys: dict[str, str]
    count: int
    fps: float
    duration_seconds: float


def make_validation_contract(
    count: int = _DEFAULT_COUNT,
    fps: float = _DEFAULT_FPS,
    duration_seconds: float = _DEFAULT_DURATION,
) -> EpisodeValidationContract:
    from scripts.rollout_artifacts import EpisodeValidationContract  # noqa: PLC0415

    return EpisodeValidationContract(
        count=count,
        fps=fps,
        duration_seconds=duration_seconds,
        level_five=False,
    )


def episode_contract_from_manifest(episode_dir: Path) -> EpisodeValidationContract:
    try:
        manifest = json.loads(
            (episode_dir / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return make_validation_contract()
    if not isinstance(manifest, dict):
        return make_validation_contract()
    count = manifest.get("accepted_rollout_count")
    fps = manifest.get("target_fps")
    duration = manifest.get("duration_seconds")
    if (
        type(count) is not int
        or type(fps) not in (int, float)
        or type(duration) not in (int, float)
    ):
        return make_validation_contract()
    return make_validation_contract(
        count=count,
        fps=float(fps),
        duration_seconds=float(duration),
    )


def load_plan_resolution(
    plan_path: Path,
    split: str,
    root: Path,
) -> PlanResolution | None:
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    selected = payload.get("selected")
    if not isinstance(selected, list):
        return None

    contract = payload.get("contract")
    count = _DEFAULT_COUNT
    fps = _DEFAULT_FPS
    duration = _DEFAULT_DURATION
    if isinstance(contract, dict):
        raw_count = contract.get("count")
        raw_fps = contract.get("fps")
        raw_duration = contract.get("duration")
        if type(raw_count) is int:
            count = raw_count
        if type(raw_fps) in (int, float):
            fps = float(raw_fps)
        if type(raw_duration) in (int, float):
            duration = float(raw_duration)

    split_root = root / split
    split_resolved = split_root.resolve(strict=False)
    source_keys: dict[str, str] = {}
    for item in selected:
        if not isinstance(item, dict) or item.get("split") != split:
            continue
        output_path = item.get("output_path")
        relative_path = item.get("relative_path")
        if not isinstance(output_path, str) or not isinstance(relative_path, str):
            continue
        try:
            candidate = (root / output_path).resolve(strict=False)
            candidate.relative_to(split_resolved)
        except (OSError, ValueError):
            continue
        source_keys[str(candidate)] = relative_path

    return PlanResolution(source_keys, count, fps, duration)
