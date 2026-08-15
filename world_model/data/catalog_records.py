"""Conversion from canonical validator output to immutable catalog records."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from world_model.data.types import (
    EpisodeRecord,
    FrameRecord,
    ShotAction,
    ShotRecord,
    SplitName,
)

if TYPE_CHECKING:
    from scripts.rollout_artifacts import ValidatedEpisode


def build_episode_record(
    validated_episode: ValidatedEpisode,
    name: str,
    split: str,
    root: Path,
    source_level_key: str | None,
) -> EpisodeRecord:
    episode_relative = validated_episode.directory.relative_to(root).as_posix()
    shot_records: list[ShotRecord] = []
    for validated_shot in validated_episode.shots:
        action = validated_shot.action
        if action is None:
            action = ShotAction(
                (0.0, 0.0, float(validated_shot.release_x), 0.0, 0.0)
            )
        shot_relative = f"{episode_relative}/{validated_shot.relative_path}"
        rebased_frames = tuple(
            FrameRecord(
                index=frame.index,
                relative_path=f"{episode_relative}/{frame.relative_path}",
                timestamp_seconds=frame.timestamp_seconds,
            )
            for frame in validated_shot.frames
        )
        shot_records.append(
            ShotRecord(
                name=validated_shot.name,
                relative_path=shot_relative,
                action=action,
                frames=rebased_frames,
            )
        )
    return EpisodeRecord(
        name=name,
        split=SplitName(split),
        relative_path=episode_relative,
        shots=tuple(shot_records),
        capture_contract=validated_episode.capture_contract,
        source_level_key=source_level_key,
    )
