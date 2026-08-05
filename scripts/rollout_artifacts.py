from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeAlias

from scripts.physics_artifact_validation import _capture_contract, validate_physics_shot_artifact
from scripts.rollout_validation_types import (
    EpisodeAccepted, EpisodeRejected, EpisodeRejectionCode, EpisodeSummary,
    EpisodeValidationContract, EpisodeValidationMode, EpisodeValidationResult,
    PhysicsArtifactError, ValidatedEpisode, ValidatedShot,
    reject as _reject,
)

from world_model.data.types import (
    LEGACY_RGB_V1,
    PHYSICS_CAPTURE_V1,
    FrameRecord,
    ShotAction,
)

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
def _artifact_rejection(path: Path, root: Path, *, directory: bool = False) -> EpisodeRejected | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return _reject(EpisodeRejectionCode.ESCAPING_ARTIFACT, path)
    if ".." in relative.parts:
        return _reject(EpisodeRejectionCode.ESCAPING_ARTIFACT, path)
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            return _reject(EpisodeRejectionCode.SYMLINK_ARTIFACT, path)
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except FileNotFoundError:
        return _reject(EpisodeRejectionCode.MISSING_ARTIFACT, path)
    except (OSError, ValueError):
        return _reject(EpisodeRejectionCode.ESCAPING_ARTIFACT, path)
    expected_type = path.is_dir() if directory else path.is_file()
    if not expected_type:
        return _reject(EpisodeRejectionCode.INVALID_SHOT_ARTIFACT, path)
    access_mode = os.R_OK | os.X_OK if directory else os.R_OK
    if not os.access(path, access_mode):
        return _reject(EpisodeRejectionCode.UNREADABLE_ARTIFACT, path)
    return None


def _read_json(path: Path, *, max_bytes: int | None = None) -> JsonObject | None:
    try:
        if max_bytes is not None and path.stat().st_size > max_bytes:
            return None
        with path.open("rb") as stream:
            encoded = stream.read() if max_bytes is None else stream.read(max_bytes + 1)
        if max_bytes is not None and len(encoded) > max_bytes:
            return None
        payload: JsonValue = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _attempts_are_canonical(manifest: JsonObject, count: int) -> bool:
    attempts = manifest.get("attempts")
    if not isinstance(attempts, list) or manifest.get("attempt_count") != len(attempts):
        return False
    accepted = 0
    for attempt in attempts:
        if not isinstance(attempt, dict) or not isinstance(attempt.get("artifact_validation"), dict):
            return False
        validation = attempt["artifact_validation"]
        if attempt.get("accepted") is True:
            accepted += 1
            if (attempt.get("attempt_status"), validation.get("accepted"), validation.get("classification"), validation.get("retryable"), validation.get("retry_decision")) != ("accepted", True, "gameplay-valid", False, "accept"):
                return False
        elif (attempt.get("accepted"), attempt.get("attempt_status"), validation.get("accepted"), validation.get("retryable"), validation.get("retry_decision")) != (False, "invalid_retryable", False, True, "retry"):
            return False
    return accepted == count


def _metadata_frame_matches(value: JsonValue, frame_name: str, path_prefixes: tuple[str, ...]) -> bool:
    if not isinstance(value, str):
        return False
    return any(value == f"{prefix}/{frame_name}" for prefix in path_prefixes)


def _shot_action(value: JsonValue) -> tuple[float, ShotAction | None] | None:
    if not isinstance(value, dict):
        return None
    release = value.get("drag_release", value.get("release"))
    if not isinstance(release, list) or not release or type(release[0]) not in (int, float) or release[0] == 0:
        return None
    start = value.get("drag_start")
    hold = value.get("holdTime")
    action = None
    if isinstance(start, list) and len(start) >= 2 and len(release) >= 2 and all(type(item) in (int, float) for item in (*start[:2], *release[:2], hold)):
        action = ShotAction((start[0], start[1], release[0], release[1], hold))
    return release[0], action


def _validated_shots(
    root: Path,
    action_log: JsonObject,
    count: int,
    *,
    mode: EpisodeValidationMode,
    capture_contract: str,
) -> tuple[ValidatedShot, ...] | EpisodeRejected:
    materialize_frames = mode.materializes_frames
    accepted = action_log.get("accepted_trials")
    if not isinstance(accepted, list) or len(accepted) != count:
        return _reject(EpisodeRejectionCode.INVALID_ACTION_LOG, "accepted_trials")
    shots: list[ValidatedShot] = []
    for trial in accepted:
        if not isinstance(trial, dict) or not isinstance(trial.get("shot_name"), str):
            return _reject(EpisodeRejectionCode.INVALID_ACTION_LOG, "accepted_trials")
        action = _shot_action(trial.get("action"))
        if action is None:
            return _reject(EpisodeRejectionCode.INVALID_ACTION_LOG, "accepted_trials.action")
        shot_name = trial["shot_name"]
        shot_dir = root / shot_name
        required_paths = [(shot_dir, True), (shot_dir / "metadata.json", False), (shot_dir / "frames", True)]
        if capture_contract != PHYSICS_CAPTURE_V1.contract_name:
            required_paths.append((shot_dir / "pre_shot.png", False))
        for path, directory in required_paths:
            rejection = _artifact_rejection(path, root, directory=directory)
            if rejection is not None:
                return rejection
        metadata = _read_json(shot_dir / "metadata.json")
        if metadata is None:
            return _reject(EpisodeRejectionCode.MALFORMED_JSON, shot_dir / "metadata.json")
        if capture_contract == PHYSICS_CAPTURE_V1.contract_name:
            try:
                validate_physics_shot_artifact(shot_dir)
            except PhysicsArtifactError:
                return _reject(EpisodeRejectionCode.INVALID_SHOT_ARTIFACT, shot_dir)
        frame_count = metadata.get("frame_count")
        if type(frame_count) is not int or frame_count < 1:
            return _reject(EpisodeRejectionCode.NONCONTIGUOUS_FRAMES, shot_dir / "frames")
        frames: list[FrameRecord] = []
        metadata_frames = metadata.get("frames")
        if metadata_frames is not None and (not isinstance(metadata_frames, list) or len(metadata_frames) != frame_count):
            return _reject(EpisodeRejectionCode.INVALID_FRAME_METADATA, shot_dir / "metadata.json")
        frames_dir = shot_dir / "frames"
        try:
            with os.scandir(frames_dir) as directory_entries:
                frame_entries = {entry.name: entry for entry in directory_entries}
        except FileNotFoundError:
            return _reject(EpisodeRejectionCode.NONCONTIGUOUS_FRAMES, frames_dir)
        except OSError:
            return _reject(EpisodeRejectionCode.UNREADABLE_ARTIFACT, frames_dir)
        metadata_path_prefixes = [str(frames_dir), str(frames_dir.absolute()), "frames"]
        current_directory = Path.cwd()
        if frames_dir.is_absolute() and frames_dir.is_relative_to(current_directory):
            metadata_path_prefixes.append(str(frames_dir.relative_to(Path.cwd())))
        frame_path_prefixes = tuple(dict.fromkeys(metadata_path_prefixes))
        for index in range(frame_count):
            frame_name = f"frame_{index:06d}.png"
            frame_path = frames_dir / frame_name
            frame_entry = frame_entries.get(frame_name)
            if frame_entry is None:
                return _reject(EpisodeRejectionCode.NONCONTIGUOUS_FRAMES, frame_path)
            if frame_entry.is_symlink():
                return _reject(EpisodeRejectionCode.SYMLINK_ARTIFACT, frame_path)
            if not frame_entry.is_file(follow_symlinks=False):
                return _reject(EpisodeRejectionCode.INVALID_SHOT_ARTIFACT, frame_path)
            if not os.access(frame_path, os.R_OK):
                return _reject(EpisodeRejectionCode.UNREADABLE_ARTIFACT, frame_path)
            frame_value = None if metadata_frames is None else metadata_frames[index]
            if frame_value is not None and (not isinstance(frame_value, dict) or not _metadata_frame_matches(frame_value.get("path"), frame_name, frame_path_prefixes)):
                return _reject(EpisodeRejectionCode.INVALID_FRAME_METADATA, frame_path)
            timestamp = frame_value.get("t") if isinstance(frame_value, dict) else None
            if materialize_frames:
                frames.append(FrameRecord(index, f"{shot_name}/frames/{frame_name}", timestamp if type(timestamp) in (int, float) else None))
        shots.append(ValidatedShot(shot_name, str(shot_dir.relative_to(root)), tuple(frames), frame_count, action[0], action[1]))
    return tuple(shots)


def validate_rollout_episode(
    root: Path,
    contract: EpisodeValidationContract,
    *,
    mode: EpisodeValidationMode = EpisodeValidationMode.MATERIALIZED,
    capture_contract: str | None = None,
) -> EpisodeValidationResult:
    if root.is_symlink():
        return _reject(EpisodeRejectionCode.SYMLINK_ARTIFACT, root)
    root_rejection = _artifact_rejection(root, root, directory=True)
    if root_rejection is not None:
        return root_rejection
    required: Final = (root / "manifest.json", root / "action_log.json", root / "action_log.jsonl")
    for path in required:
        rejection = _artifact_rejection(path, root)
        if rejection is not None:
            return rejection
    manifest = _read_json(required[0])
    action_log = _read_json(required[1])
    if manifest is None or action_log is None:
        return _reject(EpisodeRejectionCode.MALFORMED_JSON, required[0] if manifest is None else required[1])
    descriptor = _capture_contract(manifest, capture_contract)
    if isinstance(descriptor, EpisodeRejected):
        return descriptor
    if capture_contract is not None and descriptor.contract_name != capture_contract:
        return _reject(EpisodeRejectionCode.INVALID_EPISODE_CONTRACT, required[0])
    valid_contract = (
        manifest.get("capture_source") == ("capture_physics_rollout" if descriptor.contract_name == PHYSICS_CAPTURE_V1.contract_name else "capture_desktop_rollout")
        and manifest.get("replay_mode") == "fresh-engine-per-rollout"
        and type(manifest.get("target_fps")) in (int, float)
        and manifest.get("target_fps") == contract.fps
        and type(manifest.get("duration_seconds")) in (int, float)
        and manifest.get("duration_seconds") == contract.duration_seconds
        and manifest.get("ui_level") == 1
        and type(manifest.get("accepted_rollout_count")) is int
        and manifest.get("accepted_rollout_count") == contract.count
        and type(manifest.get("rollout_count")) is int
        and manifest.get("rollout_count") == contract.count
        and manifest.get("collection_status") != "retry_exhausted"
        and manifest.get("collection_error") is None
    )
    if not valid_contract:
        return _reject(EpisodeRejectionCode.INVALID_EPISODE_CONTRACT, required[0])
    if not _attempts_are_canonical(manifest, contract.count):
        return _reject(EpisodeRejectionCode.INVALID_ATTEMPT_LOG, required[0])
    shots = _validated_shots(root, action_log, contract.count, mode=mode, capture_contract=descriptor.contract_name)
    if isinstance(shots, EpisodeRejected):
        return shots
    signs = tuple(-1 if shot.release_x < 0 else 1 for shot in shots)
    expected_signs = tuple(-1 if index % 2 == 0 else 1 for index in range(contract.count))
    if contract.level_five and signs != expected_signs:
        return _reject(EpisodeRejectionCode.LEVEL_FIVE_ACTION_POLICY, required[1])
    episode = ValidatedEpisode(directory=root, shots=shots, capture_contract=descriptor)
    return EpisodeAccepted(episode) if mode.materializes_frames else EpisodeSummary(episode)
