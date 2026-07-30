from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Final, TypeAlias

from scripts.rollout_validation_types import (
    EpisodeAccepted, EpisodeRejected, EpisodeRejectionCode, EpisodeSummary,
    EpisodeValidationContract, EpisodeValidationMode, EpisodeValidationResult,
    ValidatedEpisode, ValidatedShot, reject as _reject,
)

from world_model.data.types import (
    LEGACY_RGB_V1,
    PHYSICS_CAPTURE_V1,
    CaptureContractDescriptor,
    ContractValueError,
    FrameRecord,
    ShotAction,
    SidecarPath,
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


def _read_json(path: Path) -> JsonObject | None:
    try:
        payload: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _sidecars(value: JsonValue) -> tuple[SidecarPath, ...] | None:
    if not isinstance(value, list):
        return None
    parsed: list[SidecarPath] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        path = item.get("relative_path")
        capabilities = item.get("capabilities")
        if not isinstance(path, str) or not isinstance(capabilities, list) or not all(isinstance(capability, str) for capability in capabilities):
            return None
        try:
            parsed.append(SidecarPath(path, tuple(capabilities)))
        except ContractValueError:
            return None
    return tuple(parsed)


def _capture_contract(manifest: JsonObject) -> CaptureContractDescriptor | EpisodeRejected:
    if "capture_contract" not in manifest:
        return LEGACY_RGB_V1
    raw = manifest["capture_contract"]
    if not isinstance(raw, dict):
        return _reject(EpisodeRejectionCode.MALFORMED_CAPTURE_CONTRACT, "capture_contract")
    name = raw.get("contract_name")
    if not isinstance(name, str):
        return _reject(EpisodeRejectionCode.MALFORMED_CAPTURE_CONTRACT, "capture_contract.contract_name")
    if name not in (LEGACY_RGB_V1.contract_name, PHYSICS_CAPTURE_V1.contract_name):
        return _reject(EpisodeRejectionCode.UNKNOWN_CAPTURE_CONTRACT, name)
    version = raw.get("contract_version")
    layout = raw.get("artifact_layout_version")
    player = raw.get("player_provenance")
    protocol = raw.get("protocol_provenance")
    capabilities = raw.get("declared_capabilities", [])
    sidecars = _sidecars(raw.get("sidecar_paths", []))
    valid_provenance = (player is None or isinstance(player, str)) and (protocol is None or isinstance(protocol, str))
    if not isinstance(version, str) or not isinstance(layout, str) or not valid_provenance or not isinstance(capabilities, list) or not all(isinstance(item, str) for item in capabilities) or sidecars is None:
        return _reject(EpisodeRejectionCode.MALFORMED_CAPTURE_CONTRACT, "capture_contract")
    try:
        descriptor = CaptureContractDescriptor(name, version, layout, player, protocol, tuple(capabilities), sidecars)
    except ContractValueError:
        return _reject(EpisodeRejectionCode.MALFORMED_CAPTURE_CONTRACT, "capture_contract")
    if descriptor == PHYSICS_CAPTURE_V1:
        return _reject(EpisodeRejectionCode.UNSUPPORTED_CAPTURE_CONTRACT, name)
    if descriptor != LEGACY_RGB_V1:
        return _reject(EpisodeRejectionCode.UNKNOWN_CAPTURE_CONTRACT, name)
    return descriptor


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
        for path, directory in ((shot_dir, True), (shot_dir / "metadata.json", False), (shot_dir / "frames", True), (shot_dir / "pre_shot.png", False)):
            rejection = _artifact_rejection(path, root, directory=directory)
            if rejection is not None:
                return rejection
        metadata = _read_json(shot_dir / "metadata.json")
        if metadata is None:
            return _reject(EpisodeRejectionCode.MALFORMED_JSON, shot_dir / "metadata.json")
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
    descriptor = _capture_contract(manifest)
    if isinstance(descriptor, EpisodeRejected):
        return descriptor
    valid_contract = (
        manifest.get("capture_source") == "capture_desktop_rollout"
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
    shots = _validated_shots(root, action_log, contract.count, mode=mode)
    if isinstance(shots, EpisodeRejected):
        return shots
    signs = tuple(-1 if shot.release_x < 0 else 1 for shot in shots)
    expected_signs = tuple(-1 if index % 2 == 0 else 1 for index in range(contract.count))
    if contract.level_five and signs != expected_signs:
        return _reject(EpisodeRejectionCode.LEVEL_FIVE_ACTION_POLICY, required[1])
    episode = ValidatedEpisode(directory=root, shots=shots, capture_contract=descriptor)
    return EpisodeAccepted(episode) if mode.materializes_frames else EpisodeSummary(episode)
