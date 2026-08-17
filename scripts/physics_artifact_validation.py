from __future__ import annotations

import hashlib
import json
import os
import stat
from contextlib import ExitStack
from pathlib import Path
from typing import Final, TypeAlias

from scripts.physics_rollout_contract import MAX_TOTAL_BYTES
from scripts.physics_rollout_semantics import (
    PhysicsRolloutSemanticsError,
    SEMANTICS_METADATA_FIELDS,
    validate_physics_rollout_semantics,
)
from scripts.rollout_validation_types import (
    EpisodeRejected,
    EpisodeRejectionCode,
    PhysicsArtifactError,
    PhysicsArtifactSummary,
    reject as _reject,
)
from world_model.data.types import (
    LEGACY_RGB_V1,
    PHYSICS_CAPTURE_V1,
    CaptureContractDescriptor,
    ContractValueError,
    SidecarPath,
)

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
SHA256_LENGTH: Final = 64


def _confined_file(path: Path, root: Path) -> Path:
    if path.is_symlink():
        raise PhysicsArtifactError(str(path), "symlink artifact is forbidden")
    try:
        resolved_root = root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
        resolved_path.relative_to(resolved_root)
    except FileNotFoundError as error:
        raise PhysicsArtifactError(str(path), "missing artifact") from error
    except (OSError, ValueError) as error:
        raise PhysicsArtifactError(str(path), "artifact is outside shot root") from error
    if not resolved_path.is_file():
        raise PhysicsArtifactError(str(path), "artifact is not a regular file")
    return resolved_path


def _open_rooted(parent_fd: int, name: str, artifact: Path, *, directory: bool = False) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if directory:
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
        mode = os.fstat(descriptor).st_mode
        valid_type = stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
        if not valid_type:
            os.close(descriptor)
            raise PhysicsArtifactError(str(artifact), "artifact type changed during validation")
        return descriptor
    except PhysicsArtifactError:
        raise
    except OSError as error:
        raise PhysicsArtifactError(str(artifact), "symlink or changed artifact is forbidden") from error


def _duplicate_from_start(descriptor: int) -> int:
    os.lseek(descriptor, 0, os.SEEK_SET)
    return os.dup(descriptor)


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    with os.fdopen(_duplicate_from_start(descriptor), "rb") as artifact:
        for chunk in iter(lambda: artifact.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_descriptor(descriptor: int, *, max_bytes: int) -> JsonObject | None:
    try:
        if os.fstat(descriptor).st_size > max_bytes:
            return None
        with os.fdopen(_duplicate_from_start(descriptor), "rb") as source:
            encoded = source.read(max_bytes + 1)
        if len(encoded) > max_bytes:
            return None
        payload: JsonValue = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
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


def _capture_contract(manifest: JsonObject, requested_contract: str | None) -> CaptureContractDescriptor | EpisodeRejected:
    if "capture_contract" not in manifest:
        return LEGACY_RGB_V1
    raw = manifest["capture_contract"]
    if raw == PHYSICS_CAPTURE_V1.contract_name:
        return _reject(EpisodeRejectionCode.MALFORMED_CAPTURE_CONTRACT, "capture_contract")
    if not isinstance(raw, dict):
        return _reject(EpisodeRejectionCode.MALFORMED_CAPTURE_CONTRACT, "capture_contract")
    name = raw.get("contract_name")
    if not isinstance(name, str):
        return _reject(EpisodeRejectionCode.MALFORMED_CAPTURE_CONTRACT, "capture_contract.contract_name")
    if name not in (LEGACY_RGB_V1.contract_name, PHYSICS_CAPTURE_V1.contract_name):
        return _reject(EpisodeRejectionCode.UNKNOWN_CAPTURE_CONTRACT, name)
    if name == PHYSICS_CAPTURE_V1.contract_name:
        if requested_contract != PHYSICS_CAPTURE_V1.contract_name:
            return _reject(EpisodeRejectionCode.UNSUPPORTED_CAPTURE_CONTRACT, name)
        required_hashes = (raw.get("player_sha256"), raw.get("protocol_sha256"), raw.get("archive_sha256"))
        if any(not isinstance(value, str) or len(value) != SHA256_LENGTH or any(character not in "0123456789abcdef" for character in value) for value in required_hashes):
            return _reject(EpisodeRejectionCode.MALFORMED_CAPTURE_CONTRACT, "capture_contract.provenance")
        capabilities = raw.get("declared_capabilities")
        sidecars = _sidecars(raw.get("sidecar_paths"))
        if capabilities != list(PHYSICS_CAPTURE_V1.declared_capabilities) or sidecars != PHYSICS_CAPTURE_V1.sidecar_paths:
            return _reject(EpisodeRejectionCode.MALFORMED_CAPTURE_CONTRACT, "capture_contract")
        return PHYSICS_CAPTURE_V1
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
        return descriptor
    if descriptor != LEGACY_RGB_V1:
        return _reject(EpisodeRejectionCode.UNKNOWN_CAPTURE_CONTRACT, name)
    return descriptor


def validate_physics_shot_artifact(shot_dir: Path) -> PhysicsArtifactSummary:
    from PIL import Image
    from scripts.physics_capture_contract import PhysicsContractError, load_physics_capture

    try:
        root_fd = os.open(shot_dir, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY)
    except OSError as error:
        raise PhysicsArtifactError(str(shot_dir), "shot root is a symlink or not a directory") from error
    with ExitStack() as resources:
        resources.callback(os.close, root_fd)
        metadata_path = shot_dir / "metadata.json"
        _confined_file(metadata_path, shot_dir)
        metadata_fd = _open_rooted(root_fd, "metadata.json", metadata_path)
        resources.callback(os.close, metadata_fd)
        metadata = _read_json_descriptor(metadata_fd, max_bytes=MAX_TOTAL_BYTES)
        if metadata is None:
            raise PhysicsArtifactError(str(metadata_path), "missing or malformed metadata")
        required_values = {
            "capture_contract": "physics_capture_v1",
            "schema_version": "physics_capture_v1",
            "protocol_version": 1,
            "physics_state_path": "physics_state.jsonl",
            "physics_events_path": "physics_events.jsonl",
            "sidecars_closed": True,
        }
        if any(metadata.get(key) != value for key, value in required_values.items()):
            raise PhysicsArtifactError(str(metadata_path), "capture contract metadata differs from v1")
        for field in ("player_sha256", "protocol_sha256", "archive_sha256"):
            value = metadata.get(field)
            if not isinstance(value, str) or len(value) != SHA256_LENGTH or any(character not in "0123456789abcdef" for character in value):
                raise PhysicsArtifactError(str(metadata_path), f"invalid {field}")

        state_path = shot_dir / "physics_state.jsonl"
        event_path = shot_dir / "physics_events.jsonl"
        _confined_file(state_path, shot_dir)
        state_fd = _open_rooted(root_fd, "physics_state.jsonl", state_path)
        resources.callback(os.close, state_fd)
        _confined_file(event_path, shot_dir)
        event_fd = _open_rooted(root_fd, "physics_events.jsonl", event_path)
        resources.callback(os.close, event_fd)
        try:
            capture = load_physics_capture(
                Path(f"/proc/self/fd/{state_fd}"),
                Path(f"/proc/self/fd/{event_fd}"),
            )
        except (OSError, PhysicsContractError) as error:
            raise PhysicsArtifactError(str(shot_dir), str(error)) from error

        frames_dir = shot_dir / "frames"
        if frames_dir.is_symlink() or not frames_dir.is_dir():
            raise PhysicsArtifactError(str(frames_dir), "frames artifact is not a directory")
        frames_fd = _open_rooted(root_fd, "frames", frames_dir, directory=True)
        resources.callback(os.close, frames_fd)
        expected_paths: list[Path] = []
        expected_names: list[str] = []
        for state in capture.states:
            relative_path = Path(state.rgb_frame.relative_path)
            if relative_path.is_absolute() or len(relative_path.parts) != 2 or relative_path.parts[0] != "frames" or relative_path.name in {"", ".", ".."}:
                raise PhysicsArtifactError(str(relative_path), "PNG path is outside shot root")
            expected_paths.append(shot_dir / relative_path)
            expected_names.append(relative_path.name)
        if tuple(sorted(os.listdir(frames_fd))) != tuple(expected_names) or len(capture.states) != metadata.get("physics_state_count") or len(capture.events) != metadata.get("physics_event_count") or len(capture.states) != metadata.get("frame_count"):
            raise PhysicsArtifactError(str(shot_dir), "PNG/state/event counts or paths differ")

        frame_hashes: list[str] = []
        for state, frame_path, frame_name in zip(capture.states, expected_paths, expected_names, strict=True):
            _confined_file(frame_path, shot_dir)
            frame_fd = _open_rooted(frames_fd, frame_name, frame_path)
            resources.callback(os.close, frame_fd)
            if state.rgb_frame.render_frame != state.clock.render_frame:
                raise PhysicsArtifactError(str(frame_path), "PNG/state render frame differs")
            try:
                with os.fdopen(_duplicate_from_start(frame_fd), "rb") as source, Image.open(source) as image:
                    image.verify()
                with os.fdopen(_duplicate_from_start(frame_fd), "rb") as source, Image.open(source) as image:
                    dimensions = image.size
            except OSError as error:
                raise PhysicsArtifactError(str(frame_path), "invalid PNG") from error
            if dimensions != (state.rgb_frame.width_pixels, state.rgb_frame.height_pixels):
                raise PhysicsArtifactError(str(frame_path), "PNG dimensions differ from state")
            frame_hashes.append(_sha256_descriptor(frame_fd))

        expected_checksums = [{"relative_path": str(path.relative_to(shot_dir)), "sha256": digest} for path, digest in zip(expected_paths, frame_hashes, strict=True)]
        state_hash = _sha256_descriptor(state_fd)
        event_hash = _sha256_descriptor(event_fd)
        if metadata.get("frame_checksums") != expected_checksums or metadata.get("physics_state_sha256") != state_hash or metadata.get("physics_events_sha256") != event_hash:
            raise PhysicsArtifactError(str(metadata_path), "recorded checksums differ from files")
        if any(field in metadata for field in SEMANTICS_METADATA_FIELDS):
            expected_identity = metadata.get("expected_initial_engine_state_identity")
            if expected_identity is not None and not isinstance(expected_identity, str):
                raise PhysicsArtifactError(str(metadata_path), "invalid expected initial engine state identity")
            scenario_context = metadata.get("scenario_context")
            if scenario_context is not None and not isinstance(scenario_context, dict):
                raise PhysicsArtifactError(str(metadata_path), "scenario_context is not an object")
            try:
                semantics = validate_physics_rollout_semantics(
                    capture,
                    expected_initial_engine_state_identity=expected_identity,
                )
            except PhysicsRolloutSemanticsError as error:
                raise PhysicsArtifactError(str(shot_dir), str(error)) from error
            if any(metadata.get(field) != value for field, value in semantics.items()):
                raise PhysicsArtifactError(str(metadata_path), "recorded rollout semantics differ from sidecars")
        return PhysicsArtifactSummary(len(capture.states), len(capture.events), tuple(frame_hashes), state_hash, event_hash)
