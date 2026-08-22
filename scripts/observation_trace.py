"""Versioned synchronized agent/canonical observation artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from PIL import Image


SCHEMA = "observation_trace_manifest_v1"
MANIFEST_NAME = "observation_trace_manifest.json"
EXPOSURE_ROLES = frozenset({
    "training", "calibration", "model_selection", "final_evaluation",
})
TRANSFORMS: dict[str, dict[str, Any]] = {
    "agent_rgb8_native_v1": {
        "method": "identity",
        "output_width_pixels": None,
        "output_height_pixels": None,
        "resampling": "none",
    },
    "agent_rgb8_nearest_2x2_v1": {
        "method": "resize",
        "output_width_pixels": 2,
        "output_height_pixels": 2,
        "resampling": "nearest",
    },
    "agent_rgb8_nearest_320x240_v1": {
        "method": "resize",
        "output_width_pixels": 320,
        "output_height_pixels": 240,
        "resampling": "nearest",
    },
}


class ObservationTraceError(ValueError):
    """The observation trace is malformed, stale, or inaccessible."""


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ObservationTraceError(f"{name} must be a finite number")
    return float(value)


def _vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ObservationTraceError(f"{name} is incomplete")
    return [_number(item, name) for item in value]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _identity(namespace: str, value: Any) -> str:
    return f"{namespace}:sha256:{hashlib.sha256(_canonical_json(value)).hexdigest()}"


def _content_identity(png: bytes) -> str:
    return f"rgb-png-v1:sha256:{hashlib.sha256(png).hexdigest()}"


def _png_size(png: bytes) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(png)) as image:
            image.verify()
        with Image.open(io.BytesIO(png)) as image:
            if image.mode != "RGB":
                raise ObservationTraceError("observation PNG must use RGB pixels")
            return image.size
    except (OSError, ValueError) as error:
        if isinstance(error, ObservationTraceError):
            raise
        raise ObservationTraceError("observation artifact is not a valid PNG") from error


def _validate_engine_capture(capture: Mapping[str, Any], canonical_png: bytes) -> None:
    required = {
        "schema_version", "capture_id", "sequence", "source_frame_identity",
        "render_frame", "render_time_seconds", "fixed_step", "fixed_time_seconds",
        "source", "camera", "viewport", "coordinates",
        "world_to_observation_transform",
    }
    if set(capture) != required:
        raise ObservationTraceError("observation capture metadata fields are incomplete")
    if capture["schema_version"] != "observation_capture_engine_v1":
        raise ObservationTraceError("observation capture engine schema is unsupported")
    if capture["source"] != "synchronized_observation_endpoint":
        raise ObservationTraceError(
            "desktop and ordinary screenshot sources are not synchronized observations"
        )
    capture_id = capture["capture_id"]
    if not isinstance(capture_id, str) or not capture_id:
        raise ObservationTraceError("observation capture identity is missing")
    sequence = capture["sequence"]
    render_frame = capture["render_frame"]
    fixed_step = capture["fixed_step"]
    for value, name, minimum in (
        (sequence, "capture sequence", 1),
        (render_frame, "render frame", 0),
        (fixed_step, "fixed step", 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ObservationTraceError(f"{name} is invalid")
    _number(capture["render_time_seconds"], "render time")
    _number(capture["fixed_time_seconds"], "fixed time")
    expected_source = (
        f"source-frame-v1:{capture_id}:{sequence}:{render_frame}:{fixed_step}"
    )
    if capture["source_frame_identity"] != expected_source:
        raise ObservationTraceError("source frame identity is stale")

    camera = capture["camera"]
    camera_fields = {
        "camera_identity", "projection_kind", "position_world", "rotation_xyzw",
        "orthographic_size_world_units", "vertical_field_of_view_degrees",
        "near_clip_world_units", "far_clip_world_units", "aspect_ratio",
        "world_to_camera_matrix", "camera_to_clip_matrix",
    }
    if not isinstance(camera, Mapping) or set(camera) != camera_fields:
        raise ObservationTraceError("camera state is incomplete")
    if not isinstance(camera["camera_identity"], str) or not camera["camera_identity"]:
        raise ObservationTraceError("camera identity is incomplete")
    if camera["projection_kind"] not in {"orthographic", "perspective"}:
        raise ObservationTraceError("camera projection is unsupported")
    _vector(camera["position_world"], 3, "camera position")
    _vector(camera["rotation_xyzw"], 4, "camera rotation")
    _vector(camera["world_to_camera_matrix"], 16, "camera world-to-camera matrix")
    _vector(camera["camera_to_clip_matrix"], 16, "camera projection matrix")
    near = _number(camera["near_clip_world_units"], "camera near clip")
    far = _number(camera["far_clip_world_units"], "camera far clip")
    aspect = _number(camera["aspect_ratio"], "camera aspect ratio")
    if near < 0 or far <= near or aspect <= 0:
        raise ObservationTraceError("camera clip or aspect declaration is invalid")
    if camera["projection_kind"] == "orthographic":
        if _number(camera["orthographic_size_world_units"], "camera orthographic size") <= 0:
            raise ObservationTraceError("camera orthographic size is invalid")
        if camera["vertical_field_of_view_degrees"] is not None:
            raise ObservationTraceError("orthographic camera declares a field of view")
    else:
        if camera["orthographic_size_world_units"] is not None:
            raise ObservationTraceError("perspective camera declares an orthographic size")
        field_of_view = _number(
            camera["vertical_field_of_view_degrees"], "camera field of view"
        )
        if not 0 < field_of_view < 180:
            raise ObservationTraceError("camera field of view is invalid")

    viewport = capture["viewport"]
    viewport_fields = {
        "width_pixels", "height_pixels", "camera_pixel_rect",
        "screen_width_pixels", "screen_height_pixels", "pixel_origin",
    }
    if not isinstance(viewport, Mapping) or set(viewport) != viewport_fields:
        raise ObservationTraceError("viewport metadata is incomplete")
    width, height = _png_size(canonical_png)
    for field, expected in (
        ("width_pixels", width), ("height_pixels", height),
        ("screen_width_pixels", width), ("screen_height_pixels", height),
    ):
        if viewport[field] != expected:
            raise ObservationTraceError("viewport dimensions differ from canonical observation")
    rect = _vector(viewport["camera_pixel_rect"], 4, "viewport camera pixel rect")
    if rect[2] <= 0 or rect[3] <= 0 or viewport["pixel_origin"] != "bottom_left":
        raise ObservationTraceError("viewport declaration is invalid")

    expected_coordinates = {
        "world_space": "unity_world_2d",
        "world_units": "unity_unit",
        "observation_space": "rgb_pixel",
        "observation_units": "pixel",
        "observation_origin": "top_left",
        "observation_x_axis": "right",
        "observation_y_axis": "down",
        "channel_order": "RGB",
        "sample_type": "uint8",
        "color_space": "sRGB",
    }
    if capture["coordinates"] != expected_coordinates:
        raise ObservationTraceError("coordinate and unit declarations are incomplete")

    transform = capture["world_to_observation_transform"]
    transform_fields = {
        "method", "world_to_camera_matrix", "camera_to_clip_matrix",
        "clip_to_ndc", "ndc_to_observation_matrix",
    }
    if not isinstance(transform, Mapping) or set(transform) != transform_fields:
        raise ObservationTraceError("world-to-observation transform is incomplete")
    if (
        transform["method"] != "unity_world_to_clip_to_top_left_pixel_v1"
        or transform["clip_to_ndc"] != "homogeneous_divide"
        or transform["world_to_camera_matrix"] != camera["world_to_camera_matrix"]
        or transform["camera_to_clip_matrix"] != camera["camera_to_clip_matrix"]
    ):
        raise ObservationTraceError("world-to-observation transform is inconsistent")
    expected_ndc = [
        width / 2.0, 0.0, width / 2.0,
        0.0, -height / 2.0, height / 2.0,
        0.0, 0.0, 1.0,
    ]
    if _vector(
        transform["ndc_to_observation_matrix"], 9,
        "world-to-observation NDC matrix",
    ) != expected_ndc:
        raise ObservationTraceError("world-to-observation transform differs from viewport")


def _configuration(name: str) -> dict[str, Any]:
    transform = TRANSFORMS.get(name)
    if transform is None:
        raise ObservationTraceError("observation transform is undeclared")
    value = {
        "schema": "observation_configuration_v1",
        "name": name,
        "canonical_representation": {
            "role": "canonical",
            "stage": "pre_transform",
            "media_type": "image/png",
            "pixel_format": "rgb8_srgb",
        },
        "agent_representation": {
            "role": "agent",
            "stage": "post_transform",
            "media_type": "image/png",
            "pixel_format": "rgb8_srgb",
            "transform": dict(transform),
        },
    }
    value["identity"] = _identity("observation-configuration-v1", value)
    return value


def _transform(png: bytes, configuration: Mapping[str, Any]) -> bytes:
    transform = configuration["agent_representation"]["transform"]
    if transform["method"] == "identity":
        return png
    try:
        with Image.open(io.BytesIO(png)) as source:
            image = source.convert("RGB").resize(
                (
                    transform["output_width_pixels"],
                    transform["output_height_pixels"],
                ),
                Image.Resampling.NEAREST,
            )
            output = io.BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()
    except OSError as error:
        raise ObservationTraceError("canonical observation cannot be transformed") from error


def _artifact(
    *,
    role: str,
    relative_path: str,
    png: bytes,
    synchronization_identity: str,
    configuration_identity: str,
) -> dict[str, Any]:
    width, height = _png_size(png)
    value = {
        "role": role,
        "stage": "post_transform" if role == "agent" else "pre_transform",
        "relative_path": relative_path,
        "media_type": "image/png",
        "pixel_format": "rgb8_srgb",
        "width_pixels": width,
        "height_pixels": height,
        "content_identity": _content_identity(png),
        "synchronization_identity": synchronization_identity,
        "observation_configuration_identity": configuration_identity,
    }
    value["identity"] = _identity(f"{role}-observation-v1", value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact_bytes(root: Path, artifact: Mapping[str, Any], role: str) -> bytes:
    relative = artifact.get("relative_path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ObservationTraceError(f"{role} observation path is invalid")
    path = root / relative
    try:
        if not path.is_file() or path.is_symlink():
            raise OSError
        return path.read_bytes()
    except OSError as error:
        raise ObservationTraceError(f"{role} observation is missing") from error


def persist_observation_trace(
    root: Path,
    captures: Sequence[Mapping[str, Any]],
    *,
    observation_configuration: str,
    source_bindings: Mapping[str, str],
    exposure_role: str,
) -> dict[str, Any]:
    """Publish one immutable trace from synchronized engine endpoint captures."""
    target = Path(root)
    if target.exists():
        raise ObservationTraceError("observation trace destination already exists")
    if not captures:
        raise ObservationTraceError("observation trace requires at least one frame record")
    expected_bindings = {
        "scenario_template_identity",
        "level_instance_identity",
        "source_scenario_lineage_identity",
        "rollout_identity",
    }
    if set(source_bindings) != expected_bindings or any(
        not isinstance(source_bindings[key], str) or not source_bindings[key]
        for key in expected_bindings
    ):
        raise ObservationTraceError("observation source bindings are incomplete")
    if exposure_role not in EXPOSURE_ROLES:
        raise ObservationTraceError("observation exposure role is unknown")

    configuration = _configuration(observation_configuration)
    bound_lineage = _identity("scenario-lineage-observation-v1", {
        "source_scenario_lineage_identity": source_bindings[
            "source_scenario_lineage_identity"
        ],
        "observation_configuration_identity": configuration["identity"],
    })
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        (staging / "observations" / "canonical").mkdir(parents=True)
        (staging / "observations" / "agent").mkdir(parents=True)
        frames = []
        source_frame_identities = []
        for index, raw in enumerate(captures):
            capture = dict(raw)
            canonical_png = capture.pop("canonical_png", None)
            if not isinstance(canonical_png, bytes):
                raise ObservationTraceError("canonical observation is missing")
            _validate_engine_capture(capture, canonical_png)
            source_frame_identity = capture.get("source_frame_identity")
            if not isinstance(source_frame_identity, str) or not source_frame_identity:
                raise ObservationTraceError("source frame identity is missing")
            source_frame_identities.append(source_frame_identity)
            if len(source_frame_identities) != len(set(source_frame_identities)):
                raise ObservationTraceError("observation trace has a duplicated source frame")
            synchronization_identity = _identity("observation-synchronization-v1", {
                "source_frame_identity": source_frame_identity,
                "fixed_step": capture.get("fixed_step"),
                "render_frame": capture.get("render_frame"),
            })
            agent_png = _transform(canonical_png, configuration)
            canonical_path = f"observations/canonical/frame_{index:06d}.png"
            agent_path = f"observations/agent/frame_{index:06d}.png"
            (staging / canonical_path).write_bytes(canonical_png)
            (staging / agent_path).write_bytes(agent_png)
            canonical = _artifact(
                role="canonical",
                relative_path=canonical_path,
                png=canonical_png,
                synchronization_identity=synchronization_identity,
                configuration_identity=configuration["identity"],
            )
            agent = _artifact(
                role="agent",
                relative_path=agent_path,
                png=agent_png,
                synchronization_identity=synchronization_identity,
                configuration_identity=configuration["identity"],
            )
            frame = {
                "source_frame_identity": source_frame_identity,
                "synchronization_identity": synchronization_identity,
                "render_frame": capture.get("render_frame"),
                "render_time_seconds": capture.get("render_time_seconds"),
                "fixed_step": capture.get("fixed_step"),
                "fixed_time_seconds": capture.get("fixed_time_seconds"),
                "capture_metadata": capture,
                "agent_observation": agent,
                "canonical_observation": canonical,
            }
            frame["identity"] = _identity("observation-frame-record-v1", {
                "scenario_lineage_identity": bound_lineage,
                "source_frame_identity": source_frame_identity,
                "synchronization_identity": synchronization_identity,
            })
            frames.append(frame)

        manifest = {
            "schema": SCHEMA,
            "identity": "",
            "exposure_role": exposure_role,
            "scenario_lineage_identity": bound_lineage,
            "source_bindings": dict(source_bindings),
            "observation_configuration": configuration,
            "frame_records": frames,
            "access_policy": {
                "agent": "central_model_workflows",
                "canonical": "alignment_or_capture_diagnosis_only",
            },
        }
        manifest["identity"] = _identity("observation-trace-manifest-v1", {
            key: value for key, value in manifest.items() if key != "identity"
        })
        _write_json(staging / MANIFEST_NAME, manifest)
        validate_observation_trace(staging)
        os.replace(staging, target)
        return manifest
    finally:
        if staging.exists():
            for path in sorted(staging.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            staging.rmdir()


def capture_observation_trace(
    bridge: Any,
    root: Path,
    *,
    frame_count: int,
    observation_configuration: str,
    source_bindings: Mapping[str, str],
    exposure_role: str,
) -> dict[str, Any]:
    """Capture typed request-72 frames and publish them as one trace."""
    from src.webui.bridge import ObservationCaptureEngine

    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ObservationTraceError("observation frame count must be a positive integer")
    captures = []
    for _ in range(frame_count):
        capture = bridge.get_observation_capture()
        if not isinstance(capture, ObservationCaptureEngine):
            raise ObservationTraceError("request-72 returned no typed observation capture")
        record = _plain_json(capture.metadata)
        record["canonical_png"] = capture.canonical_png
        captures.append(record)
    return persist_observation_trace(
        root,
        captures,
        observation_configuration=observation_configuration,
        source_bindings=source_bindings,
        exposure_role=exposure_role,
    )


def validate_observation_trace(root: Path) -> dict[str, Any]:
    """Validate trace identities and exact canonical-to-agent transformation."""
    path = Path(root) / MANIFEST_NAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ObservationTraceError("observation trace manifest is missing or malformed") from error
    manifest_fields = {
        "schema", "identity", "exposure_role", "scenario_lineage_identity",
        "source_bindings", "observation_configuration", "frame_records",
        "access_policy",
    }
    if not isinstance(value, dict) or set(value) != manifest_fields or value.get("schema") != SCHEMA:
        raise ObservationTraceError("observation trace manifest schema is unsupported")
    expected_identity = _identity("observation-trace-manifest-v1", {
        key: item for key, item in value.items() if key != "identity"
    })
    if value.get("identity") != expected_identity:
        raise ObservationTraceError("observation trace manifest identity is stale")
    raw_configuration = value.get("observation_configuration")
    if not isinstance(raw_configuration, Mapping):
        raise ObservationTraceError("observation configuration is missing")
    configuration = _configuration(raw_configuration.get("name"))
    if value.get("observation_configuration") != configuration:
        raise ObservationTraceError("observation configuration identity is stale")
    bindings = value.get("source_bindings")
    binding_fields = {
        "scenario_template_identity", "level_instance_identity",
        "source_scenario_lineage_identity", "rollout_identity",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != binding_fields or any(
        not isinstance(bindings[field], str) or not bindings[field]
        for field in binding_fields
    ):
        raise ObservationTraceError("observation source bindings are incomplete")
    expected_lineage = _identity("scenario-lineage-observation-v1", {
        "source_scenario_lineage_identity": bindings["source_scenario_lineage_identity"],
        "observation_configuration_identity": configuration["identity"],
    })
    if value.get("scenario_lineage_identity") != expected_lineage:
        raise ObservationTraceError("observation configuration is not bound to scenario lineage")
    if value.get("exposure_role") not in EXPOSURE_ROLES:
        raise ObservationTraceError("observation exposure role is unknown")
    if value.get("access_policy") != {
        "agent": "central_model_workflows",
        "canonical": "alignment_or_capture_diagnosis_only",
    }:
        raise ObservationTraceError("observation access policy is stale")
    frames = value.get("frame_records")
    if not isinstance(frames, list) or not frames:
        raise ObservationTraceError("observation trace requires frame records")
    frame_ids = []
    source_frame_ids = []
    observation_ids = []
    for frame in frames:
        frame_fields = {
            "identity", "source_frame_identity", "synchronization_identity",
            "render_frame", "render_time_seconds", "fixed_step", "fixed_time_seconds",
            "capture_metadata", "agent_observation", "canonical_observation",
        }
        if not isinstance(frame, Mapping) or set(frame) != frame_fields:
            raise ObservationTraceError("observation frame record fields are incomplete")
        canonical = frame["canonical_observation"]
        agent = frame["agent_observation"]
        if not isinstance(canonical, Mapping) or not isinstance(agent, Mapping):
            raise ObservationTraceError("agent or canonical observation is missing")
        canonical_png = _artifact_bytes(Path(root), canonical, "canonical")
        agent_png = _artifact_bytes(Path(root), agent, "agent")
        metadata = frame["capture_metadata"]
        if not isinstance(metadata, Mapping):
            raise ObservationTraceError("observation capture metadata is incomplete")
        _validate_engine_capture(metadata, canonical_png)
        synchronization_identity = _identity("observation-synchronization-v1", {
            "source_frame_identity": frame["source_frame_identity"],
            "fixed_step": frame["fixed_step"],
            "render_frame": frame["render_frame"],
        })
        if (
            frame["source_frame_identity"] != metadata["source_frame_identity"]
            or frame["render_frame"] != metadata["render_frame"]
            or frame["render_time_seconds"] != metadata["render_time_seconds"]
            or frame["fixed_step"] != metadata["fixed_step"]
            or frame["fixed_time_seconds"] != metadata["fixed_time_seconds"]
            or frame["synchronization_identity"] != synchronization_identity
        ):
            raise ObservationTraceError("observation alignment metadata differs")
        expected_frame_identity = _identity("observation-frame-record-v1", {
            "scenario_lineage_identity": expected_lineage,
            "source_frame_identity": frame["source_frame_identity"],
            "synchronization_identity": synchronization_identity,
        })
        if frame["identity"] != expected_frame_identity:
            raise ObservationTraceError("observation frame record identity is stale")
        for role, artifact, artifact_png in (
            ("canonical", canonical, canonical_png),
            ("agent", agent, agent_png),
        ):
            expected_artifact = _artifact(
                role=role,
                relative_path=artifact.get("relative_path"),
                png=artifact_png,
                synchronization_identity=synchronization_identity,
                configuration_identity=configuration["identity"],
            )
            if artifact != expected_artifact:
                raise ObservationTraceError(f"{role} observation identity or metadata is stale")
            observation_ids.append(artifact["identity"])
        if canonical["identity"] == agent["identity"]:
            raise ObservationTraceError("agent and canonical observation identities are duplicated")
        if agent_png != _transform(canonical_png, configuration):
            raise ObservationTraceError("agent observation is not the declared exact transform")
        frame_ids.append(frame["identity"])
        source_frame_ids.append(frame["source_frame_identity"])
    if len(frame_ids) != len(set(frame_ids)) or len(source_frame_ids) != len(set(source_frame_ids)):
        raise ObservationTraceError("observation frame identity is duplicated")
    if len(observation_ids) != len(set(observation_ids)):
        raise ObservationTraceError("cross-role observation reuse is prohibited")
    return value


def validate_observation_exposure_boundaries(
    manifests: Sequence[Mapping[str, Any]],
) -> None:
    """Reject a source lineage or observation reused across exposure roles."""
    roles_by_lineage: dict[str, set[str]] = {}
    roles_by_source_frame: dict[str, set[str]] = {}
    roles_by_observation: dict[str, set[str]] = {}
    for manifest in manifests:
        if manifest.get("schema") != SCHEMA:
            raise ObservationTraceError("observation trace manifest schema is unsupported")
        expected_identity = _identity("observation-trace-manifest-v1", {
            key: value for key, value in manifest.items() if key != "identity"
        })
        if manifest.get("identity") != expected_identity:
            raise ObservationTraceError("observation trace manifest identity is stale")
        role = manifest.get("exposure_role")
        source = manifest.get("source_bindings", {}).get(
            "source_scenario_lineage_identity"
        )
        if role not in EXPOSURE_ROLES or not isinstance(source, str) or not source:
            raise ObservationTraceError("observation exposure provenance is incomplete")
        roles_by_lineage.setdefault(source, set()).add(role)
        for frame in manifest.get("frame_records", []):
            roles_by_source_frame.setdefault(frame["source_frame_identity"], set()).add(role)
            for artifact_role in ("agent_observation", "canonical_observation"):
                identity = frame[artifact_role]["identity"]
                roles_by_observation.setdefault(identity, set()).add(role)
    if any(len(roles) > 1 for roles in roles_by_lineage.values()):
        raise ObservationTraceError(
            "observation configuration has cross exposure reuse under one source lineage"
        )
    if any(len(roles) > 1 for roles in roles_by_source_frame.values()):
        raise ObservationTraceError("source frame has cross exposure observation reuse")
    if any(len(roles) > 1 for roles in roles_by_observation.values()):
        raise ObservationTraceError("observation artifact has cross exposure reuse")


def _authorize_observation_access(
    observation_role: str,
    workflow_kind: str,
    purpose: str,
) -> None:
    if observation_role == "agent":
        if workflow_kind in EXPOSURE_ROLES and purpose in {
            "model_input", "reported_model_input", "comparator_selection",
        }:
            return
        raise ObservationTraceError("agent observation access is unauthorized")
    if observation_role == "canonical":
        if workflow_kind == "diagnostic" and purpose in {
            "alignment_diagnosis", "capture_diagnosis",
        }:
            return
        raise ObservationTraceError(
            "canonical observation access is restricted to alignment/capture diagnosis"
        )
    raise ObservationTraceError("observation role is unknown")


def audit_observation_access(
    manifest: Mapping[str, Any],
    attempts: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Record allowed and rejected access decisions without exposing image bytes."""
    if manifest.get("schema") != SCHEMA:
        raise ObservationTraceError("observation trace manifest schema is unsupported")
    expected_manifest_identity = _identity("observation-trace-manifest-v1", {
        key: value for key, value in manifest.items() if key != "identity"
    })
    if manifest.get("identity") != expected_manifest_identity:
        raise ObservationTraceError("observation trace manifest identity is stale")
    decisions = []
    attempt_ids = []
    fields = {
        "attempt_identity", "observation_role", "workflow_kind", "purpose",
    }
    for attempt in attempts:
        if not isinstance(attempt, Mapping) or set(attempt) != fields or any(
            not isinstance(attempt[field], str) or not attempt[field] for field in fields
        ):
            raise ObservationTraceError("observation access attempt is malformed")
        attempt_ids.append(attempt["attempt_identity"])
        try:
            _authorize_observation_access(
                attempt["observation_role"],
                attempt["workflow_kind"],
                attempt["purpose"],
            )
        except ObservationTraceError as error:
            allowed = False
            reason = str(error)
        else:
            allowed = True
            reason = "authorized_by_observation_access_policy_v1"
        decisions.append({**dict(attempt), "allowed": allowed, "reason": reason})
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ObservationTraceError("observation access attempt identities are duplicated")
    report = {
        "schema": "observation_access_audit_v1",
        "identity": "",
        "observation_trace_manifest_identity": manifest["identity"],
        "decisions": decisions,
        "passed": True,
    }
    report["identity"] = _identity("observation-access-audit-v1", {
        key: value for key, value in report.items() if key != "identity"
    })
    return report


def load_observation_bytes(
    root: Path,
    *,
    frame_record_identity: str,
    observation_role: str,
    workflow_kind: str,
    purpose: str,
) -> bytes:
    """Read an observation only after validating artifact and access policy."""
    manifest = validate_observation_trace(root)
    _authorize_observation_access(observation_role, workflow_kind, purpose)
    frame = next(
        (item for item in manifest["frame_records"] if item["identity"] == frame_record_identity),
        None,
    )
    if frame is None:
        raise ObservationTraceError("observation frame record is undeclared")
    return (Path(root) / frame[f"{observation_role}_observation"]["relative_path"]).read_bytes()


__all__ = [
    "MANIFEST_NAME",
    "ObservationTraceError",
    "TRANSFORMS",
    "audit_observation_access",
    "capture_observation_trace",
    "load_observation_bytes",
    "persist_observation_trace",
    "validate_observation_exposure_boundaries",
    "validate_observation_trace",
]
