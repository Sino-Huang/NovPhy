"""Atomic persistence for collector-bound ``physics_capture_v2`` sidecars."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

from scripts.physics_capture_v2 import (
    SIDECAR,
    PhysicsCaptureV2Error,
    bind_physics_capture_v2_engine,
    load_physics_capture_v2,
    normalized_initial_engine_state_identity,
)


def _plain_json(value):
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def source_bindings_from_collection(
    scenario_manifest,
    collection_scenario,
    runtime_input,
    *,
    rollout_identity: str,
) -> dict[str, str]:
    """Resolve collector-owned identities from one frozen plan runtime input."""
    if not isinstance(rollout_identity, str) or not rollout_identity:
        raise PhysicsCaptureV2Error("stale source binding: rollout identity is missing")
    if (
        runtime_input.scenario_id != collection_scenario.scenario_id
        or runtime_input.scenario_identity != collection_scenario.identity
    ):
        raise PhysicsCaptureV2Error("stale source binding: collection scenario identity differs")
    projection = collection_scenario.scenario_manifest_projection
    if _plain_json(projection.get("scenario_manifest")) != _plain_json(
        scenario_manifest.scenario_manifest.to_dict()
    ):
        raise PhysicsCaptureV2Error("stale source binding: scenario manifest differs from collection plan")
    initial_identity = scenario_manifest.scenario_manifest.declared_initial_engine_state.identity
    if (
        initial_identity != collection_scenario.expected_initial_engine_state_identity
        or initial_identity != runtime_input.expected_initial_engine_state_identity
    ):
        raise PhysicsCaptureV2Error("stale source binding: initial engine state identity differs")
    intervention = next(
        (
            item
            for item in collection_scenario.interventions
            if item.id == runtime_input.intervention_id
            and item.identity == runtime_input.intervention_identity
        ),
        None,
    )
    if intervention is None:
        raise PhysicsCaptureV2Error("stale source binding: intervention differs from collection plan")
    manifest = scenario_manifest.scenario_manifest
    generation = getattr(manifest, "generation", None)
    generation_mode = getattr(generation, "mode", "generated")
    if generation_mode == "generated":
        if manifest.scenario_template.identity != scenario_manifest.template_record.identity:
            raise PhysicsCaptureV2Error("stale source binding: scenario template identity differs")
    elif (
        generation_mode != "legacy_static"
        or manifest.scenario_template.identity is not None
        or generation.source_path != scenario_manifest.template_record.source_reference
    ):
        raise PhysicsCaptureV2Error("stale source binding: legacy source identity differs")
    return {
        "scenario_template_id": scenario_manifest.template_record.identity,
        "level_instance_id": manifest.level_instance.identity,
        "scenario_lineage_id": manifest.scenario_lineage.identity,
        "rollout_id": rollout_identity,
        "intervention_id": intervention.identity,
    }


def persist_physics_capture_v2(
    output_dir: Path,
    engine_record: Mapping[str, object],
    *,
    source_bindings: Mapping[str, object],
    scenario_manifest_identity: str,
) -> dict[str, object]:
    """Validate the complete capture before atomically publishing its sidecar."""
    if not isinstance(scenario_manifest_identity, str) or not scenario_manifest_identity:
        raise ValueError("scenario_manifest_identity must be a nonempty string")
    capture = bind_physics_capture_v2_engine(engine_record, source_bindings)
    encoded = (
        json.dumps(
            capture.record,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / SIDECAR
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=output_dir, prefix=f".{SIDECAR}.", delete=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    record = capture.record
    return {
        "physics_capture_v2_path": SIDECAR,
        "physics_capture_v2_schema": record["schema_version"],
        "capture_id": capture.capture_id,
        "shot_id": capture.shot_id,
        "configured_fixed_step_capture_stride": capture.configured_fixed_step_capture_stride,
        "causal_entity_count": len(record["causal_entities"]),
        "collider_count": len(record["colliders"]),
        "fixed_step_sample_count": len(record["fixed_step_samples"]),
        "frame_record_count": len(record["frame_records"]),
        "event_count": len(record["events"]),
        "initial_engine_state_identity": normalized_initial_engine_state_identity(capture),
        "scenario_manifest_identity": scenario_manifest_identity,
    }


def validate_physics_capture_v2_artifact(
    output_dir: Path,
    metadata: Mapping[str, object],
):
    """Revalidate sidecar bytes and their collector metadata before publication."""
    if metadata.get("physics_capture_v2_path") != SIDECAR:
        raise PhysicsCaptureV2Error("physics_capture_v2 metadata path is stale")
    sidecar = output_dir / SIDECAR
    try:
        sidecar.stat()
    except OSError as error:
        raise PhysicsCaptureV2Error("physics_capture_v2 sidecar is missing") from error
    capture = load_physics_capture_v2(sidecar)
    record = capture.record
    expected = {
        "physics_capture_v2_schema": record["schema_version"],
        "capture_id": capture.capture_id,
        "shot_id": capture.shot_id,
        "configured_fixed_step_capture_stride": capture.configured_fixed_step_capture_stride,
        "causal_entity_count": len(record["causal_entities"]),
        "collider_count": len(record["colliders"]),
        "fixed_step_sample_count": len(record["fixed_step_samples"]),
        "frame_record_count": len(record["frame_records"]),
        "event_count": len(record["events"]),
        "initial_engine_state_identity": normalized_initial_engine_state_identity(capture),
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise PhysicsCaptureV2Error(f"physics_capture_v2 metadata {field} is stale")
    if not isinstance(metadata.get("scenario_manifest_identity"), str) or not metadata["scenario_manifest_identity"]:
        raise PhysicsCaptureV2Error("physics_capture_v2 metadata scenario manifest identity is missing")
    return capture
