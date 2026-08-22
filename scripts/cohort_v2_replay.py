"""Version-bounded deterministic replay evidence for cohort v2."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from scripts.cohort_v2_partition import CohortV2PartitionExposureManifest
from scripts.cohort_v2_scenarios import load_cohort_v2_scenario_manifest
from scripts.observation_trace import ObservationTraceError, validate_observation_trace
from scripts.physics_capture_v2 import (
    PhysicsCaptureV2Error,
    load_physics_capture_v2,
    normalized_initial_engine_state_identity,
)


PLAN_SCHEMA = "cohort_v2_replay_plan_v1"
ATTEMPT_SCHEMA = "cohort_v2_replay_attempt_v1"
REPORT_SCHEMA = "cohort_v2_replay_evidence_v1"
BUNDLE_SCHEMA = "issue_48_cohort_v2_replay_evidence_bundle_v1"
PLAN_NAME = "replay-plan.json"
ATTEMPT_NAME = "attempt.json"
REPORT_NAME = "replay-evidence.json"
BUNDLE_NAME = "bundle-manifest.json"
FROZEN_COMMAND_NAME = "frozen-replay-command.json"
NON_FINAL_ROLES = frozenset({"training", "calibration", "model_selection"})


class CohortV2ReplayError(ValueError):
    """A cohort-v2 replay input or comparison failed validation."""


def semantic_identity(namespace: str, *parts: object) -> str:
    """Build a declared semantic identity without content digests."""
    if not namespace or not parts:
        raise CohortV2ReplayError("semantic identity requires a namespace and members")
    return ":".join((namespace, *(quote(str(part), safe="") for part in parts)))


def replay_version_envelope_identity(envelope: Mapping[str, Any]) -> str:
    return semantic_identity(
        "cohort-v2-replay-version-envelope-v1",
        envelope["unity_version"],
        envelope["player_stage_profile"],
        envelope["player_source_snapshot_commit"],
        envelope["player_source_tree"],
        envelope["player_declared_file_count"],
        envelope["physics_protocol_version"],
        envelope["observation_protocol_version"],
        envelope["physics_engine_contract"],
        envelope["physics_exporter_contract"],
        envelope["observation_exporter_contract"],
        envelope["observation_trace_contract"],
        envelope["generator_identity"],
        envelope["generator_version"],
        envelope["importer_identity"],
        envelope["importer_version"],
        envelope["code_revision"],
    )


def replay_intervention_identity(
    scenario_collection_id: str,
    interface_action: Mapping[str, Any],
    engine_relative_action: Mapping[str, Any],
) -> str:
    return semantic_identity(
        "cohort-v2-replay-intervention-v1",
        scenario_collection_id,
        interface_action["action_type"],
        interface_action["coordinate_frame"],
        interface_action["drag_release"][0],
        interface_action["drag_release"][1],
        interface_action["tapTime"],
        interface_action["releaseTime"],
        interface_action["frame_height"],
        engine_relative_action["schema"],
        engine_relative_action["drag_delta_canvas_pixels"][0],
        engine_relative_action["drag_delta_canvas_pixels"][1],
        engine_relative_action["tap_time_milliseconds"],
        engine_relative_action["hold_milliseconds"],
    )


def exact_socket_comparison_rules_v1() -> dict[str, Any]:
    """Return the #44-style exact frozen-command replay rules."""
    return {
        "schema": "cohort_v2_replay_exact_socket_comparison_rules_v1",
        "identity": (
            "cohort-v2-replay-exact-socket-comparison-rules-v1:"
            "launch-relative-step-delta=1:contact-separation-delta=0.001:"
            "frozen-socket-command=exact:camera-cross-attempt-equality=false:"
            "engine-state=exact:event-payload=exact:contact-point-normal=exact:"
            "pixel-equality-required=false"
        ),
        "fixed_step_origin": "bird_launched",
        "maximum_relative_fixed_step_delta": 1,
        "maximum_contact_separation_delta": 0.001,
        "repeated_same_semantic_event_count": "reported_tolerated",
        "engine_state_at_shared_launch_relative_fixed_steps": "exact",
        "event_payload": "exact",
        "contact_point_and_normal": "exact",
        "interface_action_authority": "original_attempt_frozen_socket_command_exact",
        "observation_cross_attempt_equality": "configuration_and_access_policy",
        "camera_viewport_and_transform": "validated_per_attempt_render_provenance",
        "render_frame_and_time": "observation_provenance_only",
        "pixel_equality_required": False,
    }


def comparison_rules_for_plan_version(plan_version: int) -> dict[str, Any]:
    if plan_version < 4:
        raise CohortV2ReplayError(
            "failed audit determinations are not executable replay policies"
        )
    return exact_socket_comparison_rules_v1()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CohortV2ReplayError(f"{label} is missing or malformed") from error
    if not isinstance(value, dict):
        raise CohortV2ReplayError(f"{label} must be an object")
    return value


def _require_exact_fields(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise CohortV2ReplayError(f"{label} fields are incomplete or unsupported")


def replay_scenario_collection_identity(scenario_collection: Mapping[str, Any]) -> str:
    return semantic_identity(
        "cohort-v2-replay-scenario-collection-v1",
        scenario_collection["scenario_collection_id"],
        scenario_collection["exposure_role"],
        scenario_collection["scenario_manifest_identity"],
        scenario_collection["scenario_specification_identity"],
        scenario_collection["scenario_content_identity"],
        scenario_collection["scenario_template_identity"],
        scenario_collection["level_instance_identity"],
        scenario_collection["scenario_lineage_identity"],
        scenario_collection["intervention"]["identity"],
        scenario_collection["observation_configuration_identity"],
        *scenario_collection["coverage_strata"],
    )


def replay_plan_identity(plan: Mapping[str, Any]) -> str:
    return semantic_identity(
        "cohort-v2-replay-plan-v1",
        plan["plan_version"],
        plan["version_envelope"]["identity"],
        plan["partition_manifest_identity"],
        plan["observation_capability_bundle_identity"],
        plan["observation_access_audit_identity"],
        plan["comparison_rules"]["identity"],
        *(scenario_collection["identity"] for scenario_collection in plan["scenario_collections"]),
    )


def validate_replay_plan(root: Path) -> dict[str, Any]:
    """Validate the plan, source manifests, partition, and representative scope."""
    base = Path(root)
    plan = _load_json(base / PLAN_NAME, "cohort-v2 replay plan")
    _require_exact_fields(
        plan,
        {
            "schema", "identity", "plan_version", "partition_manifest_identity",
            "observation_capability_bundle_identity", "observation_access_audit_identity",
            "version_envelope", "comparison_rules", "max_attempts_per_role", "scenario_collections",
        },
        "cohort-v2 replay plan",
    )
    if (
        plan["schema"] != PLAN_SCHEMA
        or isinstance(plan["plan_version"], bool)
        or not isinstance(plan["plan_version"], int)
        or plan["plan_version"] <= 0
    ):
        raise CohortV2ReplayError("cohort-v2 replay plan version is unsupported")
    expected_rules = comparison_rules_for_plan_version(plan["plan_version"])
    if plan["comparison_rules"] != expected_rules:
        raise CohortV2ReplayError("cohort-v2 replay comparison rules changed")
    if plan["max_attempts_per_role"] != 1:
        raise CohortV2ReplayError("cohort-v2 replay plan must permit one original and one replay attempt")
    envelope = plan["version_envelope"]
    envelope_fields = {
        "schema", "identity", "unity_version", "player_stage_profile",
        "player_source_snapshot_commit", "player_source_tree", "player_declared_file_count",
        "physics_protocol_version", "observation_protocol_version",
        "physics_engine_contract", "physics_exporter_contract",
        "observation_exporter_contract", "observation_trace_contract",
        "generator_identity", "generator_version", "importer_identity", "importer_version",
        "code_revision",
    }
    if not isinstance(envelope, Mapping):
        raise CohortV2ReplayError("cohort-v2 replay version envelope is missing")
    _require_exact_fields(envelope, envelope_fields, "cohort-v2 replay version envelope")
    expected_envelope_identity = replay_version_envelope_identity(envelope)
    if envelope["schema"] != "cohort_v2_replay_version_envelope_v1" or envelope["identity"] != expected_envelope_identity:
        raise CohortV2ReplayError("cohort-v2 replay version envelope identity is stale")

    partition_value = _load_json(base / "source/partition-exposure-manifest.json", "source partition manifest")
    partition = CohortV2PartitionExposureManifest.from_dict(partition_value)
    if partition.identity != plan["partition_manifest_identity"]:
        raise CohortV2ReplayError("replay plan partition binding is stale")
    observation_bundle = _load_json(
        base / "source/observation-evidence-bundle.json", "source observation evidence bundle"
    )
    if (
        observation_bundle.get("identity") != plan["observation_capability_bundle_identity"]
        or observation_bundle.get("access_audit_identity") != plan["observation_access_audit_identity"]
        or observation_bundle.get("passed") is not True
    ):
        raise CohortV2ReplayError("replay plan observation capability binding is stale")

    scenario_collections = plan["scenario_collections"]
    if not isinstance(scenario_collections, list) or len(scenario_collections) < 2:
        raise CohortV2ReplayError("representative replay requires at least two scenario collections")
    scenario_collection_fields = {
        "schema", "identity", "scenario_collection_id", "exposure_role", "scenario_manifest_identity",
        "scenario_specification_identity", "scenario_content_identity",
        "scenario_template_identity", "level_instance_identity", "scenario_lineage_identity",
        "source_manifest_relative_path", "source_xml_relative_path",
        "source_template_relative_path", "generation", "intervention",
        "observation_configuration", "observation_configuration_identity",
        "coverage_strata",
    }
    partition_by_lineage = {entry.scenario_lineage_identity: entry for entry in partition.entries}
    seen_scenario_collection_ids: set[str] = set()
    lineages: set[str] = set()
    levels: set[str] = set()
    templates: set[str] = set()
    strata: set[str] = set()
    for scenario_collection in scenario_collections:
        if not isinstance(scenario_collection, Mapping):
            raise CohortV2ReplayError("cohort-v2 replay scenario collection must be an object")
        _require_exact_fields(scenario_collection, scenario_collection_fields, "cohort-v2 replay scenario collection")
        if scenario_collection["schema"] != "cohort_v2_replay_scenario_collection_v1" or scenario_collection["identity"] != replay_scenario_collection_identity(scenario_collection):
            raise CohortV2ReplayError("cohort-v2 replay scenario collection identity is stale")
        if scenario_collection["scenario_collection_id"] in seen_scenario_collection_ids:
            raise CohortV2ReplayError("cohort-v2 replay scenario collection identity is duplicated")
        seen_scenario_collection_ids.add(scenario_collection["scenario_collection_id"])
        if scenario_collection["exposure_role"] not in NON_FINAL_ROLES:
            raise CohortV2ReplayError("cohort-v2 replay scenario collection must use a non-final exposure role")
        entry = partition_by_lineage.get(scenario_collection["scenario_lineage_identity"])
        expected_partition = {
            "exposure_role": scenario_collection["exposure_role"],
            "scenario_manifest_identity": scenario_collection["scenario_manifest_identity"],
            "scenario_specification_identity": scenario_collection["scenario_specification_identity"],
            "scenario_template_identity": scenario_collection["scenario_template_identity"],
            "level_instance_identity": scenario_collection["level_instance_identity"],
        }
        if entry is None or any(getattr(entry, field) != expected for field, expected in expected_partition.items()):
            raise CohortV2ReplayError("cohort-v2 replay scenario collection differs from the frozen partition")
        relative_paths = (
            Path(scenario_collection["source_manifest_relative_path"]),
            Path(scenario_collection["source_xml_relative_path"]),
            Path(scenario_collection["source_template_relative_path"]),
        )
        if any(path.is_absolute() or ".." in path.parts for path in relative_paths):
            raise CohortV2ReplayError("cohort-v2 replay source path escapes the evidence root")
        manifest_path, xml_path, template_path = (base / path for path in relative_paths)
        scenario = load_cohort_v2_scenario_manifest(
            manifest_path,
            xml_path=xml_path,
            template_source_path=template_path,
        )
        manifest = scenario.scenario_manifest
        expected_manifest = {
            "scenario_manifest_identity": scenario.identity,
            "scenario_specification_identity": manifest.scenario_specification.identity,
            "scenario_content_identity": manifest.scenario_specification.content_identity,
            "scenario_template_identity": manifest.scenario_template.identity,
            "level_instance_identity": manifest.level_instance.identity,
            "scenario_lineage_identity": manifest.scenario_lineage.identity,
        }
        if any(scenario_collection[field] != expected for field, expected in expected_manifest.items()):
            raise CohortV2ReplayError("cohort-v2 replay source manifest binding is stale")
        expected_generation = {
            "generator_identity": manifest.generation.generator_identity,
            "generator_version": manifest.generation.generator_version,
            "importer_identity": manifest.generation.importer_identity,
            "importer_version": manifest.generation.importer_version,
        }
        if scenario_collection["generation"] != expected_generation:
            raise CohortV2ReplayError("cohort-v2 replay generator/importer binding is stale")
        intervention = scenario_collection["intervention"]
        if not isinstance(intervention, Mapping) or set(intervention) != {
            "identity", "interface_action", "engine_relative_action"
        }:
            raise CohortV2ReplayError("cohort-v2 replay intervention is incomplete")
        action = intervention["interface_action"]
        engine_action = intervention["engine_relative_action"]
        if not isinstance(action, Mapping) or set(action) != {
            "action_type", "coordinate_frame", "drag_release", "tapTime",
            "releaseTime", "frame_height",
        }:
            raise CohortV2ReplayError("cohort-v2 replay interface action is unsupported")
        if not isinstance(engine_action, Mapping) or set(engine_action) != {
            "schema", "drag_delta_canvas_pixels", "tap_time_milliseconds",
            "hold_milliseconds",
        }:
            raise CohortV2ReplayError("cohort-v2 replay engine-relative action is unsupported")
        expected_intervention = replay_intervention_identity(
            scenario_collection["scenario_collection_id"],
            action,
            engine_action,
        )
        if intervention["identity"] != expected_intervention:
            raise CohortV2ReplayError("cohort-v2 replay intervention identity is stale")
        lineages.add(scenario_collection["scenario_lineage_identity"])
        levels.add(scenario_collection["level_instance_identity"])
        templates.add(scenario_collection["scenario_template_identity"])
        if not isinstance(scenario_collection["coverage_strata"], list):
            raise CohortV2ReplayError("cohort-v2 replay coverage strata are malformed")
        strata.update(scenario_collection["coverage_strata"])
    if min(len(lineages), len(levels), len(templates)) < 2:
        raise CohortV2ReplayError("representative replay requires two lineages, levels, and templates")
    if not strata.intersection({"collision", "contact"}) or not strata.intersection({"stable", "support"}):
        raise CohortV2ReplayError("representative replay lacks collision/contact or stable/support stratum")
    if plan["identity"] != replay_plan_identity(plan):
        raise CohortV2ReplayError("cohort-v2 replay plan identity is stale")
    return plan


def _attempt_identity(plan: Mapping[str, Any], scenario_collection: Mapping[str, Any], role: str) -> str:
    return semantic_identity("cohort-v2-replay-attempt-v1", plan["identity"], scenario_collection["scenario_collection_id"], role)


def _load_attempt(base: Path, plan: Mapping[str, Any], scenario_collection: Mapping[str, Any], role: str) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    attempt_root = base / "attempts" / scenario_collection["scenario_collection_id"] / role
    value = _load_json(attempt_root / ATTEMPT_NAME, f"{scenario_collection['scenario_collection_id']} {role} attempt")
    fields = {
        "schema", "identity", "attempt_role", "scenario_collection_identity", "rollout_identity",
        "version_envelope", "partition_manifest_identity", "collection_plan_identity",
        "exposure_role", "scenario_manifest_identity", "scenario_specification_identity",
        "scenario_content_identity", "scenario_template_identity", "level_instance_identity",
        "scenario_lineage_identity", "intervention_identity", "interface_action",
        "engine_relative_action", "physics_capture_relative_path", "physics_capture_metadata",
        "observation_trace_relative_path", "observation_trace_manifest_identity",
        "observation_configuration_identity",
    }
    _require_exact_fields(value, fields, f"{scenario_collection['scenario_collection_id']} {role} attempt")
    expected_identity = _attempt_identity(plan, scenario_collection, role)
    expected = {
        "schema": ATTEMPT_SCHEMA,
        "identity": expected_identity,
        "attempt_role": role,
        "scenario_collection_identity": scenario_collection["identity"],
        "rollout_identity": expected_identity,
        "version_envelope": plan["version_envelope"],
        "partition_manifest_identity": plan["partition_manifest_identity"],
        "collection_plan_identity": plan["identity"],
        "exposure_role": scenario_collection["exposure_role"],
        "scenario_manifest_identity": scenario_collection["scenario_manifest_identity"],
        "scenario_specification_identity": scenario_collection["scenario_specification_identity"],
        "scenario_content_identity": scenario_collection["scenario_content_identity"],
        "scenario_template_identity": scenario_collection["scenario_template_identity"],
        "level_instance_identity": scenario_collection["level_instance_identity"],
        "scenario_lineage_identity": scenario_collection["scenario_lineage_identity"],
        "intervention_identity": scenario_collection["intervention"]["identity"],
        "engine_relative_action": scenario_collection["intervention"]["engine_relative_action"],
        "observation_configuration_identity": scenario_collection["observation_configuration_identity"],
    }
    if any(value.get(field) != expected_value for field, expected_value in expected.items()):
        raise CohortV2ReplayError(f"{scenario_collection['scenario_collection_id']} {role} attempt binding is stale")
    physics_path = attempt_root / value["physics_capture_relative_path"]
    observation_path = attempt_root / value["observation_trace_relative_path"]
    if physics_path.parent != attempt_root or observation_path.parent != attempt_root:
        raise CohortV2ReplayError(f"{scenario_collection['scenario_collection_id']} {role} artifact path is invalid")
    try:
        physics = load_physics_capture_v2(physics_path)
        observation = validate_observation_trace(observation_path)
    except (PhysicsCaptureV2Error, ObservationTraceError) as error:
        raise CohortV2ReplayError(str(error)) from error
    metadata = value["physics_capture_metadata"]
    expected_metadata = {
        "physics_capture_v2_schema": physics.record["schema_version"],
        "capture_id": physics.capture_id,
        "shot_id": physics.shot_id,
        "configured_fixed_step_capture_stride": physics.configured_fixed_step_capture_stride,
        "causal_entity_count": len(physics.record["causal_entities"]),
        "collider_count": len(physics.record["colliders"]),
        "fixed_step_sample_count": len(physics.record["fixed_step_samples"]),
        "frame_record_count": len(physics.record["frame_records"]),
        "event_count": len(physics.record["events"]),
        "initial_engine_state_identity": normalized_initial_engine_state_identity(physics),
        "scenario_manifest_identity": scenario_collection["scenario_manifest_identity"],
    }
    if metadata != expected_metadata:
        raise CohortV2ReplayError(f"{scenario_collection['scenario_collection_id']} {role} physics metadata is stale")
    bindings = physics.record["source_bindings"]
    expected_bindings = {
        "scenario_template_id": scenario_collection["scenario_template_identity"],
        "level_instance_id": scenario_collection["level_instance_identity"],
        "scenario_lineage_id": scenario_collection["scenario_lineage_identity"],
        "rollout_id": expected_identity,
        "intervention_id": scenario_collection["intervention"]["identity"],
    }
    if bindings != expected_bindings:
        raise CohortV2ReplayError(f"{scenario_collection['scenario_collection_id']} {role} physics source binding is stale")
    observation_bindings = observation["source_bindings"]
    expected_observation_bindings = {
        "scenario_template_identity": scenario_collection["scenario_template_identity"],
        "level_instance_identity": scenario_collection["level_instance_identity"],
        "source_scenario_lineage_identity": scenario_collection["scenario_lineage_identity"],
        "rollout_identity": expected_identity,
    }
    if (
        observation_bindings != expected_observation_bindings
        or observation["identity"] != value["observation_trace_manifest_identity"]
        or observation["exposure_role"] != scenario_collection["exposure_role"]
        or observation["observation_configuration"]["identity"] != scenario_collection["observation_configuration_identity"]
    ):
        raise CohortV2ReplayError(f"{scenario_collection['scenario_collection_id']} {role} observation binding is stale")
    return value, physics, observation


def build_frozen_replay_command(
    plan: Mapping[str, Any],
    scenario_collection: Mapping[str, Any],
    original_attempt: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the original attempt's exact socket command for one replay."""
    interface_action = original_attempt["interface_action"]
    socket_command = interface_action["socket_command"]
    payload = {
        "schema": "cohort_v2_frozen_replay_command_v1",
        "identity": "",
        "plan_identity": plan["identity"],
        "scenario_collection_identity": scenario_collection["identity"],
        "original_attempt_identity": original_attempt["identity"],
        "intervention_identity": scenario_collection["intervention"]["identity"],
        "interface_action": interface_action,
        "expected_initial_engine_state_identity": original_attempt["physics_capture_metadata"][
            "initial_engine_state_identity"
        ],
        "max_replay_attempts": 1,
    }
    payload["identity"] = semantic_identity(
        "cohort-v2-frozen-replay-command-v1",
        plan["identity"],
        scenario_collection["identity"],
        original_attempt["identity"],
        socket_command["x"],
        socket_command["y"],
        socket_command["tapTime"],
        socket_command["releaseTime"],
    )
    return payload


def _load_frozen_replay_command(
    base: Path,
    plan: Mapping[str, Any],
    scenario_collection: Mapping[str, Any],
    original_attempt: Mapping[str, Any],
) -> dict[str, Any]:
    command = _load_json(
        base / "attempts" / scenario_collection["scenario_collection_id"] / FROZEN_COMMAND_NAME,
        f"{scenario_collection['scenario_collection_id']} frozen replay command",
    )
    expected = build_frozen_replay_command(plan, scenario_collection, original_attempt)
    if command != expected:
        raise CohortV2ReplayError(f"{scenario_collection['scenario_collection_id']} frozen replay command is stale")
    return command


def _component(name: str, status: str, rule: str, details: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "component": name,
        "status": status,
        "rule": rule,
        "details": dict(details),
    }


def _first_launch_step(record: Mapping[str, Any]) -> int:
    launches = [event["fixed_step"] for event in record["events"] if event["event_type"] == "bird_launched"]
    if len(launches) != 1:
        raise CohortV2ReplayError("physics capture must contain exactly one bird_launched event")
    return launches[0]


def _event_occurrences(record: Mapping[str, Any]) -> dict[tuple[str, tuple[str, ...]], list[int]]:
    origin = _first_launch_step(record)
    result: dict[tuple[str, tuple[str, ...]], list[int]] = defaultdict(list)
    for event in record["events"]:
        key = (event["event_type"], tuple(sorted(event["participants"])))
        result[key].append(event["fixed_step"] - origin)
    return dict(result)


def _relation_occurrences(record: Mapping[str, Any], kind: str) -> dict[tuple[str, str], list[int]]:
    origin = _first_launch_step(record)
    result: dict[tuple[str, str], list[int]] = defaultdict(list)
    for sample in record["fixed_step_samples"]:
        if kind == "contact":
            pairs = {
                tuple(sorted((item["entity_a_id"], item["entity_b_id"])))
                for item in sample["contacts"]
            }
        else:
            pairs = {
                (item["supporter_entity_id"], item["supported_entity_id"])
                for item in sample["supports"]
            }
        for pair in pairs:
            result[pair].append(sample["fixed_step"] - origin)
    return dict(result)


def _compare_occurrences(
    name: str,
    original: Mapping[Any, Sequence[int]],
    replay: Mapping[Any, Sequence[int]],
    maximum_delta: int,
) -> dict[str, Any]:
    original_keys = set(original)
    replay_keys = set(replay)
    if original_keys != replay_keys:
        return _component(
            name,
            "mismatch",
            "semantic identities and first launch-relative occurrence must match",
            {
                "missing_in_replay": sorted(map(str, original_keys - replay_keys)),
                "new_in_replay": sorted(map(str, replay_keys - original_keys)),
            },
        )
    deltas = {
        str(key): abs(original[key][0] - replay[key][0])
        for key in sorted(original_keys, key=str)
    }
    if any(delta > maximum_delta for delta in deltas.values()):
        return _component(
            name,
            "mismatch",
            "first launch-relative occurrence may differ by at most one fixed step",
            {"first_occurrence_step_deltas": deltas},
        )
    count_differences = {
        str(key): [len(original[key]), len(replay[key])]
        for key in sorted(original_keys, key=str)
        if len(original[key]) != len(replay[key])
    }
    if any(deltas.values()) or count_differences:
        return _component(
            name,
            "tolerated",
            "one fixed-step scheduling delta and repeated same-semantic occurrences are version-bound tolerances",
            {
                "first_occurrence_step_deltas": deltas,
                "repeated_occurrence_count_differences": count_differences,
            },
        )
    return _component(
        name,
        "equality",
        "semantic identities and launch-relative first occurrences are exact",
        {"semantic_identity_count": len(original_keys)},
    )


def _state_samples_by_relative_step(record: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    origin = _first_launch_step(record)
    result = {}
    for sample in record["fixed_step_samples"]:
        result[sample["fixed_step"] - origin] = {
            "complete_raw_non_trigger_contacts": sample["complete_raw_non_trigger_contacts"],
            "world": sample["world"],
            "entities": [
                {
                    "entity_id": entity["entity_id"],
                    "scenario_object_id": entity["scenario_object_id"],
                    "body_present": entity["body_present"],
                    "body": entity["body"],
                    "lifecycle": entity["lifecycle"],
                }
                for entity in sample["entities"]
            ],
            "colliders": sample["colliders"],
        }
    return result


def _compare_engine_state_trace(
    original: Mapping[str, Any],
    replay: Mapping[str, Any],
    maximum_delta: int,
) -> dict[str, Any]:
    original_steps = _state_samples_by_relative_step(original)
    replay_steps = _state_samples_by_relative_step(replay)
    shared = sorted(set(original_steps) & set(replay_steps))
    mismatched = [
        step for step in shared if original_steps[step] != replay_steps[step]
    ]
    original_only = sorted(set(original_steps) - set(replay_steps))
    replay_only = sorted(set(replay_steps) - set(original_steps))
    common_end = max(shared) if shared else None
    terminal_only = (
        common_end is not None
        and all(step > common_end for step in original_only + replay_only)
        and abs(max(original_steps) - max(replay_steps)) <= maximum_delta
    )
    if mismatched or (original_only or replay_only) and not terminal_only:
        return _component(
            "engine_state_trace",
            "mismatch",
            "world, body, lifecycle, and collider state must be exact at every shared launch-relative fixed step; only a bounded terminal tail may differ",
            {
                "mismatched_relative_fixed_steps": mismatched[:20],
                "original_only_relative_fixed_steps": original_only,
                "replay_only_relative_fixed_steps": replay_only,
            },
        )
    return _component(
        "engine_state_trace",
        "tolerated" if original_only or replay_only else "equality",
        "world, body, lifecycle, and collider state are exact at every shared launch-relative fixed step",
        {
            "shared_fixed_step_count": len(shared),
            "original_terminal_tail": original_only,
            "replay_terminal_tail": replay_only,
        },
    )


def _artifact_identity_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    origin = _first_launch_step(record)
    frames = []
    for frame in record["frame_records"]:
        expected = f"state:{frame['fixed_step']}"
        if frame["state_id"] != expected:
            raise CohortV2ReplayError("frame-record state identity is not fixed-step derived")
        frames.append((frame["fixed_step"] - origin, frame["forced_terminal"]))

    event_suffixes: dict[tuple[str, tuple[str, ...]], list[int]] = defaultdict(list)
    for event in record["events"]:
        parts = event["event_id"].split(":")
        if (
            len(parts) != 4
            or parts[0] != "event"
            or parts[1] != str(event["fixed_step"])
            or parts[2] != event["event_type"]
            or not parts[3].isdigit()
        ):
            raise CohortV2ReplayError("event identity is not fixed-step and semantic derived")
        key = (event["event_type"], tuple(sorted(event["participants"])))
        event_suffixes[key].append(int(parts[3]))

    contact_suffixes: dict[tuple[str, ...], list[int]] = defaultdict(list)
    relational_context = {}
    for sample in record["fixed_step_samples"]:
        contact_semantics = {}
        for index, contact in enumerate(sample["contacts"]):
            expected = f"contact:{sample['fixed_step']}:{index:04d}"
            if contact["contact_id"] != expected:
                raise CohortV2ReplayError(
                    "contact identity is not fixed-step and canonical-order derived"
                )
            key = (
                contact["entity_a_id"],
                contact["entity_b_id"],
                contact["collider_a_id"],
                contact["collider_b_id"],
            )
            contact_semantics[contact["contact_id"]] = key
            contact_suffixes[key].append(index)

        def resolve(contact_ids: Sequence[str]) -> list[tuple[str, ...]]:
            try:
                return sorted(contact_semantics[contact_id] for contact_id in contact_ids)
            except KeyError as error:
                raise CohortV2ReplayError(
                    "entity or support context references an unknown contact identity"
                ) from error

        relational_context[sample["fixed_step"] - origin] = {
            "entities": [
                {
                    "entity_id": entity["entity_id"],
                    "contacts": resolve(entity["contact_ids"]),
                    "supported_by_entity_ids": entity["supported_by_entity_ids"],
                    "supports_entity_ids": entity["supports_entity_ids"],
                }
                for entity in sample["entities"]
            ],
            "supports": [
                {
                    "supporter_entity_id": support["supporter_entity_id"],
                    "supported_entity_id": support["supported_entity_id"],
                    "contacts": resolve(support["contact_ids"]),
                }
                for support in sample["supports"]
            ],
        }
    return {
        "frames": frames,
        "event_suffixes": dict(event_suffixes),
        "contact_suffixes": dict(contact_suffixes),
        "relational_context": relational_context,
    }


def _common_prefixes_equal(
    original: Mapping[Any, Sequence[Any]], replay: Mapping[Any, Sequence[Any]]
) -> bool:
    for key in set(original) | set(replay):
        common = min(len(original.get(key, ())), len(replay.get(key, ())))
        if list(original.get(key, ()))[:common] != list(replay.get(key, ()))[:common]:
            return False
    return True


def _compare_artifact_identities(
    original: Mapping[str, Any],
    replay: Mapping[str, Any],
    maximum_delta: int,
) -> dict[str, Any]:
    try:
        original_projection = _artifact_identity_projection(original)
        replay_projection = _artifact_identity_projection(replay)
    except CohortV2ReplayError as error:
        return _component(
            "deterministic_artifact_identities",
            "mismatch",
            "frame-record, event, contact, entity-context, and support-context identities must be derivable and aligned",
            {"reason": str(error)},
        )
    original_context = original_projection["relational_context"]
    replay_context = replay_projection["relational_context"]
    shared = set(original_context) & set(replay_context)
    context_mismatches = sorted(
        step for step in shared if original_context[step] != replay_context[step]
    )
    frames_original = original_projection["frames"]
    frames_replay = replay_projection["frames"]
    shared_frame_count = min(len(frames_original), len(frames_replay))
    frame_prefix_matches = (
        frames_original[:shared_frame_count] == frames_replay[:shared_frame_count]
    )
    terminal_frame_delta = abs(len(frames_original) - len(frames_replay))
    identities_match = (
        not context_mismatches
        and frame_prefix_matches
        and terminal_frame_delta <= maximum_delta
        and _common_prefixes_equal(
            original_projection["event_suffixes"],
            replay_projection["event_suffixes"],
        )
        and _common_prefixes_equal(
            original_projection["contact_suffixes"],
            replay_projection["contact_suffixes"],
        )
    )
    return _component(
        "deterministic_artifact_identities",
        "equality" if identities_match and terminal_frame_delta == 0 else (
            "tolerated" if identities_match else "mismatch"
        ),
        "frame-record, event, contact, entity-context, and support-context identities must be derivable and aligned under the declared timing/count bounds",
        {
            "context_mismatched_relative_fixed_steps": context_mismatches[:20],
            "original_frame_record_count": len(frames_original),
            "replay_frame_record_count": len(frames_replay),
        },
    )


def _event_payload_occurrences(record: Mapping[str, Any]) -> dict[tuple[str, tuple[str, ...]], list[Any]]:
    result: dict[tuple[str, tuple[str, ...]], list[Any]] = defaultdict(list)
    for event in record["events"]:
        key = (event["event_type"], tuple(sorted(event["participants"])))
        result[key].append(event["payload"])
    return dict(result)


def _compare_event_payloads(
    original: Mapping[Any, Sequence[Any]],
    replay: Mapping[Any, Sequence[Any]],
) -> dict[str, Any]:
    keys = set(original) | set(replay)
    mismatched = [
        str(key)
        for key in sorted(keys, key=str)
        if list(original.get(key, ()))[: min(len(original.get(key, ())), len(replay.get(key, ())))]
        != list(replay.get(key, ()))[: min(len(original.get(key, ())), len(replay.get(key, ())))]
    ]
    if mismatched:
        return _component(
            "event_payload_semantics",
            "mismatch",
            "payloads for every compared same-semantic event occurrence must be exact",
            {"mismatched_event_semantics": mismatched},
        )
    count_differences = {
        str(key): [len(original.get(key, ())), len(replay.get(key, ()))]
        for key in sorted(keys, key=str)
        if len(original.get(key, ())) != len(replay.get(key, ()))
    }
    return _component(
        "event_payload_semantics",
        "tolerated" if count_differences else "equality",
        "event payloads are exact; repeated same-semantic occurrence counts are reported tolerances",
        {"repeated_occurrence_count_differences": count_differences},
    )


def _contact_geometry_occurrences(record: Mapping[str, Any]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    origin = _first_launch_step(record)
    result: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for sample in record["fixed_step_samples"]:
        for contact in sample["contacts"]:
            key = (
                contact["entity_a_id"],
                contact["entity_b_id"],
                contact["collider_a_id"],
                contact["collider_b_id"],
            )
            result[key].append({
                "relative_fixed_step": sample["fixed_step"] - origin,
                "point": contact["point"],
                "normal_a_to_b": contact["normal_a_to_b"],
                "separation": contact["separation"],
            })
    return dict(result)


def _compare_contact_geometry(
    original: Mapping[Any, Sequence[Mapping[str, Any]]],
    replay: Mapping[Any, Sequence[Mapping[str, Any]]],
    maximum_step_delta: int,
    maximum_separation_delta: float,
) -> dict[str, Any]:
    mismatched = []
    maximum_observed_separation_delta = 0.0
    for key in sorted(set(original) | set(replay), key=str):
        original_values = original.get(key, ())
        replay_values = replay.get(key, ())
        for left, right in zip(original_values, replay_values):
            separation_delta = abs(left["separation"] - right["separation"])
            maximum_observed_separation_delta = max(
                maximum_observed_separation_delta, separation_delta
            )
            if (
                abs(left["relative_fixed_step"] - right["relative_fixed_step"])
                > maximum_step_delta
                or left["point"] != right["point"]
                or left["normal_a_to_b"] != right["normal_a_to_b"]
                or separation_delta > maximum_separation_delta
            ):
                mismatched.append(str(key))
                break
    if mismatched:
        return _component(
            "contact_geometry",
            "mismatch",
            "contact collider identities, point, normal, and bounded separation must match for every compared occurrence",
            {
                "mismatched_contact_semantics": mismatched,
                "maximum_observed_separation_delta": maximum_observed_separation_delta,
            },
        )
    count_differences = {
        str(key): [len(original.get(key, ())), len(replay.get(key, ()))]
        for key in sorted(set(original) | set(replay), key=str)
        if len(original.get(key, ())) != len(replay.get(key, ()))
    }
    tolerated = bool(count_differences or maximum_observed_separation_delta)
    return _component(
        "contact_geometry",
        "tolerated" if tolerated else "equality",
        "contact point and normal are exact; relative timing, separation, and repeated counts use the declared bounds",
        {
            "maximum_observed_separation_delta": maximum_observed_separation_delta,
            "repeated_occurrence_count_differences": count_differences,
        },
    )


def _final_lifecycle_projection(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "entity_id": entity["entity_id"],
            "scenario_object_id": entity["scenario_object_id"],
            "body_present": entity["body_present"],
            "lifecycle": entity["lifecycle"],
        }
        for entity in record["fixed_step_samples"][-1]["entities"]
    ]


def _coverage_witness(record: Mapping[str, Any], stratum: str) -> bool:
    if stratum == "contact":
        return any(sample["contacts"] for sample in record["fixed_step_samples"])
    if stratum == "collision":
        contacts = {
            (sample["fixed_step"], frozenset((contact["entity_a_id"], contact["entity_b_id"])))
            for sample in record["fixed_step_samples"]
            for contact in sample["contacts"]
        }
        return any(
            event["event_type"] == "collision"
            and (event["fixed_step"], frozenset(event["participants"])) in contacts
            for event in record["events"]
        )
    if stratum == "support":
        supports = [
            {(item["supporter_entity_id"], item["supported_entity_id"]) for item in sample["supports"]}
            for sample in record["fixed_step_samples"]
        ]
        return any(previous & current for previous, current in zip(supports, supports[1:]))
    if stratum == "stable":
        return record["terminal_evidence"]["reason"] == "stable_entered"
    raise CohortV2ReplayError(f"unsupported replay coverage stratum: {stratum}")


def compare_replay_scenario_collection(base: Path, plan: Mapping[str, Any], scenario_collection: Mapping[str, Any]) -> dict[str, Any]:
    """Compare one original/replay pair and report every component verdict."""
    components: list[dict[str, Any]] = []
    try:
        original_attempt, original, original_observation = _load_attempt(base, plan, scenario_collection, "original")
        replay_attempt, replay, replay_observation = _load_attempt(base, plan, scenario_collection, "replay")
    except CohortV2ReplayError as error:
        components.append(_component("artifact_availability_and_binding", "unavailable", "both complete bound attempts are required", {"reason": str(error)}))
        return {
            "schema": "cohort_v2_replay_scenario_collection_verdict_v1",
            "scenario_collection_identity": scenario_collection["identity"],
            "scenario_collection_id": scenario_collection["scenario_collection_id"],
            "original_attempt_identity": _attempt_identity(plan, scenario_collection, "original"),
            "replay_attempt_identity": _attempt_identity(plan, scenario_collection, "replay"),
            "same_scenario_lineage": True,
            "same_exposure_role": True,
            "components": components,
            "passed": False,
        }

    if original_attempt["identity"] == replay_attempt["identity"] or original.capture_id == replay.capture_id or original.shot_id == replay.shot_id:
        components.append(_component("attempt_identity", "mismatch", "original and replay attempts, captures, and shots must be distinct", {}))
    else:
        components.append(_component("attempt_identity", "equality", "attempt roles are distinct while source lineage and exposure role are shared", {
            "original_attempt_identity": original_attempt["identity"],
            "replay_attempt_identity": replay_attempt["identity"],
        }))
    components.append(_component("version_envelope", "equality", "every declared version field is exact", {"identity": plan["version_envelope"]["identity"]}))
    components.append(_component("source_manifests_and_scenario_content", "equality", "plan, partition, manifest, specification, content, template, level, and lineage identities are exact", {
        "collection_plan_identity": plan["identity"],
        "partition_manifest_identity": plan["partition_manifest_identity"],
        "scenario_manifest_identity": scenario_collection["scenario_manifest_identity"],
        "scenario_content_identity": scenario_collection["scenario_content_identity"],
    }))
    try:
        frozen_command = _load_frozen_replay_command(
            Path(base), plan, scenario_collection, original_attempt
        )
    except CohortV2ReplayError as error:
        components.append(_component(
            "intervention",
            "unavailable",
            "the original attempt must freeze one immutable exact socket command",
            {"reason": str(error)},
        ))
    else:
        action_exact = (
            original_attempt["interface_action"] == replay_attempt["interface_action"]
            == frozen_command["interface_action"]
            and original_attempt["engine_relative_action"]
            == replay_attempt["engine_relative_action"]
            == scenario_collection["intervention"]["engine_relative_action"]
        )
        components.append(_component(
            "intervention",
            "equality" if action_exact else "mismatch",
            "the #44-style original action and exact frozen socket command must be replayed without remapping",
            {
                "frozen_replay_command_identity": frozen_command["identity"],
                "socket_command": frozen_command["interface_action"]["socket_command"],
            },
        ))

    original_record = original.record
    replay_record = replay.record
    original_initial = normalized_initial_engine_state_identity(original)
    replay_initial = normalized_initial_engine_state_identity(replay)
    components.append(_component(
        "initial_engine_state",
        "equality" if original_initial == replay_initial else "mismatch",
        "normalized initial engine state must be exact",
        {"original_identity": original_initial, "replay_identity": replay_initial},
    ))
    catalogs_equal = (
        original_record["coordinate_convention"] == replay_record["coordinate_convention"]
        and original_record["causal_entities"] == replay_record["causal_entities"]
        and original_record["colliders"] == replay_record["colliders"]
    )
    components.append(_component(
        "entity_and_collider_identities",
        "equality" if catalogs_equal else "mismatch",
        "coordinate, causal-entity, and collider catalogs must be exact",
        {
            "causal_entity_count": len(original_record["causal_entities"]),
            "collider_count": len(original_record["colliders"]),
        },
    ))
    stride_equal = original.configured_fixed_step_capture_stride == replay.configured_fixed_step_capture_stride
    components.append(_component(
        "fixed_step_capture_contract",
        "equality" if stride_equal else "mismatch",
        "configured authoritative fixed-step stride must be exact",
        {"original_stride": original.configured_fixed_step_capture_stride, "replay_stride": replay.configured_fixed_step_capture_stride},
    ))
    rules = plan["comparison_rules"]
    max_delta = rules["maximum_relative_fixed_step_delta"]
    components.append(_compare_engine_state_trace(
        original_record, replay_record, max_delta
    ))
    components.append(_compare_artifact_identities(
        original_record, replay_record, max_delta
    ))
    components.append(_compare_occurrences("event_semantics", _event_occurrences(original_record), _event_occurrences(replay_record), max_delta))
    components.append(_compare_event_payloads(
        _event_payload_occurrences(original_record),
        _event_payload_occurrences(replay_record),
    ))
    components.append(_compare_occurrences("contact_semantics", _relation_occurrences(original_record, "contact"), _relation_occurrences(replay_record, "contact"), max_delta))
    components.append(_compare_contact_geometry(
        _contact_geometry_occurrences(original_record),
        _contact_geometry_occurrences(replay_record),
        max_delta,
        rules["maximum_contact_separation_delta"],
    ))
    components.append(_compare_occurrences("support_semantics", _relation_occurrences(original_record, "support"), _relation_occurrences(replay_record, "support"), max_delta))
    lifecycle_original = _final_lifecycle_projection(original_record)
    lifecycle_replay = _final_lifecycle_projection(replay_record)
    components.append(_component(
        "terminal_entity_lifecycle",
        "equality" if lifecycle_original == lifecycle_replay else "mismatch",
        "terminal entity presence and lifecycle semantics must be exact",
        {"terminal_entity_count": len(lifecycle_original)},
    ))
    original_origin = _first_launch_step(original_record)
    replay_origin = _first_launch_step(replay_record)
    original_terminal = original_record["terminal_evidence"]
    replay_terminal = replay_record["terminal_evidence"]
    terminal_delta = abs(
        (original_terminal["fixed_step"] - original_origin)
        - (replay_terminal["fixed_step"] - replay_origin)
    )
    if original_terminal["reason"] != replay_terminal["reason"] or terminal_delta > max_delta:
        terminal_status = "mismatch"
    elif terminal_delta:
        terminal_status = "tolerated"
    else:
        terminal_status = "equality"
    components.append(_component(
        "termination_semantics",
        terminal_status,
        "termination reason is exact and launch-relative terminal timing may differ by one fixed step",
        {
            "reason": original_terminal["reason"],
            "original_launch_relative_fixed_step": original_terminal["fixed_step"] - original_origin,
            "replay_launch_relative_fixed_step": replay_terminal["fixed_step"] - replay_origin,
            "fixed_step_delta": terminal_delta,
        },
    ))
    original_minimum = original_record["minimum_contact_separation"]
    replay_minimum = replay_record["minimum_contact_separation"]
    if original_minimum["observed"] != replay_minimum["observed"]:
        separation_status = "mismatch"
        separation_delta = None
    elif not original_minimum["observed"]:
        separation_status = "equality"
        separation_delta = None
    else:
        separation_delta = abs(original_minimum["separation"] - replay_minimum["separation"])
        if separation_delta == 0:
            separation_status = "equality"
        elif separation_delta <= rules["maximum_contact_separation_delta"]:
            separation_status = "tolerated"
        else:
            separation_status = "mismatch"
    components.append(_component(
        "minimum_contact_separation",
        separation_status,
        "observability is exact and Unity-2D separation may differ by at most 0.001 world units",
        {"absolute_delta": separation_delta},
    ))
    observation_original = {
        key: original_observation[key]
        for key in ("schema", "exposure_role", "observation_configuration", "access_policy")
    }
    observation_replay = {
        key: replay_observation[key]
        for key in ("schema", "exposure_role", "observation_configuration", "access_policy")
    }
    components.append(_component(
        "observation_synchronization_and_access",
        "equality" if observation_original == observation_replay else "mismatch",
        "each trace independently validates camera, viewport, transform, synchronization, and agent/canonical access; configuration and access policy are exact across attempts",
        {
            "observation_configuration_identity": scenario_collection["observation_configuration_identity"],
            "frame_record_count": len(original_observation["frame_records"]),
            "agent_access": original_observation["access_policy"]["agent"],
            "canonical_access": original_observation["access_policy"]["canonical"],
        },
    ))
    components.append(_component(
        "cross_attempt_camera_viewport_and_transform_equality",
        "not_required",
        "camera, viewport, and world-to-observation transform are complete and validated per attempt but remain render provenance",
        {
            "original_frame_record_count": len(original_observation["frame_records"]),
            "replay_frame_record_count": len(replay_observation["frame_records"]),
        },
    ))
    components.append(_component(
        "cross_attempt_pixel_equality",
        "not_required",
        "pixel equality was not prospectively declared by the observation comparison contract",
        {"pixel_equality_required": False},
    ))
    coverage = {
        stratum: {
            "original": _coverage_witness(original_record, stratum),
            "replay": _coverage_witness(replay_record, stratum),
        }
        for stratum in scenario_collection["coverage_strata"]
    }
    components.append(_component(
        "representative_coverage_strata",
        "equality" if all(value["original"] and value["replay"] for value in coverage.values()) else "mismatch",
        "every prospectively declared stratum must be witnessed in both attempts",
        coverage,
    ))
    passed = all(component["status"] in {"equality", "tolerated", "not_required"} for component in components)
    return {
        "schema": "cohort_v2_replay_scenario_collection_verdict_v1",
        "scenario_collection_identity": scenario_collection["identity"],
        "scenario_collection_id": scenario_collection["scenario_collection_id"],
        "original_attempt_identity": original_attempt["identity"],
        "replay_attempt_identity": replay_attempt["identity"],
        "same_scenario_lineage": True,
        "same_exposure_role": True,
        "components": components,
        "passed": passed,
    }


def build_replay_report(base: Path) -> dict[str, Any]:
    plan = validate_replay_plan(base)
    verdicts = [compare_replay_scenario_collection(Path(base), plan, scenario_collection) for scenario_collection in plan["scenario_collections"]]
    coverage = {
        "scenario_collection_count": len(verdicts),
        "non_final_scenario_lineage_count": len({scenario_collection["scenario_lineage_identity"] for scenario_collection in plan["scenario_collections"]}),
        "level_instance_count": len({scenario_collection["level_instance_identity"] for scenario_collection in plan["scenario_collections"]}),
        "scenario_template_count": len({scenario_collection["scenario_template_identity"] for scenario_collection in plan["scenario_collections"]}),
        "intervention_count": len({scenario_collection["intervention"]["identity"] for scenario_collection in plan["scenario_collections"]}),
        "strata": sorted({stratum for scenario_collection in plan["scenario_collections"] for stratum in scenario_collection["coverage_strata"]}),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "identity": "",
        "plan_identity": plan["identity"],
        "version_envelope_identity": plan["version_envelope"]["identity"],
        "partition_manifest_identity": plan["partition_manifest_identity"],
        "comparison_rules_identity": plan["comparison_rules"]["identity"],
        "benchmark_agent_action_provenance": "not_required",
        "retry_count": 0,
        "unavailable_components": [
            {"scenario_collection_id": verdict["scenario_collection_id"], "component": component["component"]}
            for verdict in verdicts
            for component in verdict["components"]
            if component["status"] == "unavailable"
        ],
        "coverage": coverage,
        "scenario_collection_verdicts": verdicts,
        "passed": all(verdict["passed"] for verdict in verdicts),
    }
    report["identity"] = semantic_identity(
        "cohort-v2-replay-evidence-v1",
        report["plan_identity"],
        report["version_envelope_identity"],
        *(f"{verdict['scenario_collection_id']}={verdict['original_attempt_identity']}->{verdict['replay_attempt_identity']}:{verdict['passed']}" for verdict in verdicts),
    )
    return report


def build_issue_48_bundle(
    report: Mapping[str, Any],
    plan: Mapping[str, Any],
    artifacts: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": BUNDLE_SCHEMA,
        "identity": semantic_identity(
            "issue-48-cohort-v2-replay-evidence-bundle-v1",
            report["identity"],
            *artifacts,
        ),
        "issue": 48,
        "replay_evidence_identity": report["identity"],
        "plan_identity": plan["identity"],
        "version_envelope_identity": plan["version_envelope"]["identity"],
        "partition_manifest_identity": plan["partition_manifest_identity"],
        "observation_capability_bundle_identity": plan[
            "observation_capability_bundle_identity"
        ],
        "artifacts": list(artifacts),
        "passed": report["passed"],
        "limitations": [
            "Cross-attempt pixel equality is not claimed; each trace independently validates its declared exact canonical-to-agent transform.",
            "The comparison rules apply only to the exact version envelope recorded by this bundle.",
            "Benchmark-agent action provenance is optional and was not used.",
        ],
    }


def validate_issue_48_evidence(root: Path) -> dict[str, Any]:
    """Recompute the immutable issue-48 report and exact bundle membership."""
    base = Path(root)
    report = _load_json(base / REPORT_NAME, "cohort-v2 replay evidence")
    expected_report = build_replay_report(base)
    if report != expected_report:
        raise CohortV2ReplayError("cohort-v2 replay evidence is stale")
    bundle = _load_json(base / BUNDLE_NAME, "issue-48 replay evidence bundle")
    _require_exact_fields(
        bundle,
        {
            "schema", "identity", "issue", "replay_evidence_identity", "plan_identity",
            "version_envelope_identity", "partition_manifest_identity",
            "observation_capability_bundle_identity", "artifacts", "passed", "limitations",
        },
        "issue-48 replay evidence bundle",
    )
    plan = validate_replay_plan(base)
    artifacts = sorted(
        path.relative_to(base).as_posix()
        for path in base.rglob("*")
        if path.is_file() and path.name != BUNDLE_NAME
    )
    expected_bundle = build_issue_48_bundle(report, plan, artifacts)
    if bundle != expected_bundle:
        raise CohortV2ReplayError("issue-48 replay bundle membership or identity is stale")
    return bundle
