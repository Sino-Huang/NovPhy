"""Frozen plans and public reader for issue #62's multi-shot cohort."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Callable, Final, Mapping

from scripts.cohort_v2_macro_semantics import validate_capture_macro_derivation
from scripts.cohort_v2_micro_relations import (
    validate_capture_micro_relation_derivation,
)
from scripts.cohort_v2_physical_violations import (
    validate_capture_physical_violation_derivation,
)
from scripts.observation_trace import (
    load_observation_bytes,
    validate_observation_exposure_boundaries,
    validate_observation_trace,
)
from scripts.physics_capture_v2 import load_physics_capture_v2
from scripts.cohort_v2_scenarios import load_cohort_v2_scenario_manifest
from world_model.data.cohort_v2 import build_cohort_v2_central_frame_records
from world_model.data.deployment_temporal import (
    AgentObservation,
    DecisionTargets,
    DecisionTransition,
    DeploymentTemporalError,
    DeploymentTrajectory,
    DeploymentTrajectoryReader,
    ExecutedAction,
    TrajectoryLineageBinding,
    TrajectoryLineageManifest,
)


PLAN_SCHEMA: Final = "multi_shot_successor_collection_plan_v4"
PLAN_IDENTITY_NAMESPACE: Final = "issue-62-collection-plan-v4"
PLANNED_ACTION_IDENTITY_NAMESPACE: Final = "issue-62-planned-action-v3"
PILOT_REPORT_SCHEMA: Final = "multi_shot_successor_pilot_report_v3"
PILOT_REPORT_IDENTITY_NAMESPACE: Final = "issue-62-pilot-report-v3"
RELEASE_SCHEMA: Final = "multi_shot_successor_cohort_release_v3"
TRAJECTORY_SCHEMA: Final = "multi_shot_successor_trajectory_v3"
PUBLIC_ROLES: Final = ("training", "calibration", "model_selection")
BEHAVIOR_POLICIES: Final = (
    "uniform_random",
    "stratified_bounds",
    "trajectory_guided_direct_pig",
)
GUIDED_ACTION_STRATUM: Final = "direct_pig__lowest_clear_full_pull__tap_early"
ACTION_BOUNDS: Final = {
    "drag_x": [-160, -10],
    "drag_y": [-80, 80],
    "tap_time_ms": [0, 1000],
    "release_time_ms": 600,
}
GENERATOR_FAMILIES: Final = (
    {
        "name": "type010101",
        "template_source": (
            "tasks/task_templates/novelty_level_0/type010101/Levels/"
            "00001_0_1_010101_0_1.xml"
        ),
        "workbook_row": 3,
        "canonical_template_name": "0_1_010101_0_1",
        "reference_point": [1.00798, -2.1274],
        "min_coordinate": [-7.88, -2.39049],
        "max_coordinate": [1.229969, 1.809741],
        "authored_bird_count": 1,
    },
    {
        "name": "type010102",
        "template_source": (
            "tasks/task_templates/novelty_level_0/type010102/Levels/"
            "00001_0_1_010102_0_2.xml"
        ),
        "workbook_row": 4,
        "canonical_template_name": "0_1_010102_0_2",
        "reference_point": [1.02408, -1.84657],
        "min_coordinate": [-7.235919, -1.95804],
        "max_coordinate": [1.444081, 1.53147],
        "authored_bird_count": 3,
    },
)


class SuccessorCohortError(ValueError):
    """The issue-62 plan, trajectory, or release is inconsistent."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def successor_identity(namespace: str, *fields: Any) -> str:
    content = _canonical(list(fields)).encode("utf-8")
    return f"{namespace}:sha256:{hashlib.sha256(content).hexdigest()}"


def _with_identity(namespace: str, field: str, value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload[field] = successor_identity(namespace, value)
    return payload


def _expected_identity(namespace: str, field: str, value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != field}
    return successor_identity(namespace, payload)


def release_identity_for_plan(plan_identity: str) -> str:
    return successor_identity("issue-62-successor-release-v3", plan_identity)


def _action_stratum(drag_x: int, drag_y: int, tap_time_ms: int) -> str:
    x_name = "far" if drag_x < -100 else "near"
    if drag_y < -27:
        y_name = "low"
    elif drag_y > 27:
        y_name = "high"
    else:
        y_name = "middle"
    tap_name = "early" if tap_time_ms < 500 else "late"
    return f"x_{x_name}__y_{y_name}__tap_{tap_name}"


def _sample_action(
    rng: random.Random,
    *,
    policy: str,
    policy_action_index: int,
    action_index: int,
) -> dict[str, Any]:
    x_bounds = tuple(ACTION_BOUNDS["drag_x"])
    y_bounds = tuple(ACTION_BOUNDS["drag_y"])
    tap_bounds = tuple(ACTION_BOUNDS["tap_time_ms"])
    if policy == "uniform_random":
        drag_x = rng.randint(*x_bounds)
        drag_y = rng.randint(*y_bounds)
        tap = rng.randint(*tap_bounds)
    elif policy == "stratified_bounds":
        stratum_index = policy_action_index % 12
        x_bin = stratum_index % 2
        y_bin = (stratum_index // 2) % 3
        tap_bin = (stratum_index // 6) % 2
        x_ranges = ((-160, -101), (-100, -10))
        y_ranges = ((-80, -28), (-27, 27), (28, 80))
        tap_ranges = ((0, 499), (500, 1000))
        drag_x = rng.randint(*x_ranges[x_bin])
        drag_y = rng.randint(*y_ranges[y_bin])
        tap = rng.randint(*tap_ranges[tap_bin])
    elif policy == "trajectory_guided_direct_pig":
        return _with_identity(
            PLANNED_ACTION_IDENTITY_NAMESPACE,
            "identity",
            {
                "action_index": action_index,
                "selection_mode": "trajectory_guided_direct_pig_clearance",
                "target_kind": "pig",
                "target_rank": 0,
                "aim_point": "visible_polygon_upper_edge",
                "trajectory_arc": "lowest_clear_full_pull",
                "minimum_pull": "trajectory_drag_radius",
                "clearance_model": "near_target_margin_inflated_obstacles",
                "bird_radius_world": 0.17,
                "clearance_margin_world": 0.34,
                "clearance_margin_minimum_target_distance_world": 8.0,
                "tap_time_ms": 0,
                "release_time_ms": ACTION_BOUNDS["release_time_ms"],
                "action_stratum": GUIDED_ACTION_STRATUM,
            },
        )
    else:
        raise SuccessorCohortError("behavior policy is unsupported")
    value = {
        "action_index": action_index,
        "selection_mode": "frozen_relative_drag",
        "drag_x": drag_x,
        "drag_y": drag_y,
        "tap_time_ms": tap,
        "release_time_ms": ACTION_BOUNDS["release_time_ms"],
        "action_stratum": _action_stratum(drag_x, drag_y, tap),
    }
    return _with_identity(PLANNED_ACTION_IDENTITY_NAMESPACE, "identity", value)


def _planned_action_count(
    family: Mapping[str, Any], policy: str, max_shots: int
) -> int:
    count = min(max_shots, int(family["authored_bird_count"]))
    return min(count, 2) if policy == "trajectory_guided_direct_pig" else count


def _lineage_slots(
    phase: str,
    role_counts: Mapping[str, int],
    *,
    max_shots: int,
) -> list[dict[str, Any]]:
    phase_offset = 6_300_000 if phase == "pilot" else 63_000_000
    slots = []
    global_ordinal = 0
    policy_action_counts = Counter()
    for role_index, role in enumerate(PUBLIC_ROLES):
        for role_ordinal in range(role_counts[role]):
            family = GENERATOR_FAMILIES[(role_ordinal + role_index) % 2]
            policy = BEHAVIOR_POLICIES[
                (role_ordinal + role_index) % len(BEHAVIOR_POLICIES)
            ]
            generation_seed = phase_offset + role_index * 1_000_000 + role_ordinal
            slot_payload = {
                "phase": phase,
                "exposure_role": role,
                "ordinal": global_ordinal,
                "role_ordinal": role_ordinal,
                "generator_family": family["name"],
                "generation_seed": generation_seed,
                "behavior_policy": policy,
            }
            slot_identity = successor_identity(
                "issue-62-scenario-lineage-slot-v3", slot_payload
            )
            rng = random.Random(f"{slot_identity}:actions")
            action_count = _planned_action_count(family, policy, max_shots)
            actions = [
                _sample_action(
                    rng,
                    policy=policy,
                    policy_action_index=policy_action_counts[policy] + index,
                    action_index=index,
                )
                for index in range(action_count)
            ]
            policy_action_counts[policy] += action_count
            slots.append({
                **slot_payload,
                "slot_identity": slot_identity,
                "planned_actions": actions,
            })
            global_ordinal += 1
    return slots


def build_pilot_plan() -> dict[str, Any]:
    role_counts = {role: 12 for role in PUBLIC_ROLES}
    payload = {
        "schema": PLAN_SCHEMA,
        "phase": "pilot",
        "source_issue": 62,
        "non_final_only": True,
        "role_counts": role_counts,
        "generator_families": [dict(item) for item in GENERATOR_FAMILIES],
        "behavior_policy_mixture": {
            policy: "balanced_by_frozen_slot_ordinal" for policy in BEHAVIOR_POLICIES
        },
        "action_bounds": dict(ACTION_BOUNDS),
        "max_shots": 6,
        "fixed_retry_limit": 2,
        "lineages": _lineage_slots("pilot", role_counts, max_shots=6),
        "nested_training_scales": [],
        "pilot_report_identity": None,
        "resource_decision": {
            "bounded_pilot_lineages": sum(role_counts.values()),
            "production_membership_frozen_after_pilot": True,
        },
    }
    return validate_successor_plan(
        _with_identity(PLAN_IDENTITY_NAMESPACE, "identity", payload)
    )


def validate_pilot_report(
    report: Mapping[str, Any],
    *,
    pilot_plan: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(report)
    if (
        value.get("schema") != PILOT_REPORT_SCHEMA
        or value.get("pilot_plan_identity") != pilot_plan.get("identity")
        or value.get("final_evaluation_opened") is not False
        or value.get("planned_lineage_count") != len(pilot_plan.get("lineages", ()))
        or value.get("completed_lineage_count") != len(pilot_plan.get("lineages", ()))
        or value.get("accepted_lineage_count", 0) < 1
        or value.get("passed") is not True
        or value.get("identity")
        != _expected_identity(PILOT_REPORT_IDENTITY_NAMESPACE, "identity", value)
    ):
        raise SuccessorCohortError("pilot report is incomplete or stale")
    return value


def build_production_plan(
    pilot_report: Mapping[str, Any],
    *,
    pilot_plan: Mapping[str, Any],
    maximum_training_lineages: int,
) -> dict[str, Any]:
    report = validate_pilot_report(pilot_report, pilot_plan=pilot_plan)
    if (
        type(maximum_training_lineages) is not int
        or not 200 <= maximum_training_lineages <= 10_000
    ):
        raise SuccessorCohortError(
            "production training-lineage cap must be between 200 and 10000"
        )
    role_counts = {
        "training": maximum_training_lineages,
        "calibration": 200,
        "model_selection": 200,
    }
    slots = _lineage_slots("production", role_counts, max_shots=6)
    training = [
        item["slot_identity"]
        for item in slots
        if item["exposure_role"] == "training"
    ]
    scale_counts = [
        count for count in (6, 200, 1_000, 5_000, 10_000)
        if count <= maximum_training_lineages
    ]
    if scale_counts[-1] != maximum_training_lineages:
        scale_counts.append(maximum_training_lineages)
    scales = [
        {
            "name": f"training_{count}",
            "lineage_count": count,
            "slot_identities": training[:count],
        }
        for count in scale_counts
    ]
    typical_seconds = float(report["runtime_cost"]["median_seconds_per_lineage"])
    typical_bytes = int(report["runtime_cost"]["median_bytes_per_lineage"])
    scheduled = sum(role_counts.values())
    payload = {
        "schema": PLAN_SCHEMA,
        "phase": "production",
        "source_issue": 62,
        "non_final_only": True,
        "role_counts": role_counts,
        "generator_families": [dict(item) for item in GENERATOR_FAMILIES],
        "behavior_policy_mixture": {
            policy: "balanced_by_frozen_slot_ordinal" for policy in BEHAVIOR_POLICIES
        },
        "action_bounds": dict(ACTION_BOUNDS),
        "max_shots": 6,
        "fixed_retry_limit": 2,
        "lineages": slots,
        "nested_training_scales": scales,
        "pilot_report_identity": report["identity"],
        "resource_decision": {
            "maximum_training_lineages": maximum_training_lineages,
            "literature_ladder": [6, 200, 1_000, 5_000, 10_000],
            "resource_limited_below_10000": maximum_training_lineages < 10_000,
            "estimated_collection_hours": typical_seconds * scheduled / 3600.0,
            "estimated_collection_bytes": typical_bytes * scheduled,
            "held_out_lineages_per_role": 200,
            "membership_selected_without_production_outcomes": True,
        },
    }
    return validate_successor_plan(
        _with_identity(PLAN_IDENTITY_NAMESPACE, "identity", payload)
    )


def validate_successor_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(plan)
    required = {
        "schema", "identity", "phase", "source_issue", "non_final_only",
        "role_counts", "generator_families", "behavior_policy_mixture",
        "action_bounds", "max_shots", "fixed_retry_limit", "lineages",
        "nested_training_scales", "pilot_report_identity", "resource_decision",
    }
    if set(value) != required or value.get("schema") != PLAN_SCHEMA:
        raise SuccessorCohortError("successor collection plan fields differ")
    if (
        value.get("phase") not in {"pilot", "production"}
        or value.get("source_issue") != 62
        or value.get("non_final_only") is not True
        or value.get("action_bounds") != ACTION_BOUNDS
        or value.get("max_shots") != 6
        or value.get("fixed_retry_limit") != 2
        or value.get("generator_families")
        != [dict(item) for item in GENERATOR_FAMILIES]
        or set(value.get("behavior_policy_mixture", {}))
        != set(BEHAVIOR_POLICIES)
        or set(value.get("role_counts", {})) != set(PUBLIC_ROLES)
        or value.get("identity")
        != _expected_identity(PLAN_IDENTITY_NAMESPACE, "identity", value)
    ):
        raise SuccessorCohortError("successor collection plan contract differs")
    lineages = value.get("lineages")
    if not isinstance(lineages, list) or not lineages:
        raise SuccessorCohortError("successor collection plan has no lineages")
    counts = Counter()
    slots = set()
    seeds = set()
    families = {item["name"]: item for item in GENERATOR_FAMILIES}
    for expected_ordinal, slot in enumerate(lineages):
        if not isinstance(slot, Mapping) or set(slot) != {
            "phase", "exposure_role", "ordinal", "role_ordinal",
            "generator_family", "generation_seed", "behavior_policy",
            "slot_identity", "planned_actions",
        }:
            raise SuccessorCohortError("successor lineage slot fields differ")
        role = slot["exposure_role"]
        if (
            slot["phase"] != value["phase"]
            or role not in PUBLIC_ROLES
            or slot["ordinal"] != expected_ordinal
            or slot["generator_family"] not in families
            or slot["behavior_policy"] not in BEHAVIOR_POLICIES
            or slot["slot_identity"] in slots
            or slot["generation_seed"] in seeds
        ):
            raise SuccessorCohortError("successor lineage slot is invalid or reused")
        actions = slot["planned_actions"]
        expected_action_count = _planned_action_count(
            families[slot["generator_family"]],
            str(slot["behavior_policy"]),
            int(value["max_shots"]),
        )
        if not isinstance(actions, list) or len(actions) != expected_action_count:
            raise SuccessorCohortError("successor lineage action plan is incomplete")
        for action_index, action in enumerate(actions):
            if not isinstance(action, Mapping):
                raise SuccessorCohortError("successor planned action is malformed")
            common_valid = (
                action.get("action_index") == action_index
                and action.get("release_time_ms")
                == ACTION_BOUNDS["release_time_ms"]
                and action.get("identity")
                == _expected_identity(
                    PLANNED_ACTION_IDENTITY_NAMESPACE, "identity", action
                )
            )
            if action.get("selection_mode") == "frozen_relative_drag":
                mode_valid = (
                    set(action) == {
                        "action_index", "selection_mode", "drag_x", "drag_y",
                        "tap_time_ms", "release_time_ms", "action_stratum",
                        "identity",
                    }
                    and action.get("action_stratum")
                    == _action_stratum(
                        action.get("drag_x"), action.get("drag_y"),
                        action.get("tap_time_ms"),
                    )
                )
            else:
                mode_valid = (
                    set(action) == {
                        "action_index", "selection_mode", "target_kind",
                        "target_rank", "aim_point", "trajectory_arc",
                        "clearance_model",
                        "bird_radius_world",
                        "clearance_margin_world",
                        "clearance_margin_minimum_target_distance_world",
                        "minimum_pull",
                        "tap_time_ms", "release_time_ms", "action_stratum",
                        "identity",
                    }
                    and action.get("selection_mode")
                    == "trajectory_guided_direct_pig_clearance"
                    and action.get("target_kind") == "pig"
                    and action.get("target_rank") == 0
                    and action.get("aim_point") == "visible_polygon_upper_edge"
                    and action.get("trajectory_arc") == "lowest_clear_full_pull"
                    and action.get("minimum_pull") == "trajectory_drag_radius"
                    and action.get("clearance_model")
                    == "near_target_margin_inflated_obstacles"
                    and action.get("bird_radius_world") == 0.17
                    and action.get("clearance_margin_world") == 0.34
                    and action.get(
                        "clearance_margin_minimum_target_distance_world"
                    ) == 8.0
                    and action.get("tap_time_ms") == 0
                    and action.get("action_stratum") == GUIDED_ACTION_STRATUM
                )
            if not common_valid or not mode_valid:
                raise SuccessorCohortError("successor planned action is malformed")
        counts[role] += 1
        slots.add(slot["slot_identity"])
        seeds.add(slot["generation_seed"])
    if dict(counts) != value["role_counts"]:
        raise SuccessorCohortError("successor plan role counts differ")
    scales = value["nested_training_scales"]
    if value["phase"] == "pilot":
        if scales or value["pilot_report_identity"] is not None:
            raise SuccessorCohortError("pilot plan cannot contain production membership")
    else:
        if not isinstance(value["pilot_report_identity"], str) or not scales:
            raise SuccessorCohortError("production plan lacks its pilot or scales")
        training = [
            item["slot_identity"] for item in lineages
            if item["exposure_role"] == "training"
        ]
        previous = 0
        for scale in scales:
            count = scale.get("lineage_count")
            if (
                type(count) is not int
                or count <= previous
                or scale.get("slot_identities") != training[:count]
            ):
                raise SuccessorCohortError(
                    "production training scales are not exact nested lineages"
                )
            previous = count
        if scales[0]["lineage_count"] != 6 or previous != len(training):
            raise SuccessorCohortError(
                "production training scales omit the six-lineage or full subset"
            )
    return value


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise SuccessorCohortError(f"cannot load {label}: {path}") from error
    if not isinstance(value, dict):
        raise SuccessorCohortError(f"{label} is not an object")
    return value


def _load_shot(
    trajectory_root: Path,
    shot: Mapping[str, Any],
    *,
    release_identity: str,
    role: str,
) -> dict[str, Any]:
    shot_root = trajectory_root / str(shot["path"])
    capture = load_physics_capture_v2(shot_root / "physics_capture_v2.json")
    observation_root = shot_root / "observation-trace"
    observation = validate_observation_trace(observation_root)
    references = {item["kind"]: item for item in shot["derivations"]}
    if set(references) != {"micro", "macro", "physical-violations"}:
        raise SuccessorCohortError("successor shot derivation inventory differs")
    derivations = {
        kind: _load(shot_root / item["path"], f"{kind} derivation")
        for kind, item in references.items()
    }
    source_reference = f"{shot['path']}/physics_capture_v2.json"
    validators = {
        "micro": validate_capture_micro_relation_derivation,
        "macro": validate_capture_macro_derivation,
        "physical-violations": validate_capture_physical_violation_derivation,
    }
    for kind, validator in validators.items():
        if derivations[kind].get("identity") != references[kind]["identity"]:
            raise SuccessorCohortError("successor derivation identity differs")
        validator(
            derivations[kind],
            capture,
            source_reference=source_reference,
            source_capture_bundle_identity=release_identity,
        )
    frames = build_cohort_v2_central_frame_records(capture, derivations)
    observations = {item["fixed_step"]: item for item in observation["frame_records"]}
    if (
        shot.get("capture_id") != capture.capture_id
        or shot.get("observation_manifest_identity") != observation["identity"]
        or observation.get("exposure_role") != role
        or tuple(item.fixed_step for item in frames) != tuple(observations)
    ):
        raise SuccessorCohortError("successor shot alignment or role differs")
    return {
        "root": shot_root,
        "capture": capture,
        "observation_root": observation_root,
        "observation_manifest": observation,
        "observations": observations,
        "frames": {item.fixed_step: item for item in frames},
    }


def _agent_observation(shot: Mapping[str, Any], fixed_step: int, role: str) -> AgentObservation:
    metadata = shot["observations"].get(fixed_step)
    if metadata is None:
        raise SuccessorCohortError("transition observation fixed step is absent")
    return AgentObservation(
        identity=str(metadata["agent_observation"]["identity"]),
        fixed_step=fixed_step,
        fixed_time_seconds=float(metadata["fixed_time_seconds"]),
        png=load_observation_bytes(
            shot["observation_root"],
            frame_record_identity=metadata["identity"],
            observation_role="agent",
            workflow_kind=role,
            purpose="model_input",
        ),
        observation_role="agent",
    )


def load_successor_trajectory(
    trajectory_root: Path,
    *,
    release_identity: str,
) -> DeploymentTrajectory:
    root = Path(trajectory_root)
    record = _load(root / "trajectory.json", "successor trajectory")
    if (
        record.get("schema") != TRAJECTORY_SCHEMA
        or record.get("complete") is not True
        or record.get("exposure_role") not in PUBLIC_ROLES
        or record.get("release_identity") != release_identity
    ):
        raise SuccessorCohortError("successor trajectory contract differs")
    scenario = load_cohort_v2_scenario_manifest(
        root / "scenario-manifest.json",
        xml_path=root / "scenario.xml",
    )
    manifest = scenario.scenario_manifest
    if (
        scenario.identity != record.get("scenario_manifest_identity")
        or manifest.scenario_lineage.identity
        != record.get("scenario_lineage_identity")
        or manifest.level_instance.identity != record.get("level_instance_identity")
        or manifest.scenario_template.identity
        != record.get("scenario_template_identity")
    ):
        raise SuccessorCohortError("successor scenario authority differs")
    role = record["exposure_role"]
    raw_shots = record.get("shots")
    raw_transitions = record.get("transitions")
    if (
        not isinstance(raw_shots, list)
        or not raw_shots
        or not isinstance(raw_transitions, list)
        or len(raw_shots) != len(raw_transitions)
    ):
        raise SuccessorCohortError("successor trajectory is not complete and multi-shot")
    shots = {
        index: _load_shot(root, item, release_identity=release_identity, role=role)
        for index, item in enumerate(raw_shots)
    }
    if any(item.get("shot_index") != index for index, item in enumerate(raw_shots)):
        raise SuccessorCohortError("successor shot order differs")
    for index, raw_shot in enumerate(raw_shots):
        capture = shots[index]["capture"]
        observation = shots[index]["observation_manifest"]
        bindings = capture.source_bindings
        observation_bindings = observation["source_bindings"]
        if (
            raw_shot.get("planned_action_identity")
            != record["planned_actions"][index]["identity"]
            or raw_shot.get("action", {}).get("identity")
            != raw_shot.get("planned_action_identity")
            or bindings["intervention_id"]
            != raw_shot.get("planned_action_identity")
            or bindings["scenario_lineage_id"]
            != record["scenario_lineage_identity"]
            or bindings["level_instance_id"] != record["level_instance_identity"]
            or bindings["scenario_template_id"]
            != record["scenario_template_identity"]
            or observation_bindings["rollout_identity"] != bindings["rollout_id"]
            or observation_bindings["source_scenario_lineage_identity"]
            != record["scenario_lineage_identity"]
        ):
            raise SuccessorCohortError("successor shot source binding differs")
    transitions = []
    observations_for_boundary = []
    for index, raw in enumerate(raw_transitions):
        if raw.get("decision_index") != index:
            raise SuccessorCohortError("successor transition order differs")

        def observation(reference: Mapping[str, Any] | None) -> AgentObservation | None:
            if reference is None:
                return None
            return _agent_observation(
                shots[int(reference["shot_index"])],
                int(reference["fixed_step"]),
                role,
            )

        prior = observation(raw.get("prior_observation"))
        current = observation(raw.get("current_observation"))
        next_observation = observation(raw.get("next_observation"))
        if current is None or next_observation is None:
            raise SuccessorCohortError("successor transition observation is missing")
        target_reference = raw["next_observation"]
        target_frame = shots[int(target_reference["shot_index"])]["frames"].get(
            int(target_reference["fixed_step"])
        )
        if target_frame is None:
            raise SuccessorCohortError("successor target frame is missing")
        raw_action = raw_shots[index]["action"]
        action = ExecutedAction(
            raw_action["identity"],
            raw_action["interface_action"],
            raw_action["engine_relative_action"],
            raw_action["legal"],
        )
        source_bindings = {
            "release_identity": release_identity,
            "scenario_manifest_identity": record["scenario_manifest_identity"],
            "source_capture_id": shots[index]["capture"].capture_id,
            "target_capture_id": target_frame.capture_id,
            "source_transition_path": raw_shots[index]["path"],
            "source_target_path": raw_shots[int(target_reference["shot_index"])][
                "path"
            ],
        }
        targets = DecisionTargets(
            next_observation,
            target_frame.identity,
            target_frame.state_id,
            {
                "engine_state": target_frame.engine_state,
                "events": target_frame.events,
                "predicates": target_frame.labels,
                "terminal": target_frame.terminal,
            },
        )
        transition = DecisionTransition(
            identity=raw["identity"],
            scenario_lineage_identity=record["scenario_lineage_identity"],
            exposure_role=role,
            decision_index=index,
            prior_observation=prior,
            current_observation=current,
            action=action,
            targets=targets,
            terminal_status=raw["terminal_status"],
            source_bindings=source_bindings,
        )
        expected_identity = successor_identity(
            "issue-62-decision-transition-v1",
            record["scenario_lineage_identity"],
            index,
            action.identity,
            current.identity,
            next_observation.identity,
        )
        if transition.identity != expected_identity:
            raise SuccessorCohortError("successor decision identity differs")
        transitions.append(transition)
        observations_for_boundary.append(
            shots[index]["observation_manifest"]
        )
    validate_observation_exposure_boundaries(observations_for_boundary)
    trajectory = DeploymentTrajectory(
        identity=record["trajectory_identity"],
        scenario_lineage_identity=record["scenario_lineage_identity"],
        exposure_role=role,
        transitions=tuple(transitions),
        complete=True,
    )
    expected_trajectory_identity = successor_identity(
        "issue-62-decision-trajectory-v1",
        record["scenario_lineage_identity"],
        tuple(item.identity for item in transitions),
    )
    if trajectory.identity != expected_trajectory_identity:
        raise SuccessorCohortError("successor trajectory identity differs")
    return trajectory


class SuccessorCohortReader:
    """Expose one non-final role as complete deployment decision trajectories."""

    def __init__(
        self,
        release_root: Path,
        *,
        exposure_role: str,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        if exposure_role not in PUBLIC_ROLES:
            raise SuccessorCohortError("successor reader permits only non-final roles")
        root = Path(release_root).resolve()
        manifest = _load(root / "manifest.json", "successor release manifest")
        plan = validate_successor_plan(
            _load(root / "production-plan.json", "successor production plan")
        )
        expected_release_identity = release_identity_for_plan(plan["identity"])
        if (
            manifest.get("schema") != RELEASE_SCHEMA
            or manifest.get("identity") != expected_release_identity
            or manifest.get("production_plan_identity") != plan["identity"]
            or manifest.get("included_roles") != list(PUBLIC_ROLES)
            or manifest.get("final_evaluation_collected") is not False
            or manifest.get("passed") is not True
        ):
            raise SuccessorCohortError("successor release identity or role boundary differs")
        records = [
            item for item in manifest.get("trajectories", ())
            if item.get("exposure_role") == exposure_role
        ]
        if len(records) != plan["role_counts"][exposure_role]:
            raise SuccessorCohortError("successor release role inventory differs")
        loaded = []
        for index, item in enumerate(records, start=1):
            loaded.append(load_successor_trajectory(
                root / item["path"], release_identity=expected_release_identity
            ))
            if progress is not None:
                progress(index, len(records))
        trajectories = tuple(loaded)
        bindings = tuple(
            TrajectoryLineageBinding(
                trajectory_identity=item.identity,
                scenario_lineage_identity=item.scenario_lineage_identity,
                exposure_role=item.exposure_role,
                transition_identities=tuple(
                    transition.identity for transition in item.transitions
                ),
                initial_observation_identity=(
                    item.transitions[0].current_observation.identity
                ),
                terminal_observation_identity=(
                    item.transitions[-1].targets.next_observation.identity
                ),
            )
            for item in trajectories
        )
        lineage_manifest = TrajectoryLineageManifest.create(
            expected_release_identity, bindings
        )
        expected_manifest_identity = manifest["lineage_manifests"][exposure_role]
        if lineage_manifest.identity != expected_manifest_identity:
            raise SuccessorCohortError("successor lineage manifest differs")
        self.release_identity = expected_release_identity
        self.exposure_role = exposure_role
        self.trajectories = trajectories
        self.lineage_manifest = lineage_manifest
        self.trajectory_reader = DeploymentTrajectoryReader(
            trajectories,
            exposure_role=exposure_role,
            lineage_manifest=lineage_manifest,
        )

    @staticmethod
    def validate_role_isolation(readers: tuple["SuccessorCohortReader", ...]) -> None:
        if {item.exposure_role for item in readers} != set(PUBLIC_ROLES):
            raise DeploymentTemporalError(
                "successor role isolation requires all three non-final roles"
            )
        DeploymentTrajectoryReader.validate_role_isolation(tuple(
            item.trajectory_reader for item in readers
        ))


__all__ = [
    "ACTION_BOUNDS",
    "BEHAVIOR_POLICIES",
    "GENERATOR_FAMILIES",
    "PILOT_REPORT_SCHEMA",
    "PLAN_SCHEMA",
    "PUBLIC_ROLES",
    "RELEASE_SCHEMA",
    "SuccessorCohortError",
    "SuccessorCohortReader",
    "TRAJECTORY_SCHEMA",
    "build_pilot_plan",
    "build_production_plan",
    "load_successor_trajectory",
    "release_identity_for_plan",
    "successor_identity",
    "validate_pilot_report",
    "validate_successor_plan",
]
