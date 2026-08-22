"""Accepted cohort-v2 macro derivation over validated physics-capture-v2 records."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, Final

from scripts.physics_capture_v2_types import PhysicsCaptureV2


DERIVATION_SCHEMA: Final = "cohort_v2_macro_derivation_v1"
DERIVATION_SPEC_SCHEMA: Final = "cohort_v2_macro_derivation_spec_v1"
DERIVATION_SPEC_VERSION: Final = "cohort_v2_macro_derivation_v1"
DERIVATION_SPEC_IDENTITY: Final = (
    "cohort-v2-macro-derivation-spec-v1:steady-state+structure-unstable"
)
LINEAR_SPEED_SQUARED_THRESHOLD: Final = 0.0001
ANGULAR_SPEED_THRESHOLD_DEGREES_PER_SECOND: Final = 0.01
DEBOUNCE_FIXED_STEPS: Final = 2
ACCEPTED_PREDICATES: Final = ("steady-state", "structure-unstable")
EXCLUDED_PREDICATES: Final = ("cascade-active", "collapsed", "pigs-cleared")
STABILITY_EVENT_TYPES: Final = frozenset(("stable_entered", "stable_exited"))


class CohortV2MacroSemanticsError(ValueError):
    """A v2 macro derivation is unavailable, stale, or semantically inconsistent."""


def derivation_spec(
    *,
    source_runtime_bundle_identity: str,
    source_snapshot_commit: str,
) -> dict[str, Any]:
    """Return the frozen accepted rules and numeric inputs for issue #49."""
    return {
        "schema": DERIVATION_SPEC_SCHEMA,
        "identity": DERIVATION_SPEC_IDENTITY,
        "version": DERIVATION_SPEC_VERSION,
        "capture_schema_version": "physics_capture_v2",
        "source_authority": {
            "runtime_bundle_identity": source_runtime_bundle_identity,
            "source_snapshot_commit": source_snapshot_commit,
        },
        "event_clock": {
            "occurrence_authority": "fixed_step",
            "projection_rule": "previous_state_fixed_step_lt_event_lte_current_state_fixed_step",
        },
        "numeric_inputs": {
            "linear_speed_squared_threshold": LINEAR_SPEED_SQUARED_THRESHOLD,
            "angular_speed_threshold_degrees_per_second": (
                ANGULAR_SPEED_THRESHOLD_DEGREES_PER_SECOND
            ),
            "debounce_fixed_steps": DEBOUNCE_FIXED_STEPS,
        },
        "rules": {
            "steady-state": (
                "A debounced state over complete consecutive fixed-step samples. A candidate "
                "is stable exactly when every dynamic body has squared linear speed at most "
                "0.0001 and absolute angular speed at most 0.01 degrees/second. The candidate "
                "must persist for two fixed steps before the state is established or changes; "
                "every post-initial change must match a same-step engine stability event."
            ),
            "structure-unstable": (
                "Available only with an immediately preceding complete fixed-step sample and "
                "an available steady-state label; true exactly when steady-state is false and "
                "the directed engine support-relation set differs from the predecessor."
            ),
        },
        "availability": {
            "steady-state": (
                "unavailable_incomplete_debounce_window until two consecutive candidate "
                "samples establish the initial state"
            ),
            "structure-unstable": (
                "unavailable_no_predecessor without an immediately preceding complete sample; "
                "unavailable_steady_state when its steady-state prerequisite is unavailable"
            ),
        },
        "accepted_predicates": list(ACCEPTED_PREDICATES),
        "excluded_predicates": {
            predicate: "excluded_not_emitted_not_false" for predicate in EXCLUDED_PREDICATES
        },
    }


def _support_pairs(sample: Mapping[str, Any]) -> frozenset[tuple[str, str]]:
    return frozenset(
        (support["supporter_entity_id"], support["supported_entity_id"])
        for support in sample["supports"]
    )


def _motion_evidence(sample: Mapping[str, Any]) -> tuple[bool, float, float]:
    maximum_linear_speed_squared = 0.0
    maximum_angular_speed = 0.0
    for entity in sample["entities"]:
        body = entity["body"]
        if not entity["body_present"] or body["body_type"] != "dynamic":
            continue
        velocity = body["velocity"]
        linear_speed_squared = float(velocity[0]) ** 2 + float(velocity[1]) ** 2
        angular_speed = abs(float(body["angular_velocity_degrees_per_second"]))
        maximum_linear_speed_squared = max(maximum_linear_speed_squared, linear_speed_squared)
        maximum_angular_speed = max(maximum_angular_speed, angular_speed)
    quiet = (
        maximum_linear_speed_squared <= LINEAR_SPEED_SQUARED_THRESHOLD
        and maximum_angular_speed <= ANGULAR_SPEED_THRESHOLD_DEGREES_PER_SECOND
    )
    return quiet, maximum_linear_speed_squared, maximum_angular_speed


def _source_interval(
    capture: PhysicsCaptureV2,
    source_reference: str,
    previous_fixed_step: int | None,
    fixed_step: int,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    event_ids = [
        event["event_id"]
        for event in events
        if (previous_fixed_step is None or event["fixed_step"] > previous_fixed_step)
        and event["fixed_step"] <= fixed_step
    ]
    return {
        "source_capture_reference": source_reference,
        "capture_id": capture.capture_id,
        "shot_id": capture.shot_id,
        "previous_fixed_step": previous_fixed_step,
        "current_fixed_step": fixed_step,
        "projected_event_ids": event_ids,
        "derivation_spec_version": DERIVATION_SPEC_VERSION,
    }


def derive_capture_macro_labels(
    capture: PhysicsCaptureV2,
    *,
    source_reference: str,
    source_capture_bundle_identity: str,
) -> dict[str, Any]:
    """Derive both accepted predicates and verify engine stability transitions exactly."""
    record = capture.record
    samples = record["fixed_step_samples"]
    events = record["events"]
    stability_events_by_step = {
        event["fixed_step"]: event
        for event in events
        if event["event_type"] in STABILITY_EVENT_TYPES
    }
    if len(stability_events_by_step) != sum(
        event["event_type"] in STABILITY_EVENT_TYPES for event in events
    ):
        raise CohortV2MacroSemanticsError(
            f"{capture.capture_id}: multiple stability events share one fixed step"
        )

    stable_state: bool | None = None
    candidate_state: bool | None = None
    candidate_run_start = 0
    candidate_run_length = 0
    last_transition_event_id: str | None = None
    previous_step: int | None = None
    previous_supports: frozenset[tuple[str, str]] | None = None
    labels: list[dict[str, Any]] = []

    for sample in samples:
        step = sample["fixed_step"]
        quiet, max_linear_squared, max_angular = _motion_evidence(sample)
        if candidate_state is None or candidate_state != quiet:
            candidate_state = quiet
            candidate_run_start = step
            candidate_run_length = 1
        else:
            candidate_run_length += 1

        prior_stable_state = stable_state
        initial_acquisition = stable_state is None and candidate_run_length >= DEBOUNCE_FIXED_STEPS
        transition = (
            stable_state is not None
            and candidate_state != stable_state
            and candidate_run_length >= DEBOUNCE_FIXED_STEPS
        )
        if initial_acquisition or transition:
            stable_state = candidate_state

        stability_event = stability_events_by_step.get(step)
        if transition:
            expected_type = "stable_entered" if stable_state else "stable_exited"
            if stability_event is None or stability_event["event_type"] != expected_type:
                raise CohortV2MacroSemanticsError(
                    f"{capture.capture_id} fixed step {step}: derived {expected_type} "
                    "does not match a same-step engine event"
                )
            last_transition_event_id = stability_event["event_id"]
        elif stability_event is not None:
            if not (
                initial_acquisition
                and stability_event["event_type"]
                == ("stable_entered" if stable_state else "stable_exited")
            ):
                raise CohortV2MacroSemanticsError(
                    f"{capture.capture_id} fixed step {step}: engine stability event "
                    "does not match the debounced state transition"
                )
            last_transition_event_id = stability_event["event_id"]

        source_interval = _source_interval(
            capture,
            source_reference,
            previous_step,
            step,
            events,
        )
        if stable_state is None:
            steady_value = None
            steady_availability = "unavailable_incomplete_debounce_window"
        else:
            steady_value = stable_state
            steady_availability = "available"
        steady_label = {
            "value": steady_value,
            "availability": steady_availability,
            "source_interval": source_interval,
            "evidence": {
                "candidate_stable": quiet,
                "candidate_run_start_fixed_step": candidate_run_start,
                "candidate_run_length": candidate_run_length,
                "maximum_linear_speed_squared": max_linear_squared,
                "maximum_absolute_angular_speed_degrees_per_second": max_angular,
                "transition_event_id": (
                    stability_event["event_id"]
                    if stability_event is not None and (transition or initial_acquisition)
                    else None
                ),
                "last_transition_event_id": last_transition_event_id,
                "prior_debounced_value": prior_stable_state,
            },
        }

        supports = _support_pairs(sample)
        if previous_supports is None:
            unstable_value = None
            unstable_availability = "unavailable_no_predecessor"
            added_supports: list[list[str]] = []
            removed_supports: list[list[str]] = []
        elif stable_state is None:
            unstable_value = None
            unstable_availability = "unavailable_steady_state"
            added_supports = []
            removed_supports = []
        else:
            added_supports = [list(pair) for pair in sorted(supports - previous_supports)]
            removed_supports = [list(pair) for pair in sorted(previous_supports - supports)]
            unstable_value = not stable_state and bool(added_supports or removed_supports)
            unstable_availability = "available"
        unstable_label = {
            "value": unstable_value,
            "availability": unstable_availability,
            "source_interval": source_interval,
            "evidence": {
                "added_support_relations": added_supports,
                "removed_support_relations": removed_supports,
                "steady_state_value": steady_value,
            },
        }
        labels.append(
            {
                "fixed_step": step,
                "predicates": {
                    "steady-state": steady_label,
                    "structure-unstable": unstable_label,
                },
            }
        )
        previous_step = step
        previous_supports = supports

    return {
        "schema": DERIVATION_SCHEMA,
        "identity": f"cohort-v2-macro-derivation-v1:{capture.capture_id}",
        "derivation_spec_identity": DERIVATION_SPEC_IDENTITY,
        "source": {
            "capture_reference": source_reference,
            "capture_bundle_identity": source_capture_bundle_identity,
            "capture_id": capture.capture_id,
            "shot_id": capture.shot_id,
            "source_bindings": dict(capture.source_bindings),
        },
        "predicates": list(ACCEPTED_PREDICATES),
        "label_count": len(labels),
        "labels": labels,
    }


def validate_capture_macro_derivation(
    derivation: Mapping[str, Any],
    capture: PhysicsCaptureV2,
    *,
    source_reference: str,
    source_capture_bundle_identity: str,
) -> None:
    expected = derive_capture_macro_labels(
        capture,
        source_reference=source_reference,
        source_capture_bundle_identity=source_capture_bundle_identity,
    )
    if derivation != expected:
        raise CohortV2MacroSemanticsError(
            f"{capture.capture_id}: macro derivation differs from exact re-derivation"
        )


def finite_json_tree(value: object) -> bool:
    """Return whether all numeric evidence is finite; useful for bundle validation."""
    if type(value) in (int, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(finite_json_tree(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return all(finite_json_tree(child) for child in value)
    return value is None or isinstance(value, (str, bool))
