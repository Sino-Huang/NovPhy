"""Accepted cohort-v2 endpoint physical-violation derivation."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, Final

from scripts.physics_capture_v2_types import PhysicsCaptureV2


DERIVATION_SCHEMA: Final = "cohort_v2_physical_violation_derivation_v1"
DERIVATION_SPEC_SCHEMA: Final = "cohort_v2_physical_violation_derivation_spec_v1"
DERIVATION_SPEC_VERSION: Final = "cohort_v2_physical_violation_derivation_v1"
DERIVATION_SPEC_IDENTITY: Final = (
    "cohort-v2-physical-violation-derivation-spec-v1:"
    "excess-penetration+unsupported-stationary-or-floating-body"
)
EXCESS_PENETRATION: Final = "excess_penetration"
UNSUPPORTED_STATIONARY: Final = "unsupported_stationary_or_floating_body"
AGGREGATE_PREDICATE: Final = "any(violation)"
ACCEPTED_PREDICATES: Final = (EXCESS_PENETRATION, UNSUPPORTED_STATIONARY)
EXCLUDED_PREDICATES: Final = ("illegal_contact",)

# Frozen before issue #50's semantic-probe collection. Resting contact slop in
# the accepted #44 probes is about 0.005 Unity units; the 0.006 tolerance keeps
# it negative while retaining independently observed impact witnesses.
PENETRATION_TOLERANCE_UNITY_UNITS: Final = 0.006
LINEAR_SPEED_SQUARED_THRESHOLD: Final = 0.0001
ANGULAR_SPEED_THRESHOLD_DEGREES_PER_SECOND: Final = 0.01
STABILITY_WINDOW_FIXED_STEPS: Final = 2


class CohortV2PhysicalViolationError(ValueError):
    """A physical-violation derivation is stale or semantically inconsistent."""


def derivation_spec(
    *,
    source_authorities: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Return issue #50's frozen two-label rules and aggregate semantics."""
    return {
        "schema": DERIVATION_SPEC_SCHEMA,
        "identity": DERIVATION_SPEC_IDENTITY,
        "version": DERIVATION_SPEC_VERSION,
        "capture_schema_version": "physics_capture_v2",
        "source_authorities": [dict(authority) for authority in source_authorities],
        "numeric_inputs": {
            "penetration_tolerance_unity_units": PENETRATION_TOLERANCE_UNITY_UNITS,
            "linear_speed_squared_threshold": LINEAR_SPEED_SQUARED_THRESHOLD,
            "angular_speed_threshold_degrees_per_second": (
                ANGULAR_SPEED_THRESHOLD_DEGREES_PER_SECOND
            ),
            "stability_window_fixed_steps": STABILITY_WINDOW_FIXED_STEPS,
        },
        "rules": {
            EXCESS_PENETRATION: (
                "At a retained fixed step, true exactly when at least one complete "
                "Unity-authored non-trigger contact has separation strictly less than "
                "-0.006 Unity units. Complete enumeration with no such contact is false."
            ),
            UNSUPPORTED_STATIONARY: (
                "For each causal entity, true exactly when two consecutive complete "
                "fixed-step records show an active present body under applicable nonzero "
                "world gravity, squared linear speed at most 0.0001, absolute angular "
                "speed at most 0.01 degrees/second, and no support relation anywhere in "
                "the two-step window. Complete support or nonstationarity is false."
            ),
            AGGREGATE_PREDICATE: (
                "Kleene any over excess penetration and the active-present body domain: "
                "any true component is true; all available false components are false; "
                "otherwise the aggregate is unavailable. Concept-undefined inactive or "
                "absent entity labels remain separately unavailable and are outside the "
                "aggregate domain; no unavailable input is converted to zero."
            ),
        },
        "availability": {
            EXCESS_PENETRATION: (
                "available only after whole-capture validation proves collider geometry, "
                "coordinate convention, separation, and fixed-step contact completeness"
            ),
            UNSUPPORTED_STATIONARY: (
                "unavailable_incomplete_stability_window until two consecutive active "
                "present-body records exist; unavailable_inactive_or_absent_body when "
                "the concept is undefined at the endpoint"
            ),
            AGGREGATE_PREDICATE: "unavailable_component unless a true component dominates",
        },
        "accepted_predicates": list(ACCEPTED_PREDICATES),
        "aggregate_predicate": AGGREGATE_PREDICATE,
        "excluded_predicates": {
            predicate: "excluded_not_emitted_not_false"
            for predicate in EXCLUDED_PREDICATES
        },
        "physical_regime_label_used": False,
    }


def _source_records(
    capture: PhysicsCaptureV2,
    source_reference: str,
    source_capture_bundle_identity: str,
    fixed_steps: Sequence[int],
    *,
    entity_id: str | None = None,
    contact_ids: Sequence[str] = (),
    collider_ids: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "source_capture_reference": source_reference,
        "source_capture_bundle_identity": source_capture_bundle_identity,
        "capture_id": capture.capture_id,
        "shot_id": capture.shot_id,
        "fixed_steps": list(fixed_steps),
        "entity_id": entity_id,
        "contact_ids": sorted(set(contact_ids)),
        "collider_ids": sorted(set(collider_ids)),
        "derivation_spec_version": DERIVATION_SPEC_VERSION,
        "derivation_spec_identity": DERIVATION_SPEC_IDENTITY,
    }


def _excess_penetration_label(
    capture: PhysicsCaptureV2,
    sample: Mapping[str, Any],
    source_reference: str,
    source_capture_bundle_identity: str,
) -> dict[str, Any]:
    contacts = sample["contacts"]
    minimum = min(
        contacts,
        key=lambda contact: (contact["separation"], contact["contact_id"]),
        default=None,
    )
    violating = [
        contact
        for contact in contacts
        if contact["separation"] < -PENETRATION_TOLERANCE_UNITY_UNITS
    ]
    cited_contacts = violating if violating else ([] if minimum is None else [minimum])
    return {
        "value": bool(violating),
        "availability": "available",
        "source_records": _source_records(
            capture,
            source_reference,
            source_capture_bundle_identity,
            [sample["fixed_step"]],
            contact_ids=[contact["contact_id"] for contact in cited_contacts],
            collider_ids=[
                collider_id
                for contact in cited_contacts
                for collider_id in (contact["collider_a_id"], contact["collider_b_id"])
            ],
        ),
        "evidence": {
            "coordinate_convention": dict(capture.record["coordinate_convention"]),
            "collider_geometry_complete": True,
            "raw_non_trigger_contact_enumeration_complete": sample[
                "complete_raw_non_trigger_contacts"
            ],
            "penetration_tolerance_unity_units": PENETRATION_TOLERANCE_UNITY_UNITS,
            "minimum_contact_separation": (
                None if minimum is None else minimum["separation"]
            ),
            "violating_contact_ids": [contact["contact_id"] for contact in violating],
        },
    }


def _entity_by_id(sample: Mapping[str, Any], entity_id: str) -> Mapping[str, Any]:
    return next(entity for entity in sample["entities"] if entity["entity_id"] == entity_id)


def _support_evidence(
    sample: Mapping[str, Any], entity_id: str
) -> list[dict[str, Any]]:
    return [
        {
            "fixed_step": sample["fixed_step"],
            "supporter_entity_id": support["supporter_entity_id"],
            "contact_ids": list(support["contact_ids"]),
        }
        for support in sample["supports"]
        if support["supported_entity_id"] == entity_id
    ]


def _unsupported_label(
    capture: PhysicsCaptureV2,
    samples: Sequence[Mapping[str, Any]],
    entity_id: str,
    source_reference: str,
    source_capture_bundle_identity: str,
) -> dict[str, Any]:
    current = _entity_by_id(samples[-1], entity_id)
    fixed_steps = [sample["fixed_step"] for sample in samples]
    support_rows = [
        support
        for sample in samples
        for support in _support_evidence(sample, entity_id)
    ]
    contact_ids = [
        contact_id
        for sample in samples
        for contact_id in _entity_by_id(sample, entity_id)["contact_ids"]
    ]
    source_records = _source_records(
        capture,
        source_reference,
        source_capture_bundle_identity,
        fixed_steps,
        entity_id=entity_id,
        contact_ids=contact_ids,
    )

    if current["lifecycle"] != "active" or not current["body_present"]:
        return {
            "entity_id": entity_id,
            "value": None,
            "availability": "unavailable_inactive_or_absent_body",
            "source_records": source_records,
            "evidence": {
                "lifecycle": current["lifecycle"],
                "body_present": current["body_present"],
            },
        }

    complete_window = len(samples) == STABILITY_WINDOW_FIXED_STEPS and all(
        (entity := _entity_by_id(sample, entity_id))["lifecycle"] == "active"
        and entity["body_present"]
        for sample in samples
    )
    if not complete_window:
        return {
            "entity_id": entity_id,
            "value": None,
            "availability": "unavailable_incomplete_stability_window",
            "source_records": source_records,
            "evidence": {
                "required_fixed_step_count": STABILITY_WINDOW_FIXED_STEPS,
                "observed_fixed_step_count": len(samples),
                "active_present_body_window_complete": False,
            },
        }

    bodies = [_entity_by_id(sample, entity_id)["body"] for sample in samples]
    worlds = [sample["world"] for sample in samples]
    linear_speed_squared = [
        float(body["velocity"][0]) ** 2 + float(body["velocity"][1]) ** 2
        for body in bodies
    ]
    angular_speed = [
        abs(float(body["angular_velocity_degrees_per_second"])) for body in bodies
    ]
    gravity_applicable = all(
        body["gravity_applicable"]
        and (
            float(world["gravity_vector"][0]) ** 2
            + float(world["gravity_vector"][1]) ** 2
        )
        > 0.0
        for body, world in zip(bodies, worlds)
    )
    stationary = (
        max(linear_speed_squared) <= LINEAR_SPEED_SQUARED_THRESHOLD
        and max(angular_speed) <= ANGULAR_SPEED_THRESHOLD_DEGREES_PER_SECOND
    )
    supported = bool(support_rows)
    value = gravity_applicable and stationary and not supported
    return {
        "entity_id": entity_id,
        "value": value,
        "availability": "available",
        "source_records": source_records,
        "evidence": {
            "active_lifecycle_complete": True,
            "body_presence_complete": True,
            "gravity_applicable_all_steps": gravity_applicable,
            "world_context": [
                {
                    "fixed_step": sample["fixed_step"],
                    "world_id": sample["world"]["world_id"],
                    "gravity_vector": list(sample["world"]["gravity_vector"]),
                }
                for sample in samples
            ],
            "linear_speed_squared_by_fixed_step": linear_speed_squared,
            "maximum_linear_speed_squared": max(linear_speed_squared),
            "linear_speed_squared_threshold": LINEAR_SPEED_SQUARED_THRESHOLD,
            "absolute_angular_speed_by_fixed_step": angular_speed,
            "maximum_absolute_angular_speed_degrees_per_second": max(angular_speed),
            "angular_speed_threshold_degrees_per_second": (
                ANGULAR_SPEED_THRESHOLD_DEGREES_PER_SECOND
            ),
            "stability_window_fixed_steps": STABILITY_WINDOW_FIXED_STEPS,
            "support_relations": support_rows,
            "support_contact_evidence_complete": True,
            "stationary": stationary,
            "supported": supported,
        },
    }


def _aggregate(excess: Mapping[str, Any], unsupported: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate_body_labels = [
        label
        for label in unsupported
        if label["availability"] != "unavailable_inactive_or_absent_body"
    ]
    components = [
        {
            "predicate": EXCESS_PENETRATION,
            "entity_id": None,
            "value": excess["value"],
            "availability": excess["availability"],
        },
        *(
            {
                "predicate": UNSUPPORTED_STATIONARY,
                "entity_id": label["entity_id"],
                "value": label["value"],
                "availability": label["availability"],
            }
            for label in aggregate_body_labels
        ),
    ]
    if any(component["value"] is True for component in components):
        value = True
        availability = "available"
    elif any(component["value"] is None for component in components):
        value = None
        availability = "unavailable_component"
    else:
        value = False
        availability = "available"
    return {
        "predicate": AGGREGATE_PREDICATE,
        "value": value,
        "availability": availability,
        "components": components,
    }


def derive_capture_physical_violations(
    capture: PhysicsCaptureV2,
    *,
    source_reference: str,
    source_capture_bundle_identity: str,
) -> dict[str, Any]:
    """Derive both accepted predicates and the unavailable-preserving aggregate."""
    samples = capture.record["fixed_step_samples"]
    entity_ids = list(capture.record["causal_entities"])
    labels: list[dict[str, Any]] = []
    label_count = 0
    for index, sample in enumerate(samples):
        window = samples[max(0, index - STABILITY_WINDOW_FIXED_STEPS + 1): index + 1]
        excess = _excess_penetration_label(
            capture,
            sample,
            source_reference,
            source_capture_bundle_identity,
        )
        unsupported = [
            _unsupported_label(
                capture,
                window,
                entity_id,
                source_reference,
                source_capture_bundle_identity,
            )
            for entity_id in entity_ids
        ]
        aggregate = _aggregate(excess, unsupported)
        labels.append(
            {
                "fixed_step": sample["fixed_step"],
                "predicates": {
                    EXCESS_PENETRATION: excess,
                    UNSUPPORTED_STATIONARY: unsupported,
                },
                "aggregate": aggregate,
            }
        )
        label_count += 2 + len(unsupported)

    return {
        "schema": DERIVATION_SCHEMA,
        "identity": f"cohort-v2-physical-violation-derivation-v1:{capture.capture_id}",
        "derivation_spec_identity": DERIVATION_SPEC_IDENTITY,
        "source": {
            "capture_reference": source_reference,
            "capture_bundle_identity": source_capture_bundle_identity,
            "capture_id": capture.capture_id,
            "shot_id": capture.shot_id,
            "source_bindings": dict(capture.source_bindings),
        },
        "predicates": list(ACCEPTED_PREDICATES),
        "aggregate_predicate": AGGREGATE_PREDICATE,
        "excluded_predicates": {
            predicate: "excluded_not_emitted_not_false"
            for predicate in EXCLUDED_PREDICATES
        },
        "fixed_step_count": len(labels),
        "label_count": label_count,
        "labels": labels,
    }


def validate_capture_physical_violation_derivation(
    derivation: Mapping[str, Any],
    capture: PhysicsCaptureV2,
    *,
    source_reference: str,
    source_capture_bundle_identity: str,
) -> None:
    expected = derive_capture_physical_violations(
        capture,
        source_reference=source_reference,
        source_capture_bundle_identity=source_capture_bundle_identity,
    )
    if derivation != expected:
        raise CohortV2PhysicalViolationError(
            f"{capture.capture_id}: physical-violation derivation differs from exact re-derivation"
        )


def finite_json_tree(value: object) -> bool:
    if type(value) in (int, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(finite_json_tree(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return all(finite_json_tree(child) for child in value)
    return value is None or isinstance(value, (str, bool))
