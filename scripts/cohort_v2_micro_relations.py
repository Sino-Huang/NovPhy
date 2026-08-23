"""Source-bound contact and support derivations for cohort-v2 captures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from scripts.physics_capture_v2_types import PhysicsCaptureV2


DERIVATION_SCHEMA: Final = "cohort_v2_micro_relation_derivation_v1"
DERIVATION_SPEC_IDENTITY: Final = (
    "cohort-v2-micro-relation-derivation-spec-v1:contact+supports"
)
ACCEPTED_PREDICATES: Final = ("contact", "supports")


class CohortV2MicroRelationError(ValueError):
    """A micro-relation derivation is incomplete or differs from its source."""


def derive_capture_micro_relations(
    capture: PhysicsCaptureV2,
    *,
    source_reference: str,
    source_capture_bundle_identity: str,
) -> dict[str, Any]:
    """Project complete engine contacts and support_v1 relations by fixed step."""
    labels = []
    for sample in capture.record["fixed_step_samples"]:
        if sample["complete_raw_non_trigger_contacts"] is not True:
            raise CohortV2MicroRelationError(
                f"{capture.capture_id} fixed step {sample['fixed_step']}: "
                "contact enumeration is incomplete"
            )
        contact_pairs = sorted(
            {
                tuple(sorted((contact["entity_a_id"], contact["entity_b_id"])))
                for contact in sample["contacts"]
            }
        )
        support_pairs = sorted(
            {
                (
                    support["supporter_entity_id"],
                    support["supported_entity_id"],
                )
                for support in sample["supports"]
            }
        )
        labels.append(
            {
                "fixed_step": sample["fixed_step"],
                "predicates": {
                    "contact": {
                        "availability": "available",
                        "relations": [list(pair) for pair in contact_pairs],
                        "complete_raw_non_trigger_contacts": True,
                    },
                    "supports": {
                        "availability": "available",
                        "relations": [list(pair) for pair in support_pairs],
                        "derivation_version": "support_v1",
                    },
                },
            }
        )
    return {
        "schema": DERIVATION_SCHEMA,
        "identity": f"cohort-v2-micro-relation-derivation-v1:{capture.capture_id}",
        "derivation_spec_identity": DERIVATION_SPEC_IDENTITY,
        "source": {
            "capture_reference": source_reference,
            "capture_bundle_identity": source_capture_bundle_identity,
            "capture_id": capture.capture_id,
            "shot_id": capture.shot_id,
            "source_bindings": dict(capture.source_bindings),
        },
        "predicates": list(ACCEPTED_PREDICATES),
        "fixed_step_count": len(labels),
        "labels": labels,
    }


def validate_capture_micro_relation_derivation(
    derivation: Mapping[str, Any],
    capture: PhysicsCaptureV2,
    *,
    source_reference: str,
    source_capture_bundle_identity: str,
) -> None:
    expected = derive_capture_micro_relations(
        capture,
        source_reference=source_reference,
        source_capture_bundle_identity=source_capture_bundle_identity,
    )
    if derivation != expected:
        raise CohortV2MicroRelationError(
            f"{capture.capture_id}: micro-relation derivation differs from exact re-derivation"
        )

