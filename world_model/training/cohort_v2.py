"""Training and evaluation consumers for validated cohort-v2 central tuples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import torch

from world_model.data.cohort_v2 import (
    CAPABILITY_DECLARATION_IDENTITY,
    CENTRAL_LABELS,
    COHORT_V2_TRANSITION_RELEASE_IDENTITIES,
    CohortV2CentralFrameRecord,
    CohortV2IngestionError,
    CohortV2OracleWindow,
    CohortV2OracleWindowDataset,
    CohortV2ReleaseReader,
)
from world_model.model import (
    Abstraction,
    BooleanTransitionValue,
    MacroTransitionBatch,
    MacroTransitionInput,
    MicroTransitionBatch,
    MicroTransitionInput,
    PredictionPair,
    RelationTransitionValue,
    TransitionRequest,
)


PHYSICAL_VIOLATION_ENDPOINT_QUANTITIES = (
    "excess_penetration",
    "unsupported_stationary_or_floating_body",
)


class LabelAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class CohortV2EndpointScore:
    endpoint_count: int
    scored_value_count: int
    correct_value_count: int
    unavailable_value_count: int
    scored_relation_count: int
    correct_relation_count: int
    unavailable_relation_count: int


@dataclass(frozen=True, slots=True)
class CohortV2EndpointPlausibility:
    """Accepted physical-violation incidence at one source endpoint."""

    available_value_count: int
    unavailable_value_count: int
    violation_count: int

    @property
    def violation_rate(self) -> float | None:
        if self.available_value_count == 0:
            return None
        return self.violation_count / self.available_value_count


class CohortV2EndpointPredictor(Protocol):
    def __call__(self, frame_record: CohortV2CentralFrameRecord) -> Mapping[str, Any]: ...


def build_cohort_v2_oracle_window_loader(
    dataset: CohortV2OracleWindowDataset,
) -> torch.utils.data.DataLoader:
    """Build the deterministic training loader used by the ingestion smoke path."""
    if not isinstance(dataset, CohortV2OracleWindowDataset):
        raise CohortV2IngestionError(
            "Cohort-v2 training requires a validated oracle-window dataset"
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=tuple,
    )


def _relation_input(
    window: CohortV2OracleWindow, predicate: str
) -> RelationTransitionValue:
    label = window.context.labels[predicate]
    availability = label.get("availability")
    relations = label.get("relations")
    if availability == "available":
        return RelationTransitionValue(availability, relations)
    if isinstance(availability, str) and availability.startswith("unavailable_"):
        if relations not in (None, ()):
            raise CohortV2IngestionError(
                f"Unavailable {predicate} contains fabricated relations"
            )
        return RelationTransitionValue(availability, None)
    raise CohortV2IngestionError(f"{predicate} availability is malformed")


def _boolean_input(
    window: CohortV2OracleWindow, predicate: str
) -> BooleanTransitionValue:
    label = window.context.labels[predicate]
    availability = label.get("availability")
    value = label.get("value")
    if availability == "available":
        return BooleanTransitionValue(availability, value)
    if isinstance(availability, str) and availability.startswith("unavailable_"):
        if value is not None:
            raise CohortV2IngestionError(
                f"Unavailable {predicate} contains a fabricated value"
            )
        return BooleanTransitionValue(availability, None)
    raise CohortV2IngestionError(f"{predicate} availability is malformed")


def build_cohort_v2_transition_request(
    pair: PredictionPair,
    windows: tuple[CohortV2OracleWindow, ...],
) -> TransitionRequest:
    """Adapt validated oracle windows to one exclusive model request."""
    if type(pair) is not PredictionPair:
        raise CohortV2IngestionError("Transition request requires a prediction pair")
    if type(windows) is not tuple or not windows or any(
        type(window) is not CohortV2OracleWindow for window in windows
    ):
        raise CohortV2IngestionError(
            "Transition request requires validated cohort-v2 oracle windows"
        )
    for window in windows:
        if (
            window.source_release_identity
            not in COHORT_V2_TRANSITION_RELEASE_IDENTITIES
            or window.capability_declaration_identity
            != CAPABILITY_DECLARATION_IDENTITY
        ):
            raise CohortV2IngestionError(
                "Transition request crosses the approved cohort-v2 release boundary"
            )
        if window.requested_horizon != pair.delta:
            raise CohortV2IngestionError(
                "Transition request horizon does not match its oracle windows"
            )
    if pair.abstraction is Abstraction.CONTINUOUS:
        return TransitionRequest(pair, None)
    if pair.abstraction is Abstraction.MICRO:
        samples = tuple(
            MicroTransitionInput(
                frame_record_identity=window.context.identity,
                contact=_relation_input(window, "contact"),
                supports=_relation_input(window, "supports"),
            )
            for window in windows
        )
        return TransitionRequest(pair, MicroTransitionBatch(samples))
    samples = tuple(
        MacroTransitionInput(
            frame_record_identity=window.context.identity,
            steady_state=_boolean_input(window, "steady-state"),
            structure_unstable=_boolean_input(window, "structure-unstable"),
        )
        for window in windows
    )
    return TransitionRequest(pair, MacroTransitionBatch(samples))


def _availability(label: Mapping[str, Any]) -> LabelAvailability:
    value = label.get("availability")
    if value == LabelAvailability.AVAILABLE:
        return LabelAvailability.AVAILABLE
    if isinstance(value, str) and value.startswith("unavailable_"):
        return LabelAvailability.UNAVAILABLE
    raise CohortV2IngestionError("Central label availability is malformed")


def _score_boolean_label(
    label: Mapping[str, Any], prediction: Any
) -> tuple[int, int, int]:
    availability = _availability(label)
    value = label.get("value")
    if availability is LabelAvailability.AVAILABLE:
        if type(value) is not bool or type(prediction) is not bool:
            raise CohortV2IngestionError(
                "Available endpoint labels require boolean predictions"
            )
        return 1, int(value == prediction), 0
    if value is not None:
        raise CohortV2IngestionError("Unavailable endpoint label was converted to a value")
    return 0, 0, 1


def score_cohort_v2_endpoint_plausibility(
    endpoint: CohortV2CentralFrameRecord,
) -> CohortV2EndpointPlausibility:
    """Summarize the two accepted derivations already attached to a v5 record.

    The derivations are engine-record quantities, not a decoder for a predicted
    continuous carrier. This function consumes their accepted labels and never
    recreates their physical thresholds.
    """
    if not isinstance(endpoint, CohortV2CentralFrameRecord) or set(
        endpoint.labels
    ) != set(CENTRAL_LABELS):
        raise CohortV2IngestionError("Endpoint central tuple is incomplete")
    labels = [endpoint.labels[PHYSICAL_VIOLATION_ENDPOINT_QUANTITIES[0]]]
    unsupported = endpoint.labels[PHYSICAL_VIOLATION_ENDPOINT_QUANTITIES[1]]
    if not isinstance(unsupported, tuple):
        raise CohortV2IngestionError("Unsupported-body endpoint labels are malformed")
    labels.extend(unsupported)

    available = unavailable = violations = 0
    for label in labels:
        if not isinstance(label, Mapping):
            raise CohortV2IngestionError("Physical-violation endpoint label is malformed")
        scored, plausible, missing = _score_boolean_label(label, False)
        available += scored
        unavailable += missing
        violations += scored - plausible
    return CohortV2EndpointPlausibility(
        available_value_count=available,
        unavailable_value_count=unavailable,
        violation_count=violations,
    )


def _relation_pairs(value: Any, label: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CohortV2IngestionError(f"{label} relation prediction is malformed")
    pairs = tuple(tuple(item) for item in value)
    if not all(
        len(pair) == 2 and all(isinstance(entity, str) and entity for entity in pair)
        for pair in pairs
    ):
        raise CohortV2IngestionError(f"{label} relation prediction is malformed")
    return pairs


def _score_relation_label(
    predicate: str,
    label: Mapping[str, Any],
    prediction: Any,
) -> tuple[int, int, int]:
    availability = _availability(label)
    if availability is LabelAvailability.UNAVAILABLE:
        if label.get("relations") not in (None, ()):
            raise CohortV2IngestionError(
                "Unavailable relation label was converted to relations"
            )
        return 0, 0, 1
    truth = _relation_pairs(label.get("relations"), predicate)
    predicted = _relation_pairs(prediction, predicate)
    return 1, int(truth == predicted), 0


def score_cohort_v2_endpoints(
    reader: CohortV2ReleaseReader,
    predictor: CohortV2EndpointPredictor,
) -> CohortV2EndpointScore:
    """Score complete role endpoints without turning unavailable values into negatives."""
    scored = correct = unavailable = 0
    scored_relations = correct_relations = unavailable_relations = 0
    for rollout in reader.rollouts:
        endpoint = rollout.frame_records[-1]
        if endpoint.terminal is None or set(endpoint.labels) != set(CENTRAL_LABELS):
            raise CohortV2IngestionError("Endpoint central tuple is incomplete")
        prediction = predictor(endpoint)
        if not isinstance(prediction, Mapping) or set(prediction) != set(CENTRAL_LABELS):
            raise CohortV2IngestionError(
                "Endpoint predictor returned a partial central prediction"
            )
        for predicate in ("contact", "supports"):
            counts = _score_relation_label(
                predicate, endpoint.labels[predicate], prediction[predicate]
            )
            scored_relations += counts[0]
            correct_relations += counts[1]
            unavailable_relations += counts[2]
        for predicate in (
            "steady-state", "structure-unstable", "excess_penetration"
        ):
            counts = _score_boolean_label(
                endpoint.labels[predicate], prediction[predicate]
            )
            scored += counts[0]
            correct += counts[1]
            unavailable += counts[2]
        unsupported = endpoint.labels["unsupported_stationary_or_floating_body"]
        unsupported_prediction = prediction[
            "unsupported_stationary_or_floating_body"
        ]
        if not isinstance(unsupported, tuple) or not isinstance(
            unsupported_prediction, Mapping
        ):
            raise CohortV2IngestionError(
                "Unsupported-body endpoint prediction is malformed"
            )
        entity_ids = tuple(item.get("entity_id") for item in unsupported)
        if (
            not all(isinstance(item, str) and item for item in entity_ids)
            or len(set(entity_ids)) != len(entity_ids)
            or set(unsupported_prediction) != set(entity_ids)
        ):
            raise CohortV2IngestionError("Unsupported-body endpoint prediction is partial")
        for item in unsupported:
            counts = _score_boolean_label(
                item, unsupported_prediction[item["entity_id"]]
            )
            scored += counts[0]
            correct += counts[1]
            unavailable += counts[2]
    return CohortV2EndpointScore(
        endpoint_count=len(reader.rollouts),
        scored_value_count=scored,
        correct_value_count=correct,
        unavailable_value_count=unavailable,
        scored_relation_count=scored_relations,
        correct_relation_count=correct_relations,
        unavailable_relation_count=unavailable_relations,
    )


__all__ = [
    "CohortV2EndpointPredictor",
    "CohortV2EndpointPlausibility",
    "CohortV2EndpointScore",
    "LabelAvailability",
    "PHYSICAL_VIOLATION_ENDPOINT_QUANTITIES",
    "build_cohort_v2_oracle_window_loader",
    "build_cohort_v2_transition_request",
    "score_cohort_v2_endpoint_plausibility",
    "score_cohort_v2_endpoints",
]
