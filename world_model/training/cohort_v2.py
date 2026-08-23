"""Training and evaluation consumers for validated cohort-v2 central tuples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import torch

from world_model.data.cohort_v2 import (
    CENTRAL_LABELS,
    CohortV2CentralFrameRecord,
    CohortV2IngestionError,
    CohortV2OracleWindowDataset,
    CohortV2ReleaseReader,
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
    "CohortV2EndpointScore",
    "LabelAvailability",
    "build_cohort_v2_oracle_window_loader",
    "score_cohort_v2_endpoints",
]
