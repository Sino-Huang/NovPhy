"""Aggregate metrics and temporal oracle ceiling for exhaustive pair scores."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from world_model.training.grid_data import MotionRegime
from world_model.training.pair_grid import SENSITIVITY_LAMBDAS, BestPairLabel


@dataclass(frozen=True, slots=True)
class PairAggregate:
    partition: str
    motion_regime: MotionRegime | None
    delta: int
    count: int
    truncation_count: int
    truncation_rate: float
    latent_mse_mean: float
    weighted_error_mean: float
    weighted_error_p50: float
    weighted_error_p90: float
    compute_cost_mean: float
    primary_selection_count: int
    sensitivity_selection_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FixedPairCeiling:
    delta: int
    state_count: int
    state_digest: str
    primary_mean: float


@dataclass(frozen=True, slots=True)
class TemporalOracleCeiling:
    state_count: int
    state_digest: str
    oracle_primary_mean: float
    fixed_pairs: tuple[FixedPairCeiling, ...]


def percentile(values: tuple[float, ...], probability: float) -> float:
    ordered = sorted(values)
    rank = probability * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def state_digest(state_ids: tuple[str, ...]) -> str:
    payload = json.dumps(state_ids, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def aggregate_labels(
    labels: tuple[tuple[str, MotionRegime, str, BestPairLabel], ...],
) -> tuple[PairAggregate, ...]:
    aggregates: list[PairAggregate] = []
    partitions = ("controller-train", "evaluation")
    regimes: tuple[MotionRegime | None, ...] = (None, *tuple(MotionRegime))
    for partition in partitions:
        for regime in regimes:
            selected = tuple(
                label for row_partition, row_regime, _state_id, label in labels
                if row_partition == partition and (regime is None or row_regime is regime)
            )
            for delta in (1, 5, 15):
                metrics = tuple(
                    metric for label in selected for metric in label.metrics if metric.pair.delta == delta
                )
                if not metrics:
                    continue
                errors = tuple(metric.weighted_prediction_error for metric in metrics)
                truncation_count = sum(
                    metric.effective_delta < metric.requested_delta for metric in metrics
                )
                sensitivity = tuple(
                    sum(item.selected_pair.delta == delta for label in selected for item in label.sensitivity if item.lambda_cost == lambda_cost)
                    for lambda_cost in SENSITIVITY_LAMBDAS
                )
                aggregates.append(
                    PairAggregate(
                        partition=partition,
                        motion_regime=regime,
                        delta=delta,
                        count=len(metrics),
                        truncation_count=truncation_count,
                        truncation_rate=truncation_count / len(metrics),
                        latent_mse_mean=sum(metric.latent_mse for metric in metrics) / len(metrics),
                        weighted_error_mean=sum(errors) / len(errors),
                        weighted_error_p50=percentile(errors, 0.5),
                        weighted_error_p90=percentile(errors, 0.9),
                        compute_cost_mean=sum(metric.compute_cost for metric in metrics) / len(metrics),
                        primary_selection_count=sum(label.selected_pair.delta == delta for label in selected),
                        sensitivity_selection_counts=sensitivity,
                    )
                )
    return tuple(aggregates)


def oracle_ceiling(
    labels: tuple[tuple[str, MotionRegime, str, BestPairLabel], ...],
    error_scale: float,
) -> TemporalOracleCeiling:
    evaluation = tuple((state_id, label) for partition, _regime, state_id, label in labels if partition == "evaluation")
    ids = tuple(state_id for state_id, _label in evaluation)
    digest_value = state_digest(ids)
    fixed: list[FixedPairCeiling] = []
    for delta in (1, 5, 15):
        objectives = tuple(
            metric.weighted_prediction_error / error_scale + metric.compute_cost
            for _state_id, label in evaluation for metric in label.metrics if metric.pair.delta == delta
        )
        fixed.append(FixedPairCeiling(delta, len(evaluation), digest_value, sum(objectives) / len(objectives)))
    return TemporalOracleCeiling(
        state_count=len(evaluation),
        state_digest=digest_value,
        oracle_primary_mean=sum(label.primary_objective for _state_id, label in evaluation) / len(evaluation),
        fixed_pairs=tuple(fixed),
    )


__all__ = ["FixedPairCeiling", "PairAggregate", "TemporalOracleCeiling", "aggregate_labels", "oracle_ceiling", "percentile", "state_digest"]
