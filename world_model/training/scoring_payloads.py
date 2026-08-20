"""Canonical JSON payloads for exhaustive scoring artifacts."""
from __future__ import annotations

from world_model.training.grid_artifacts import JsonValue
from world_model.training.pair_grid import SENSITIVITY_LAMBDAS, PairMetric


def metric_payload(metric: PairMetric) -> dict[str, str | int | float]:
    return {
        "alpha": "continuous",
        "compute_cost": metric.compute_cost,
        "delta": metric.pair.delta,
        "duration_weight": metric.duration_weight,
        "effective_delta": metric.effective_delta,
        "latent_mse": metric.latent_mse,
        "requested_delta": metric.requested_delta,
        "weighted_error": metric.weighted_prediction_error,
    }


def state_payload(scored) -> dict[str, JsonValue]:
    return {
        "context_position": scored.example.context_position,
        "frame_count": scored.example.frame_count,
        "metrics": [metric_payload(metric) for metric in scored.label.metrics],
        "motion_regime": str(scored.example.motion_regime),
        "partition": str(scored.example.partition),
        "primary_objective": scored.label.primary_objective,
        "record_type": "state_score",
        "schema_version": "exhaustive_pair_scores_v1",
        "selected_delta": scored.label.selected_pair.delta,
        "sensitivity": [
            {"lambda_cost": item.lambda_cost, "selected_delta": item.selected_pair.delta}
            for item in scored.label.sensitivity
        ],
        "state_id": scored.example.state_id,
    }


def aggregate_payload(metric) -> dict[str, JsonValue]:
    return {
        "compute_cost_mean": metric.compute_cost_mean,
        "count": metric.count,
        "delta": metric.delta,
        "latent_mse_mean": metric.latent_mse_mean,
        "motion_regime": None if metric.motion_regime is None else str(metric.motion_regime),
        "partition": str(metric.partition),
        "primary_selection_count": metric.primary_selection_count,
        "sensitivity_lambdas": list(SENSITIVITY_LAMBDAS),
        "sensitivity_selection_counts": list(metric.sensitivity_selection_counts),
        "truncation_count": metric.truncation_count,
        "truncation_rate": metric.truncation_rate,
        "weighted_error_mean": metric.weighted_error_mean,
        "weighted_error_p50": metric.weighted_error_p50,
        "weighted_error_p90": metric.weighted_error_p90,
    }


def ceiling_payload(ceiling) -> dict[str, JsonValue]:
    return {
        "fixed_pairs": [
            {
                "delta": item.delta,
                "primary_mean": item.primary_mean,
                "state_count": item.state_count,
            }
            for item in ceiling.fixed_pairs
        ],
        "oracle_definition": "per_state_primary_argmin",
        "oracle_primary_mean": ceiling.oracle_primary_mean,
        "oracle_symbol_called": False,
        "state_count": ceiling.state_count,
    }


__all__ = ["aggregate_payload", "ceiling_payload", "metric_payload", "state_payload"]
