"""Frozen temporal pair-grid and scoring contracts."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Final, Self, assert_never

from world_model.model import ABSTRACTION_ORDER, Abstraction, PredictionPair, digest

ERROR_SCALE_FLOOR: Final = 1e-12
TIE_REL_TOL: Final = 1e-6
TIE_ABS_TOL: Final = 1e-12
SENSITIVITY_LAMBDAS: Final = (0.0, 0.25, 1.0, 4.0)
APPROVED_PAIRS: Final = (
    PredictionPair(1, Abstraction.CONTINUOUS),
    PredictionPair(5, Abstraction.CONTINUOUS),
    PredictionPair(15, Abstraction.CONTINUOUS),
)


@unique
class UnavailabilityReason(StrEnum):
    SYMBOLIC_SUPERVISION_UNAVAILABLE = "symbolic_supervision_unavailable"
    PREDICTION_UNAVAILABLE = "prediction_unavailable"


@dataclass(frozen=True, slots=True)
class PairGridContractError(ValueError):
    field: str
    reason: str

    def __str__(self) -> str:
        return f"invalid {self.field}: {self.reason}"


@dataclass(frozen=True, slots=True)
class NoValidPairError(PairGridContractError):
    field: str = "pair candidates"
    reason: str = "no valid pair metrics"


@dataclass(frozen=True, slots=True)
class ExcludedAbstraction:
    abstraction: Abstraction
    reason: UnavailabilityReason


APPROVED_EXCLUSIONS: Final = (
    ExcludedAbstraction(
        Abstraction.MICRO,
        UnavailabilityReason.SYMBOLIC_SUPERVISION_UNAVAILABLE,
    ),
    ExcludedAbstraction(
        Abstraction.MACRO,
        UnavailabilityReason.SYMBOLIC_SUPERVISION_UNAVAILABLE,
    ),
)


@dataclass(frozen=True, slots=True)
class PairGridConfig:
    pairs: tuple[PredictionPair, ...] = APPROVED_PAIRS
    exclusions: tuple[ExcludedAbstraction, ...] = APPROVED_EXCLUSIONS

    def __post_init__(self) -> None:
        if type(self.pairs) is not tuple or not self.pairs:
            raise PairGridContractError("pairs", "must be a nonempty immutable tuple")
        if len(set(self.pairs)) != len(self.pairs):
            raise PairGridContractError("pairs", "must not contain duplicates")
        if self.pairs != APPROVED_PAIRS:
            raise PairGridContractError("pairs", "must equal the approved temporal grid")
        if self.exclusions != APPROVED_EXCLUSIONS:
            raise PairGridContractError(
                "exclusions", "must declare micro and macro symbolic supervision unavailable"
            )

    @property
    def identity(self) -> str:
        return digest(
            (
                "pair-grid-config-v1",
                tuple(pair.identity for pair in self.pairs),
                tuple(
                    (str(exclusion.abstraction), str(exclusion.reason))
                    for exclusion in self.exclusions
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class PairMetric:
    pair: PredictionPair
    requested_delta: int
    effective_delta: int
    duration_weight: float
    latent_mse: float
    weighted_prediction_error: float
    compute_cost: float

    def __post_init__(self) -> None:
        if self.pair not in APPROVED_PAIRS:
            raise PairGridContractError("pair metric", "pair is not in the approved grid")
        if self.requested_delta != self.pair.delta:
            raise PairGridContractError("requested_delta", "must equal pair.delta")
        if type(self.effective_delta) is not int or not 1 <= self.effective_delta <= self.requested_delta:
            raise PairGridContractError(
                "effective_delta", "must be an integer in [1, requested_delta]"
            )
        finite_values = (
            self.duration_weight,
            self.latent_mse,
            self.weighted_prediction_error,
            self.compute_cost,
        )
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in finite_values):
            raise PairGridContractError("pair metric", "all scalar values must be finite")
        if not 0.0 < self.duration_weight <= 1.0:
            raise PairGridContractError("duration_weight", "must lie in (0, 1]")
        if self.latent_mse < 0.0:
            raise PairGridContractError("latent_mse", "must be nonnegative")
        expected_error = self.duration_weight * self.latent_mse
        if not math.isclose(
            self.weighted_prediction_error,
            expected_error,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise PairGridContractError(
                "weighted_prediction_error", "must equal duration_weight * latent_mse"
            )
        if not math.isclose(
            self.compute_cost,
            1.0 / self.effective_delta,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise PairGridContractError("compute_cost", "must equal 1 / effective_delta")


@dataclass(frozen=True, slots=True)
class UnavailablePairMetric:
    pair: PredictionPair
    reason: UnavailabilityReason | str

    def __post_init__(self) -> None:
        if self.pair not in APPROVED_PAIRS:
            raise PairGridContractError("unavailable pair", "pair is not in the approved grid")
        try:
            reason = UnavailabilityReason(self.reason)
        except ValueError as error:
            raise PairGridContractError(
                "unavailable pair reason", f"unsupported value {self.reason!r}"
            ) from error
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True, slots=True)
class ScoreSpec:
    error_scale: float
    lambda_cost: tuple[float, ...] = SENSITIVITY_LAMBDAS

    def __post_init__(self) -> None:
        if type(self.error_scale) not in (int, float) or not math.isfinite(self.error_scale):
            raise PairGridContractError("error_scale", "must be finite")
        if self.error_scale < ERROR_SCALE_FLOOR:
            raise PairGridContractError("error_scale", f"must be at least {ERROR_SCALE_FLOOR}")
        if self.lambda_cost != SENSITIVITY_LAMBDAS:
            raise PairGridContractError("lambda_cost", "must equal the approved sensitivity set")

    @classmethod
    def from_calibration(cls, weighted_errors: tuple[float, ...]) -> Self:
        if type(weighted_errors) is not tuple or not weighted_errors:
            raise PairGridContractError(
                "calibration weighted errors", "must be a nonempty immutable tuple"
            )
        if any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0.0
            for value in weighted_errors
        ):
            raise PairGridContractError(
                "calibration weighted errors", "must contain finite nonnegative values"
            )
        ordered = sorted(float(value) for value in weighted_errors)
        rank = 0.9 * (len(ordered) - 1)
        lower = math.floor(rank)
        upper = math.ceil(rank)
        p90 = ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)
        return cls(error_scale=max(p90, ERROR_SCALE_FLOOR))

    @property
    def identity(self) -> str:
        return digest(("pair-score-spec-v1", self.error_scale, self.lambda_cost))


@dataclass(frozen=True, slots=True)
class PairSelection:
    lambda_cost: float
    selected_pair: PredictionPair
    tied_pairs: tuple[PredictionPair, ...]
    objective: float

    def __post_init__(self) -> None:
        if self.lambda_cost not in SENSITIVITY_LAMBDAS:
            raise PairGridContractError("selection lambda_cost", "must be approved")
        if type(self.tied_pairs) is not tuple or self.selected_pair not in self.tied_pairs:
            raise PairGridContractError("selection ties", "must contain selected_pair")
        if len(set(self.tied_pairs)) != len(self.tied_pairs):
            raise PairGridContractError("selection ties", "must not contain duplicates")
        if type(self.objective) not in (int, float) or not math.isfinite(self.objective):
            raise PairGridContractError("selection objective", "must be finite")


@dataclass(frozen=True, slots=True)
class BestPairLabel:
    metrics: tuple[PairMetric, ...]
    selected_pair: PredictionPair
    tied_pairs: tuple[PredictionPair, ...]
    primary_objective: float
    sensitivity: tuple[PairSelection, ...]

    def __post_init__(self) -> None:
        if type(self.metrics) is not tuple or not self.metrics:
            raise PairGridContractError("label metrics", "must be a nonempty immutable tuple")
        metric_pairs = tuple(metric.pair for metric in self.metrics)
        if len(set(metric_pairs)) != len(metric_pairs):
            raise PairGridContractError("label metrics", "must not contain duplicate pairs")
        if self.selected_pair not in metric_pairs:
            raise PairGridContractError("selected_pair", "must have a valid metric")
        if type(self.tied_pairs) is not tuple or self.selected_pair not in self.tied_pairs:
            raise PairGridContractError("label ties", "must contain selected_pair")
        if type(self.primary_objective) not in (int, float) or not math.isfinite(
            self.primary_objective
        ):
            raise PairGridContractError("primary objective", "must be finite")
        if tuple(item.lambda_cost for item in self.sensitivity) != SENSITIVITY_LAMBDAS:
            raise PairGridContractError("label sensitivity", "must cover approved lambdas")


def temporal_extent(frame_count: int) -> int:
    if type(frame_count) is not int or frame_count < 2:
        raise PairGridContractError("frame_count", "must be an integer of at least two")
    return frame_count - 1


def build_pair_metric(
    pair: PredictionPair | None,
    frame_count: int,
    t: int,
    latent_mse: float,
) -> PairMetric:
    horizon = temporal_extent(frame_count)
    match pair:
        case PredictionPair() as validated_pair:
            pass
        case None:
            raise PairGridContractError("pair", "must be a PredictionPair")
        case unreachable:
            assert_never(unreachable)
    if type(t) is not int or not 0 <= t < horizon:
        raise PairGridContractError("t", "must be an integer in [0, T-1]")
    if type(latent_mse) not in (int, float) or not math.isfinite(latent_mse):
        raise PairGridContractError("latent_mse", "must be finite")
    if latent_mse < 0.0:
        raise PairGridContractError("latent_mse", "must be nonnegative")
    effective_delta = min(validated_pair.delta, horizon - t)
    duration_weight = effective_delta / horizon
    return PairMetric(
        pair=validated_pair,
        requested_delta=validated_pair.delta,
        effective_delta=effective_delta,
        duration_weight=duration_weight,
        latent_mse=float(latent_mse),
        weighted_prediction_error=duration_weight * latent_mse,
        compute_cost=1.0 / effective_delta,
    )


def _select(metrics: tuple[PairMetric, ...], spec: ScoreSpec, lambda_cost: float) -> PairSelection:
    scored = tuple(
        (
            metric.weighted_prediction_error / spec.error_scale
            + lambda_cost * metric.compute_cost,
            metric,
        )
        for metric in metrics
    )
    minimum = min(objective for objective, _metric in scored)
    tied_metrics = tuple(
        metric
        for objective, metric in scored
        if math.isclose(objective, minimum, rel_tol=TIE_REL_TOL, abs_tol=TIE_ABS_TOL)
    )
    ordered_ties = tuple(
        sorted(
            tied_metrics,
            key=lambda metric: (
                metric.weighted_prediction_error,
                metric.pair.delta,
                ABSTRACTION_ORDER.index(metric.pair.abstraction),
            ),
        )
    )
    selected = ordered_ties[0]
    objective = (
        selected.weighted_prediction_error / spec.error_scale
        + lambda_cost * selected.compute_cost
    )
    return PairSelection(
        lambda_cost=lambda_cost,
        selected_pair=selected.pair,
        tied_pairs=tuple(metric.pair for metric in ordered_ties),
        objective=objective,
    )


def select_best_pair(
    candidates: tuple[PairMetric | UnavailablePairMetric | None, ...],
    spec: ScoreSpec,
) -> BestPairLabel:
    if type(candidates) is not tuple or not candidates:
        raise NoValidPairError()
    seen: set[PredictionPair] = set()
    metrics: list[PairMetric] = []
    for candidate in candidates:
        match candidate:
            case PairMetric() as metric:
                candidate_pair = metric.pair
                metrics.append(metric)
            case UnavailablePairMetric() as unavailable:
                candidate_pair = unavailable.pair
            case None:
                raise PairGridContractError("pair candidates", "must contain typed pair outcomes")
            case unreachable:
                assert_never(unreachable)
        if candidate_pair in seen:
            raise PairGridContractError("pair candidates", "must not contain duplicates")
        seen.add(candidate_pair)
    if not metrics:
        raise NoValidPairError()
    metric_tuple = tuple(metrics)
    primary = _select(metric_tuple, spec, 1.0)
    return BestPairLabel(
        metrics=metric_tuple,
        selected_pair=primary.selected_pair,
        tied_pairs=primary.tied_pairs,
        primary_objective=primary.objective,
        sensitivity=tuple(
            _select(metric_tuple, spec, lambda_cost)
            for lambda_cost in spec.lambda_cost
        ),
    )


__all__ = [
    "APPROVED_EXCLUSIONS",
    "APPROVED_PAIRS",
    "BestPairLabel",
    "ExcludedAbstraction",
    "NoValidPairError",
    "PairGridConfig",
    "PairGridContractError",
    "PairMetric",
    "PairSelection",
    "ScoreSpec",
    "UnavailablePairMetric",
    "UnavailabilityReason",
    "build_pair_metric",
    "select_best_pair",
    "temporal_extent",
]
