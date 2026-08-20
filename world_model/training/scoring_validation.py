"""Bounded validation accumulators for exhaustive score shards."""
from __future__ import annotations

from dataclasses import dataclass

from world_model.model import Abstraction, PredictionPair
from world_model.training.grid_data import MotionRegime
from world_model.training.pair_grid import (
    SENSITIVITY_LAMBDAS,
    PairMetric,
    ScoreSpec,
    select_best_pair,
)
from world_model.training.scoring import Partition, ScoreArtifactError, ScoringExample
from world_model.training.scoring_metrics import (
    FixedPairCeiling,
    PairAggregate,
    TemporalOracleCeiling,
    percentile,
)


@dataclass(slots=True)
class _MetricAccumulator:
    """Mutable scalar summary for one partition/regime/delta group."""

    count: int = 0
    truncation_count: int = 0
    latent_sum: float = 0.0
    weighted_sum: float = 0.0
    compute_sum: float = 0.0
    primary_selection_count: int = 0
    sensitivity_selection_counts: list[int] | None = None
    errors: list[float] | None = None

    def __post_init__(self) -> None:
        self.sensitivity_selection_counts = [0] * len(SENSITIVITY_LAMBDAS)
        self.errors = []

    def add(self, metric: PairMetric, selected_delta: int, sensitivity: list[dict[str, int]]) -> None:
        if self.sensitivity_selection_counts is None or self.errors is None:
            raise ScoreArtifactError("uninitialized score accumulator")
        self.count += 1
        self.truncation_count += int(metric.effective_delta < metric.requested_delta)
        self.latent_sum += metric.latent_mse
        self.weighted_sum += metric.weighted_prediction_error
        self.compute_sum += metric.compute_cost
        self.primary_selection_count += int(selected_delta == metric.pair.delta)
        self.errors.append(metric.weighted_prediction_error)
        for index, lambda_cost in enumerate(SENSITIVITY_LAMBDAS):
            self.sensitivity_selection_counts[index] += sum(
                int(item["lambda_cost"] == lambda_cost and item["selected_delta"] == metric.pair.delta)
                for item in sensitivity
            )

    def finish(self, partition: str, regime: MotionRegime | None, delta: int) -> PairAggregate:
        if self.errors is None or self.sensitivity_selection_counts is None or not self.errors:
            raise ScoreArtifactError("empty score accumulator")
        values = tuple(self.errors)
        return PairAggregate(
            partition=partition,
            motion_regime=regime,
            delta=delta,
            count=self.count,
            truncation_count=self.truncation_count,
            truncation_rate=self.truncation_count / self.count,
            latent_mse_mean=self.latent_sum / self.count,
            weighted_error_mean=self.weighted_sum / self.count,
            weighted_error_p50=percentile(values, 0.5),
            weighted_error_p90=percentile(values, 0.9),
            compute_cost_mean=self.compute_sum / self.count,
            primary_selection_count=self.primary_selection_count,
            sensitivity_selection_counts=tuple(self.sensitivity_selection_counts),
        )


@dataclass(frozen=True, slots=True)
class StreamValidationResult:
    """Bounded validation output consumed by the public artifact validator."""

    state_count: int
    score_count: int
    metrics: tuple[PairAggregate, ...]
    ceiling: TemporalOracleCeiling


class ScoreShardStream:
    """Validate shard records without retaining every decoded label."""

    def __init__(
        self,
        spec: ScoreSpec,
        expected_state_set_identity: str,
        expected_state_ids: frozenset[str],
    ) -> None:
        self._spec = spec
        if type(expected_state_set_identity) is not str or not expected_state_set_identity.strip():
            raise ScoreArtifactError("expected state-set identity must be nonempty")
        if (
            type(expected_state_ids) is not frozenset
            or not expected_state_ids
            or any(type(state_id) is not str or not state_id for state_id in expected_state_ids)
        ):
            raise ScoreArtifactError("expected state membership must be nonempty")
        self._expected_state_set_identity = expected_state_set_identity
        self._expected_state_ids = expected_state_ids
        self._seen: set[str] = set()
        self._state_count = 0
        self._eval_count = 0
        self._partitions: set[Partition] = set()
        self._calibration_errors: list[float] = []
        self._accumulators: dict[tuple[str, MotionRegime | None, int], _MetricAccumulator] = {}
        self._fixed_sums = {delta: 0.0 for delta in (1, 5, 15)}
        self._oracle_sum = 0.0

    def _accumulate(self, partition: str, regime: MotionRegime, metric: PairMetric, label: dict[str, object]) -> None:
        selected_delta = label["selected_delta"]
        sensitivity = label["sensitivity"]
        if type(selected_delta) is not int or type(sensitivity) is not list:
            raise ScoreArtifactError("stored pair selection has invalid types")
        for key in ((partition, None, metric.pair.delta), (partition, regime, metric.pair.delta)):
            self._accumulators.setdefault(key, _MetricAccumulator()).add(metric, selected_delta, sensitivity)

    def add_record(self, record: dict[str, object], partition_path: str) -> None:
        required = {
            "context_position", "frame_count", "metrics", "motion_regime", "partition",
            "primary_objective", "record_type", "schema_version", "selected_delta",
            "sensitivity", "state_id",
        }
        if set(record) != required or record["record_type"] != "state_score" or record["schema_version"] != "exhaustive_pair_scores_v1":
            raise ScoreArtifactError("closed state-score schema violation")
        if record["partition"] != partition_path:
            raise ScoreArtifactError("state record is in the wrong partition shard")
        state_id = record["state_id"]
        regime = MotionRegime(record["motion_regime"])
        if type(state_id) is not str or state_id in self._seen:
            raise ScoreArtifactError("state identities are duplicated or malformed")
        self._seen.add(state_id)
        self._state_count += 1
        example = ScoringExample(
            state_id, Partition(partition_path), regime, record["frame_count"], record["context_position"]
        )
        self._partitions.add(example.partition)
        metrics = tuple(
            PairMetric(
                pair=PredictionPair(metric["delta"], Abstraction.CONTINUOUS),
                requested_delta=metric["requested_delta"], effective_delta=metric["effective_delta"],
                duration_weight=metric["duration_weight"], latent_mse=metric["latent_mse"],
                weighted_prediction_error=metric["weighted_error"], compute_cost=metric["compute_cost"],
            )
            for metric in record["metrics"]
        )
        if tuple(metric.pair.delta for metric in metrics) != (1, 5, 15):
            raise ScoreArtifactError("state shard omitted or reordered a pair")
        expected = select_best_pair(metrics, self._spec)
        if record["selected_delta"] != expected.selected_pair.delta or record["primary_objective"] != expected.primary_objective:
            raise ScoreArtifactError("stored pair selection does not recompute")
        expected_sensitivity = [
            {"lambda_cost": item.lambda_cost, "selected_delta": item.selected_pair.delta}
            for item in expected.sensitivity
        ]
        if record["sensitivity"] != expected_sensitivity:
            raise ScoreArtifactError("stored pair sensitivity does not recompute")
        horizon = example.frame_count - 1
        if example.context_position < 0 or example.context_position >= horizon:
            raise ScoreArtifactError("state context position is outside its shot")
        for metric in metrics:
            if metric.effective_delta != min(metric.requested_delta, horizon - example.context_position):
                raise ScoreArtifactError("terminal clamp metadata does not recompute")
        if example.partition is Partition.CALIBRATION:
            self._calibration_errors.extend(metric.weighted_prediction_error for metric in metrics)
            return
        for metric in metrics:
            self._accumulate(str(example.partition), regime, metric, record)
        if example.partition is Partition.EVALUATION:
            self._eval_count += 1
            for metric in metrics:
                self._fixed_sums[metric.pair.delta] += metric.weighted_prediction_error / self._spec.error_scale + metric.compute_cost
            self._oracle_sum += expected.primary_objective

    def finish(
        self,
        expected_state_count: int,
        expected_score_count: int,
        declared_state_set_identity: str,
    ) -> StreamValidationResult:
        if self._state_count != expected_state_count or self._state_count * 3 != expected_score_count:
            raise ScoreArtifactError("manifest state or score count mismatch")
        if declared_state_set_identity != self._expected_state_set_identity:
            raise ScoreArtifactError("manifest state-set identity mismatch")
        if self._seen != self._expected_state_ids:
            raise ScoreArtifactError("manifest state membership mismatch")
        if self._partitions != set(Partition):
            raise ScoreArtifactError("state identities or partitions are incomplete")
        calibration_spec = ScoreSpec.from_calibration(tuple(self._calibration_errors))
        if calibration_spec != self._spec:
            raise ScoreArtifactError("frozen error_scale does not match calibration")
        metrics = tuple(
            accumulator.finish(partition, regime, delta)
            for partition in (str(Partition.CONTROLLER_TRAIN), str(Partition.EVALUATION))
            for regime in (None, *tuple(MotionRegime))
            for delta in (1, 5, 15)
            if (accumulator := self._accumulators.get((partition, regime, delta))) is not None
        )
        fixed = tuple(
            FixedPairCeiling(delta, self._eval_count, self._fixed_sums[delta] / self._eval_count)
            for delta in (1, 5, 15)
        )
        ceiling = TemporalOracleCeiling(
            self._eval_count, self._oracle_sum / self._eval_count, fixed
        )
        return StreamValidationResult(self._state_count, expected_score_count, metrics, ceiling)


__all__ = ["ScoreShardStream", "StreamValidationResult"]
