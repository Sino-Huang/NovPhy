"""Exhaustive continuous temporal pair scoring."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Protocol

from world_model.model import Abstraction, PredictionPair, identity
from world_model.training.grid_data import MotionRegime, ScoringState
from world_model.training.pair_grid import BestPairLabel, PairMetric, ScoreSpec, build_pair_metric, select_best_pair
from world_model.training.scoring_metrics import PairAggregate, TemporalOracleCeiling, aggregate_labels, oracle_ceiling


@unique
class Partition(StrEnum):
    CONTROLLER_TRAIN = "controller-train"
    CALIBRATION = "calibration"
    EVALUATION = "evaluation"


class ScoreArtifactError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ScoringExample:
    state_id: str
    partition: Partition
    motion_regime: MotionRegime
    frame_count: int
    context_position: int

    @classmethod
    def from_grid_state(
        cls,
        state: ScoringState,
        partition: Partition,
        motion_regime: MotionRegime,
    ) -> ScoringExample:
        return cls(
            state_id=identity(("exhaustive-score-state-v1", state.key)),
            partition=partition,
            motion_regime=motion_regime,
            frame_count=state.shot_frame_count,
            context_position=state.context_position,
        )

    def __post_init__(self) -> None:
        if not self.state_id:
            raise ScoreArtifactError("state_id must be nonempty")
        if type(self.frame_count) is not int or self.frame_count < 2:
            raise ScoreArtifactError("frame_count must be at least two")
        if type(self.context_position) is not int or not 0 <= self.context_position < self.frame_count - 1:
            raise ScoreArtifactError("context_position must identify a nonterminal state")


class LatentMsePredictor(Protocol):
    def latent_mse(
        self,
        examples: tuple[ScoringExample, ...],
        requested_delta: int,
        effective_delta: int,
    ) -> tuple[float, ...]: ...


@dataclass(frozen=True, slots=True)
class ScoredState:
    example: ScoringExample
    label: BestPairLabel


@dataclass(frozen=True, slots=True)
class AdvancedMetricStatus:
    metric: str
    status: str = "unavailable"
    reason: str = "required supervision is unavailable"


UNAVAILABLE_METRICS = (
    "ade", "fde", "final_state", "event", "penetration", "floating", "illegal_contact"
)


@dataclass(frozen=True, slots=True)
class ExhaustiveScoreResult:
    score_spec: ScoreSpec
    scored_states: tuple[ScoredState, ...]
    labels: tuple[ScoredState, ...]
    per_pair_metrics: tuple[PairAggregate, ...]
    temporal_oracle_ceiling: TemporalOracleCeiling
    unavailable_metrics: tuple[AdvancedMetricStatus, ...]

    @property
    def score_count(self) -> int:
        return sum(len(item.label.metrics) for item in self.scored_states)


class ExhaustiveScorer:
    def __init__(self, predictor: LatentMsePredictor) -> None:
        self._predictor = predictor

    def score(self, examples: tuple[ScoringExample, ...]) -> ExhaustiveScoreResult:
        if type(examples) is not tuple or not examples:
            raise ScoreArtifactError("scoring requires nonempty immutable examples")
        if len({example.state_id for example in examples}) != len(examples):
            raise ScoreArtifactError("state identities must be unique")
        if {example.partition for example in examples} != set(Partition):
            raise ScoreArtifactError("all scoring partitions must be nonempty")
        raw: dict[str, list[PairMetric]] = {example.state_id: [] for example in examples}
        for requested_delta in (1, 5, 15):
            pair = PredictionPair(requested_delta, Abstraction.CONTINUOUS)
            grouped: dict[int, list[ScoringExample]] = {}
            for example in examples:
                effective = min(requested_delta, example.frame_count - 1 - example.context_position)
                grouped.setdefault(effective, []).append(example)
            for effective_delta in sorted(grouped):
                batch = tuple(grouped[effective_delta])
                losses = self._predictor.latent_mse(batch, requested_delta, effective_delta)
                if type(losses) is not tuple or len(losses) != len(batch):
                    raise ScoreArtifactError("predictor returned a partial batch")
                for example, loss in zip(batch, losses, strict=True):
                    if type(loss) not in (int, float) or not math.isfinite(float(loss)) or loss < 0:
                        raise ScoreArtifactError("predictor returned invalid latent MSE")
                    raw[example.state_id].append(
                        build_pair_metric(pair, example.frame_count, example.context_position, float(loss))
                    )
        calibration_errors = tuple(
            metric.weighted_prediction_error
            for example in examples if example.partition is Partition.CALIBRATION
            for metric in raw[example.state_id]
        )
        spec = ScoreSpec.from_calibration(calibration_errors)
        scored = tuple(
            ScoredState(example, select_best_pair(tuple(raw[example.state_id]), spec))
            for example in examples
        )
        labels = tuple(item for item in scored if item.example.partition is not Partition.CALIBRATION)
        label_rows = tuple(
            (str(item.example.partition), item.example.motion_regime, item.example.state_id, item.label)
            for item in labels
        )
        return ExhaustiveScoreResult(
            score_spec=spec,
            scored_states=scored,
            labels=labels,
            per_pair_metrics=aggregate_labels(label_rows),
            temporal_oracle_ceiling=oracle_ceiling(label_rows, spec.error_scale),
            unavailable_metrics=tuple(AdvancedMetricStatus(metric) for metric in UNAVAILABLE_METRICS),
        )


from world_model.training.scoring_artifacts import (
    ScoreArtifactReceipt,
    score_state_set_identity,
    validate_score_artifacts,
    write_score_artifacts,
)

__all__ = ["AdvancedMetricStatus", "ExhaustiveScoreResult", "ExhaustiveScorer", "Partition", "ScoreArtifactError", "ScoreArtifactReceipt", "ScoredState", "ScoringExample", "score_state_set_identity", "validate_score_artifacts", "write_score_artifacts"]
