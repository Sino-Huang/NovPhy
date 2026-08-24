"""Capability-gated exhaustive pair evaluation for the cohort-v2 release."""
from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from world_model.data.cohort_v2 import (
    CAPABILITY_DECLARATION_IDENTITY,
    CohortV2OracleWindow,
    CohortV2OracleWindowDataset,
    CohortV2ReleaseReader,
)
from world_model.model import ABSTRACTION_ORDER, Abstraction, PredictionPair, identity
from world_model.training.grid_artifacts import canonical_json_bytes


SCHEMA_VERSION: Final = "cohort_v2_exhaustive_pair_evaluation_v1"
COHORT_V2_HORIZONS: Final = (1, 5, 15)
COHORT_V2_PAIRS: Final = tuple(
    PredictionPair(horizon, abstraction)
    for horizon in COHORT_V2_HORIZONS
    for abstraction in ABSTRACTION_ORDER
)
NON_FINAL_ROLES: Final = ("training", "calibration", "model_selection")
TRANSITION_CAPABILITY: Final = {
    Abstraction.CONTINUOUS: "transition.continuous",
    Abstraction.MICRO: "transition.micro",
    Abstraction.MACRO: "transition.macro",
}
CONTEXT_LABEL_REQUIREMENTS: Final = {
    Abstraction.CONTINUOUS: (),
    Abstraction.MICRO: ("contact", "supports"),
    Abstraction.MACRO: ("steady-state", "structure-unstable"),
}
ENDPOINT_CAPABILITIES: Final = (
    "excess_penetration",
    "unsupported_stationary_or_floating_body",
)
TIE_REL_TOL: Final = 1e-6
TIE_ABS_TOL: Final = 1e-12


class CohortV2EvaluationError(ValueError):
    """The exhaustive evaluation or its persisted artifact is invalid."""


@dataclass(frozen=True, slots=True)
class CohortV2PairGrid:
    """The explicitly declared central horizon and description-mode grid."""

    @property
    def horizons(self) -> tuple[int, ...]:
        return COHORT_V2_HORIZONS

    @property
    def abstractions(self) -> tuple[Abstraction, ...]:
        return ABSTRACTION_ORDER

    @property
    def pairs(self) -> tuple[PredictionPair, ...]:
        return COHORT_V2_PAIRS

    @property
    def identity(self) -> str:
        return identity((
            "cohort-v2-pair-grid-v1",
            self.horizons,
            tuple(str(item) for item in self.abstractions),
            tuple(
                (
                    str(abstraction),
                    TRANSITION_CAPABILITY[abstraction],
                    CONTEXT_LABEL_REQUIREMENTS[abstraction],
                )
                for abstraction in self.abstractions
            ),
            ENDPOINT_CAPABILITIES,
        ))


class CohortV2PairObjectiveScorer(Protocol):
    checkpoint_identity: str
    objective_identity: str
    capabilities: frozenset[str]

    def objective(
        self, window: CohortV2OracleWindow, pair: PredictionPair
    ) -> float: ...


class CohortV2BatchedPairObjectiveScorer(CohortV2PairObjectiveScorer, Protocol):
    def objective_batch(
        self,
        windows: tuple[CohortV2OracleWindow, ...],
        pair: PredictionPair,
    ) -> tuple[float, ...]: ...


@dataclass(frozen=True, slots=True)
class CohortV2PairOutcome:
    pair: PredictionPair
    requested_horizon: int
    effective_horizon: int
    target_frame_record_identity: str
    objective: float | None
    unavailable_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.requested_horizon != self.pair.delta:
            raise CohortV2EvaluationError("requested horizon must equal pair horizon")
        if not 1 <= self.effective_horizon <= self.requested_horizon:
            raise CohortV2EvaluationError("effective horizon is outside its request")
        if not self.target_frame_record_identity:
            raise CohortV2EvaluationError("target frame-record identity is missing")
        if self.objective is None:
            if not self.unavailable_reasons:
                raise CohortV2EvaluationError(
                    "unavailable pair outcomes require explicit reasons"
                )
        elif (
            type(self.objective) not in (int, float)
            or not math.isfinite(float(self.objective))
            or self.objective < 0.0
            or self.unavailable_reasons
        ):
            raise CohortV2EvaluationError("available pair objective is malformed")

    @property
    def available(self) -> bool:
        return self.objective is not None


@dataclass(frozen=True, slots=True)
class CohortV2StateEvaluation:
    state_id: str
    exposure_role: str
    attempt_id: str
    scenario_lineage_identity: str
    context_position: int
    context_fixed_step: int
    frame_record_count: int
    outcomes: tuple[CohortV2PairOutcome, ...]
    selected_pair: PredictionPair | None
    tied_pairs: tuple[PredictionPair, ...]


@dataclass(frozen=True, slots=True)
class CohortV2EvaluationResult:
    release_identity: str
    capability_declaration_identity: str
    partition_identity: str
    checkpoint_identity: str
    checkpoint_capabilities: tuple[str, ...]
    objective_identity: str
    grid: CohortV2PairGrid
    state_set_identity: str
    states: tuple[CohortV2StateEvaluation, ...]

    @property
    def outcome_count(self) -> int:
        return sum(len(state.outcomes) for state in self.states)

    @property
    def available_count(self) -> int:
        return sum(outcome.available for state in self.states for outcome in state.outcomes)

    @property
    def unavailable_count(self) -> int:
        return self.outcome_count - self.available_count

    @property
    def records_identity(self) -> str:
        return _records_identity(_canonical_records(self.states))

    @property
    def identity(self) -> str:
        return _evaluation_identity(
            self.release_identity,
            self.capability_declaration_identity,
            self.partition_identity,
            self.checkpoint_identity,
            self.checkpoint_capabilities,
            self.objective_identity,
            self.grid.identity,
            self.state_set_identity,
            self.records_identity,
        )


def _availability_reasons(
    window: CohortV2OracleWindow,
    pair: PredictionPair,
    checkpoint_capabilities: frozenset[str],
) -> tuple[str, ...]:
    reasons = []
    transition_capability = TRANSITION_CAPABILITY[pair.abstraction]
    if transition_capability not in checkpoint_capabilities:
        reasons.append(f"checkpoint_capability_unavailable:{transition_capability}")
    for predicate in CONTEXT_LABEL_REQUIREMENTS[pair.abstraction]:
        availability = window.context.labels[predicate].get("availability")
        if availability != "available":
            reasons.append(
                f"context_capability_unavailable:{predicate}:{availability}"
            )
        target_availability = window.target.labels[predicate].get("availability")
        if target_availability != "available":
            reasons.append(
                f"target_capability_unavailable:{predicate}:{target_availability}"
            )
    return tuple(reasons)


def _select_best_pair(
    outcomes: tuple[CohortV2PairOutcome, ...]
) -> tuple[PredictionPair | None, tuple[PredictionPair, ...]]:
    available = tuple(outcome for outcome in outcomes if outcome.available)
    if not available:
        return None, ()
    minimum = min(float(outcome.objective) for outcome in available)
    tied = tuple(
        outcome
        for outcome in available
        if math.isclose(
            float(outcome.objective), minimum,
            rel_tol=TIE_REL_TOL, abs_tol=TIE_ABS_TOL,
        )
    )
    ordered = tuple(sorted(
        tied,
        key=lambda outcome: (
            outcome.pair.delta,
            ABSTRACTION_ORDER.index(outcome.pair.abstraction),
        ),
    ))
    return ordered[0].pair, tuple(item.pair for item in ordered)


def cohort_v2_evaluation_state_set_identity(
    release_identity: str,
    partition_identity: str,
    state_ids: tuple[str, ...],
) -> str:
    """Name the exact nonterminal cohort-v2 evaluation scope."""
    if any(
        type(value) is not str or not value.strip()
        for value in (release_identity, partition_identity)
    ):
        raise CohortV2EvaluationError("release and partition identities must be nonempty")
    if (
        type(state_ids) is not tuple
        or not state_ids
        or any(type(state_id) is not str or not state_id for state_id in state_ids)
        or len(set(state_ids)) != len(state_ids)
    ):
        raise CohortV2EvaluationError(
            "state ids must be unique nonempty declared identities"
        )
    return identity((
        "cohort-v2-evaluation-states-v1",
        release_identity,
        partition_identity,
        "all-nonterminal-contexts",
        NON_FINAL_ROLES,
        COHORT_V2_HORIZONS,
        "terminal-clamp-v1",
        tuple(sorted(state_ids)),
    ))


def _validate_reader_bindings(
    readers: tuple[CohortV2ReleaseReader, ...],
) -> tuple[str, str]:
    if type(readers) is not tuple or len(readers) != len(NON_FINAL_ROLES):
        raise CohortV2EvaluationError(
            "evaluation requires the three non-final cohort-v2 readers"
        )
    if any(not reader.rollouts for reader in readers):
        raise CohortV2EvaluationError("evaluation readers must contain rollouts")
    roles = tuple(reader.rollouts[0].exposure_role for reader in readers)
    if roles != NON_FINAL_ROLES:
        raise CohortV2EvaluationError(
            "readers must be ordered training, calibration, model_selection"
        )
    releases = {reader.release_identity for reader in readers}
    partitions = {reader.partition_identity for reader in readers}
    if len(releases) != 1:
        raise CohortV2EvaluationError("evaluation readers cross cohort releases")
    if len(partitions) != 1:
        raise CohortV2EvaluationError("evaluation readers cross release partitions")
    return releases.pop(), partitions.pop()


def _source_state_windows(
    readers: tuple[CohortV2ReleaseReader, ...],
    grid: CohortV2PairGrid,
) -> tuple[tuple[tuple[CohortV2OracleWindow, ...], int], ...]:
    states = []
    for reader in readers:
        dataset = CohortV2OracleWindowDataset(
            reader, requested_horizons=grid.horizons
        )
        by_state: dict[str, list[CohortV2OracleWindow]] = {}
        for window in dataset:
            by_state.setdefault(window.context.identity, []).append(window)
        rollouts = {item.attempt_id: item for item in reader.rollouts}
        for windows in by_state.values():
            if tuple(item.requested_horizon for item in windows) != grid.horizons:
                raise CohortV2EvaluationError(
                    "eligible state does not cover the declared horizon grid"
                )
            first = windows[0]
            states.append((tuple(windows), len(rollouts[first.attempt_id].frame_records)))
    role_order = {role: index for index, role in enumerate(NON_FINAL_ROLES)}
    ordered = tuple(sorted(
        states,
        key=lambda item: (
            role_order[item[0][0].exposure_role],
            item[0][0].attempt_id,
            item[0][0].context_position,
        ),
    ))
    state_ids = tuple(item[0][0].context.identity for item in ordered)
    if len(set(state_ids)) != len(state_ids):
        raise CohortV2EvaluationError("eligible state identities are not unique")
    return ordered


class CohortV2ExhaustiveEvaluator:
    def __init__(
        self,
        scorer: CohortV2PairObjectiveScorer,
        grid: CohortV2PairGrid = CohortV2PairGrid(),
    ) -> None:
        if (
            type(scorer.checkpoint_identity) is not str
            or not scorer.checkpoint_identity.strip()
            or type(scorer.objective_identity) is not str
            or not scorer.objective_identity.strip()
            or type(scorer.capabilities) is not frozenset
            or any(
                type(capability) is not str or not capability.strip()
                for capability in scorer.capabilities
            )
        ):
            raise CohortV2EvaluationError("scorer provenance or capabilities are malformed")
        self._scorer = scorer
        self._grid = grid

    def evaluate(
        self, readers: tuple[CohortV2ReleaseReader, ...]
    ) -> CohortV2EvaluationResult:
        release_identity, partition_identity = _validate_reader_bindings(readers)
        states = []
        for windows, frame_record_count in _source_state_windows(readers, self._grid):
            first = windows[0]
            window_by_horizon = {
                item.requested_horizon: item for item in windows
            }
            outcomes = []
            for pair in self._grid.pairs:
                window = window_by_horizon[pair.delta]
                reasons = _availability_reasons(
                    window, pair, self._scorer.capabilities
                )
                objective = None
                if not reasons:
                    value = self._scorer.objective(window, pair)
                    if (
                        type(value) not in (int, float)
                        or not math.isfinite(float(value))
                        or value < 0.0
                    ):
                        raise CohortV2EvaluationError(
                            "pair scorer returned an invalid objective"
                        )
                    objective = float(value)
                outcomes.append(CohortV2PairOutcome(
                    pair=pair,
                    requested_horizon=pair.delta,
                    effective_horizon=window.effective_horizon,
                    target_frame_record_identity=window.target.identity,
                    objective=objective,
                    unavailable_reasons=reasons,
                ))
            outcome_tuple = tuple(outcomes)
            selected_pair, tied_pairs = _select_best_pair(outcome_tuple)
            states.append(CohortV2StateEvaluation(
                state_id=first.context.identity,
                exposure_role=first.exposure_role,
                attempt_id=first.attempt_id,
                scenario_lineage_identity=first.scenario_lineage_identity,
                context_position=first.context_position,
                context_fixed_step=first.context.fixed_step,
                frame_record_count=frame_record_count,
                outcomes=outcome_tuple,
                selected_pair=selected_pair,
                tied_pairs=tied_pairs,
            ))
        ordered_states = tuple(states)
        state_set_identity = cohort_v2_evaluation_state_set_identity(
            release_identity,
            partition_identity,
            tuple(item.state_id for item in ordered_states),
        )
        return CohortV2EvaluationResult(
            release_identity=release_identity,
            capability_declaration_identity=CAPABILITY_DECLARATION_IDENTITY,
            partition_identity=partition_identity,
            checkpoint_identity=self._scorer.checkpoint_identity,
            checkpoint_capabilities=tuple(sorted(self._scorer.capabilities)),
            objective_identity=self._scorer.objective_identity,
            grid=self._grid,
            state_set_identity=state_set_identity,
            states=ordered_states,
        )


class CohortV2ParallelExhaustiveEvaluator:
    """Batch and deterministically shard exhaustive states across scorers."""

    def __init__(
        self,
        scorers: tuple[CohortV2BatchedPairObjectiveScorer, ...],
        *,
        batch_size: int,
        grid: CohortV2PairGrid = CohortV2PairGrid(),
    ) -> None:
        if type(scorers) is not tuple or not scorers:
            raise CohortV2EvaluationError("parallel evaluation requires scorers")
        if type(batch_size) is not int or batch_size <= 0:
            raise CohortV2EvaluationError("parallel batch size must be positive")
        first = scorers[0]
        binding = (
            first.checkpoint_identity,
            first.objective_identity,
            first.capabilities,
        )
        if (
            type(first.checkpoint_identity) is not str
            or not first.checkpoint_identity.strip()
            or type(first.objective_identity) is not str
            or not first.objective_identity.strip()
            or type(first.capabilities) is not frozenset
        ):
            raise CohortV2EvaluationError(
                "parallel scorer provenance or capabilities are malformed"
            )
        for scorer in scorers:
            if (
                (
                    scorer.checkpoint_identity,
                    scorer.objective_identity,
                    scorer.capabilities,
                )
                != binding
                or not callable(getattr(scorer, "objective_batch", None))
            ):
                raise CohortV2EvaluationError(
                    "parallel scorers must share one checkpoint and objective"
                )
        self._scorers = scorers
        self._batch_size = batch_size
        self._grid = grid

    def _evaluate_shard(
        self,
        scorer: CohortV2BatchedPairObjectiveScorer,
        indexed_states: tuple[
            tuple[int, tuple[tuple[CohortV2OracleWindow, ...], int]], ...
        ],
    ) -> tuple[tuple[int, CohortV2StateEvaluation], ...]:
        objectives: dict[tuple[int, int], float] = {}
        window_maps = {
            index: {window.requested_horizon: window for window in windows}
            for index, (windows, _frame_record_count) in indexed_states
        }
        for pair_index, pair in enumerate(self._grid.pairs):
            eligible = tuple(
                (index, window_maps[index][pair.delta])
                for index, (_windows, _frame_record_count) in indexed_states
                if not _availability_reasons(
                    window_maps[index][pair.delta], pair, scorer.capabilities
                )
            )
            for start in range(0, len(eligible), self._batch_size):
                batch = eligible[start:start + self._batch_size]
                values = scorer.objective_batch(
                    tuple(window for _index, window in batch), pair
                )
                if type(values) is not tuple or len(values) != len(batch):
                    raise CohortV2EvaluationError(
                        "batched pair scorer returned a partial batch"
                    )
                for (index, _window), value in zip(batch, values, strict=True):
                    if (
                        type(value) not in (int, float)
                        or not math.isfinite(float(value))
                        or value < 0.0
                    ):
                        raise CohortV2EvaluationError(
                            "batched pair scorer returned an invalid objective"
                        )
                    objectives[(index, pair_index)] = float(value)

        states = []
        for index, (windows, frame_record_count) in indexed_states:
            first = windows[0]
            outcomes = []
            for pair_index, pair in enumerate(self._grid.pairs):
                window = window_maps[index][pair.delta]
                reasons = _availability_reasons(
                    window, pair, scorer.capabilities
                )
                objective = None if reasons else objectives[(index, pair_index)]
                outcomes.append(CohortV2PairOutcome(
                    pair=pair,
                    requested_horizon=pair.delta,
                    effective_horizon=window.effective_horizon,
                    target_frame_record_identity=window.target.identity,
                    objective=objective,
                    unavailable_reasons=reasons,
                ))
            outcome_tuple = tuple(outcomes)
            selected_pair, tied_pairs = _select_best_pair(outcome_tuple)
            states.append((index, CohortV2StateEvaluation(
                state_id=first.context.identity,
                exposure_role=first.exposure_role,
                attempt_id=first.attempt_id,
                scenario_lineage_identity=first.scenario_lineage_identity,
                context_position=first.context_position,
                context_fixed_step=first.context.fixed_step,
                frame_record_count=frame_record_count,
                outcomes=outcome_tuple,
                selected_pair=selected_pair,
                tied_pairs=tied_pairs,
            )))
        return tuple(states)

    def evaluate(
        self, readers: tuple[CohortV2ReleaseReader, ...]
    ) -> CohortV2EvaluationResult:
        release_identity, partition_identity = _validate_reader_bindings(readers)
        source_states = _source_state_windows(readers, self._grid)
        indexed = tuple(enumerate(source_states))
        shards = tuple(
            indexed[worker_index::len(self._scorers)]
            for worker_index in range(len(self._scorers))
        )
        if len(self._scorers) == 1:
            completed = (self._evaluate_shard(self._scorers[0], shards[0]),)
        else:
            with ThreadPoolExecutor(max_workers=len(self._scorers)) as executor:
                futures = tuple(
                    executor.submit(self._evaluate_shard, scorer, shard)
                    for scorer, shard in zip(self._scorers, shards, strict=True)
                )
                completed = tuple(future.result() for future in futures)
        indexed_results = tuple(item for shard in completed for item in shard)
        ordered_states = tuple(
            state for _index, state in sorted(indexed_results, key=lambda item: item[0])
        )
        state_set_identity = cohort_v2_evaluation_state_set_identity(
            release_identity,
            partition_identity,
            tuple(state.state_id for state in ordered_states),
        )
        first = self._scorers[0]
        return CohortV2EvaluationResult(
            release_identity=release_identity,
            capability_declaration_identity=CAPABILITY_DECLARATION_IDENTITY,
            partition_identity=partition_identity,
            checkpoint_identity=first.checkpoint_identity,
            checkpoint_capabilities=tuple(sorted(first.capabilities)),
            objective_identity=first.objective_identity,
            grid=self._grid,
            state_set_identity=state_set_identity,
            states=ordered_states,
        )


@dataclass(frozen=True, slots=True)
class CohortV2EvaluationReceipt:
    evaluation_identity: str
    release_identity: str
    capability_declaration_identity: str
    partition_identity: str
    checkpoint_identity: str
    checkpoint_capabilities: tuple[str, ...]
    objective_identity: str
    grid_identity: str
    state_set_identity: str
    records_identity: str
    state_count: int
    outcome_count: int
    available_count: int
    unavailable_count: int


def _pair_payload(pair: PredictionPair) -> dict[str, str | int]:
    return {"abstraction": str(pair.abstraction), "requested_horizon": pair.delta}


def _state_payload(state: CohortV2StateEvaluation) -> dict[str, object]:
    return {
        "attempt_id": state.attempt_id,
        "context_fixed_step": state.context_fixed_step,
        "context_position": state.context_position,
        "exposure_role": state.exposure_role,
        "frame_record_count": state.frame_record_count,
        "outcomes": [
            {
                **_pair_payload(outcome.pair),
                "effective_horizon": outcome.effective_horizon,
                "objective": outcome.objective,
                "status": "available" if outcome.available else "unavailable",
                "target_frame_record_identity": outcome.target_frame_record_identity,
                "unavailable_reasons": list(outcome.unavailable_reasons),
            }
            for outcome in state.outcomes
        ],
        "record_type": "state_evaluation",
        "scenario_lineage_identity": state.scenario_lineage_identity,
        "schema": SCHEMA_VERSION,
        "selected_pair": (
            None if state.selected_pair is None else _pair_payload(state.selected_pair)
        ),
        "state_id": state.state_id,
        "tied_pairs": [_pair_payload(pair) for pair in state.tied_pairs],
    }


def _canonical_records(states: tuple[CohortV2StateEvaluation, ...]) -> bytes:
    return b"".join(canonical_json_bytes(_state_payload(state)) for state in states)


def _records_identity(records: bytes) -> str:
    return f"sha256:{hashlib.sha256(records).hexdigest()}"


def _evaluation_identity(
    release_identity: str,
    capability_declaration_identity: str,
    partition_identity: str,
    checkpoint_identity: str,
    checkpoint_capabilities: tuple[str, ...],
    objective_identity: str,
    grid_identity: str,
    state_set_identity: str,
    records_identity: str,
) -> str:
    return identity((
        "cohort-v2-exhaustive-pair-evaluation-v1",
        release_identity,
        capability_declaration_identity,
        partition_identity,
        checkpoint_identity,
        checkpoint_capabilities,
        objective_identity,
        grid_identity,
        state_set_identity,
        records_identity,
    ))


def _grid_capability_payload() -> dict[str, object]:
    return {
        "context_labels": {
            str(abstraction): list(CONTEXT_LABEL_REQUIREMENTS[abstraction])
            for abstraction in ABSTRACTION_ORDER
        },
        "endpoint_labels": list(ENDPOINT_CAPABILITIES),
        "transitions": {
            str(abstraction): TRANSITION_CAPABILITY[abstraction]
            for abstraction in ABSTRACTION_ORDER
        },
    }


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_cohort_v2_evaluation(
    root: Path,
    result: CohortV2EvaluationResult,
    *,
    readers: tuple[CohortV2ReleaseReader, ...],
) -> CohortV2EvaluationReceipt:
    records = _canonical_records(result.states)
    manifest = {
        "artifact_type": "exhaustive_pair_evaluation",
        "available_count": result.available_count,
        "capability_declaration_identity": result.capability_declaration_identity,
        "checkpoint_capabilities": list(result.checkpoint_capabilities),
        "checkpoint_identity": result.checkpoint_identity,
        "evaluation_identity": result.identity,
        "exposure_roles": list(NON_FINAL_ROLES),
        "grid_capabilities": _grid_capability_payload(),
        "grid_identity": result.grid.identity,
        "horizons": list(result.grid.horizons),
        "objective_identity": result.objective_identity,
        "outcome_count": result.outcome_count,
        "pairs": [_pair_payload(pair) for pair in result.grid.pairs],
        "partition_identity": result.partition_identity,
        "records": "state_evaluations.jsonl",
        "records_identity": result.records_identity,
        "release_identity": result.release_identity,
        "schema": SCHEMA_VERSION,
        "state_count": len(result.states),
        "state_ids": [state.state_id for state in result.states],
        "state_set_identity": result.state_set_identity,
        "unavailable_count": result.unavailable_count,
    }
    _atomic_write(Path(root) / "state_evaluations.jsonl", records)
    _atomic_write(Path(root) / "manifest.json", canonical_json_bytes(manifest))
    return validate_cohort_v2_evaluation(
        Path(root),
        readers=readers,
        checkpoint_identity=result.checkpoint_identity,
        checkpoint_capabilities=frozenset(result.checkpoint_capabilities),
        objective_identity=result.objective_identity,
    )


def validate_cohort_v2_evaluation(
    root: Path,
    *,
    readers: tuple[CohortV2ReleaseReader, ...],
    checkpoint_identity: str,
    checkpoint_capabilities: frozenset[str],
    objective_identity: str,
) -> CohortV2EvaluationReceipt:
    try:
        manifest_raw = (Path(root) / "manifest.json").read_bytes()
        manifest = json.loads(manifest_raw)
        records_raw = (Path(root) / manifest["records"]).read_bytes()
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise CohortV2EvaluationError(f"cannot load evaluation artifact: {error}") from error
    lines = records_raw.splitlines()
    if canonical_json_bytes(manifest) != manifest_raw or manifest.get("schema") != SCHEMA_VERSION:
        raise CohortV2EvaluationError("evaluation manifest is noncanonical or unsupported")
    required = {
        "artifact_type", "available_count", "capability_declaration_identity",
        "checkpoint_capabilities", "checkpoint_identity", "evaluation_identity",
        "exposure_roles", "grid_capabilities", "grid_identity", "horizons",
        "objective_identity", "outcome_count", "pairs", "partition_identity",
        "records", "records_identity", "release_identity", "schema", "state_count",
        "state_ids", "state_set_identity", "unavailable_count",
    }
    if (
        set(manifest) != required
        or manifest["artifact_type"] != "exhaustive_pair_evaluation"
        or manifest["records"] != "state_evaluations.jsonl"
    ):
        raise CohortV2EvaluationError("evaluation manifest schema is malformed")
    if manifest["records_identity"] != _records_identity(records_raw):
        raise CohortV2EvaluationError("canonical-record identity does not recompute")
    if manifest["pairs"] != [_pair_payload(pair) for pair in COHORT_V2_PAIRS]:
        raise CohortV2EvaluationError("evaluation manifest pair grid is stale")
    grid = CohortV2PairGrid()
    if (
        type(checkpoint_capabilities) is not frozenset
        or any(
            type(capability) is not str or not capability.strip()
            for capability in checkpoint_capabilities
        )
    ):
        raise CohortV2EvaluationError("declared checkpoint capabilities are malformed")
    declared_checkpoint_capabilities = tuple(sorted(checkpoint_capabilities))
    release_identity, partition_identity = _validate_reader_bindings(readers)
    source_states = _source_state_windows(readers, grid)
    source_state_ids = tuple(item[0][0].context.identity for item in source_states)
    source_state_set_identity = cohort_v2_evaluation_state_set_identity(
        release_identity, partition_identity, source_state_ids
    )
    if (
        manifest["horizons"] != list(grid.horizons)
        or manifest["grid_identity"] != grid.identity
        or manifest["grid_capabilities"] != _grid_capability_payload()
        or manifest["exposure_roles"] != list(NON_FINAL_ROLES)
        or manifest["capability_declaration_identity"]
        != CAPABILITY_DECLARATION_IDENTITY
        or manifest["release_identity"] != release_identity
        or manifest["partition_identity"] != partition_identity
        or manifest["state_set_identity"] != source_state_set_identity
        or manifest["checkpoint_identity"] != checkpoint_identity
        or manifest["checkpoint_capabilities"]
        != list(declared_checkpoint_capabilities)
        or manifest["objective_identity"] != objective_identity
        or any(
            type(manifest[field]) is not str or not manifest[field]
            for field in (
                "checkpoint_identity", "objective_identity", "partition_identity",
                "records_identity", "release_identity", "state_set_identity",
            )
        )
    ):
        raise CohortV2EvaluationError("evaluation provenance is stale or malformed")
    state_ids = set()
    ordered_state_ids = []
    available_count = unavailable_count = 0
    if len(lines) != len(source_states):
        raise CohortV2EvaluationError("state membership differs from the source release")
    for line, (windows, source_frame_record_count) in zip(
        lines, source_states, strict=True
    ):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise CohortV2EvaluationError("state evaluation JSON is malformed") from error
        record_fields = {
            "attempt_id", "context_fixed_step", "context_position", "exposure_role",
            "frame_record_count", "outcomes",
            "record_type", "scenario_lineage_identity", "schema", "selected_pair",
            "state_id", "tied_pairs",
        }
        if (
            canonical_json_bytes(record) != line + b"\n"
            or set(record) != record_fields
            or record.get("schema") != SCHEMA_VERSION
            or record.get("record_type") != "state_evaluation"
        ):
            raise CohortV2EvaluationError("state evaluation is noncanonical or unsupported")
        state_id = record.get("state_id")
        outcomes = record.get("outcomes")
        context_position = record.get("context_position")
        frame_record_count = record.get("frame_record_count")
        first = windows[0]
        if (
            type(state_id) is not str
            or not state_id
            or state_id in state_ids
            or type(outcomes) is not list
            or len(outcomes) != len(COHORT_V2_PAIRS)
            or type(context_position) is not int
            or type(frame_record_count) is not int
            or not 0 <= context_position < frame_record_count - 1
        ):
            raise CohortV2EvaluationError("state membership or pair coverage is malformed")
        expected_source = {
            "attempt_id": first.attempt_id,
            "context_fixed_step": first.context.fixed_step,
            "context_position": first.context_position,
            "exposure_role": first.exposure_role,
            "frame_record_count": source_frame_record_count,
            "scenario_lineage_identity": first.scenario_lineage_identity,
            "state_id": first.context.identity,
        }
        if any(record[field] != value for field, value in expected_source.items()):
            raise CohortV2EvaluationError(
                "state membership differs from the source release"
            )
        state_ids.add(state_id)
        ordered_state_ids.append(state_id)
        validated_outcomes = []
        window_by_horizon = {
            item.requested_horizon: item for item in windows
        }
        for outcome, pair in zip(outcomes, COHORT_V2_PAIRS, strict=True):
            outcome_fields = {
                "abstraction", "effective_horizon", "objective",
                "requested_horizon", "status", "target_frame_record_identity",
                "unavailable_reasons",
            }
            source_window = window_by_horizon[pair.delta]
            if (
                set(outcome) != outcome_fields
                or outcome.get("abstraction") != str(pair.abstraction)
                or outcome.get("requested_horizon") != pair.delta
                or type(outcome.get("effective_horizon")) is not int
                or outcome["effective_horizon"]
                != source_window.effective_horizon
                or outcome.get("target_frame_record_identity")
                != source_window.target.identity
            ):
                raise CohortV2EvaluationError(
                    "state pair target or horizon differs from the source release"
                )
            expected_reasons = _availability_reasons(
                source_window, pair, checkpoint_capabilities
            )
            if expected_reasons:
                if (
                    outcome.get("status") != "unavailable"
                    or outcome.get("objective") is not None
                    or outcome.get("unavailable_reasons") != list(expected_reasons)
                ):
                    raise CohortV2EvaluationError(
                        "unavailable pair reasons differ from source capabilities"
                    )
                unavailable_count += 1
            else:
                objective = outcome.get("objective")
                if (
                    outcome.get("status") != "available"
                    or type(objective) not in (int, float)
                    or not math.isfinite(float(objective))
                    or objective < 0.0
                    or outcome.get("unavailable_reasons") != []
                ):
                    raise CohortV2EvaluationError("available pair outcome is malformed")
                available_count += 1
            validated_outcomes.append(CohortV2PairOutcome(
                pair=pair,
                requested_horizon=pair.delta,
                effective_horizon=source_window.effective_horizon,
                target_frame_record_identity=source_window.target.identity,
                objective=outcome.get("objective"),
                unavailable_reasons=tuple(outcome.get("unavailable_reasons", ())),
            ))
        selected_pair, tied_pairs = _select_best_pair(tuple(validated_outcomes))
        if (
            record.get("selected_pair")
            != (None if selected_pair is None else _pair_payload(selected_pair))
            or record.get("tied_pairs")
            != [_pair_payload(pair) for pair in tied_pairs]
        ):
            raise CohortV2EvaluationError(
                "stored deterministic selection does not recompute"
            )
    if (
        len(lines) != manifest["state_count"]
        or available_count != manifest["available_count"]
        or unavailable_count != manifest["unavailable_count"]
        or available_count + unavailable_count != manifest["outcome_count"]
        or ordered_state_ids != manifest["state_ids"]
        or tuple(ordered_state_ids) != source_state_ids
    ):
        raise CohortV2EvaluationError("evaluation manifest counts do not recompute")
    expected_identity = _evaluation_identity(
        manifest["release_identity"],
        manifest["capability_declaration_identity"],
        manifest["partition_identity"],
        manifest["checkpoint_identity"],
        tuple(manifest["checkpoint_capabilities"]),
        manifest["objective_identity"],
        manifest["grid_identity"],
        manifest["state_set_identity"],
        manifest["records_identity"],
    )
    if manifest["evaluation_identity"] != expected_identity:
        raise CohortV2EvaluationError("evaluation identity does not recompute")
    return CohortV2EvaluationReceipt(
        evaluation_identity=manifest["evaluation_identity"],
        release_identity=manifest["release_identity"],
        capability_declaration_identity=manifest["capability_declaration_identity"],
        partition_identity=manifest["partition_identity"],
        checkpoint_identity=manifest["checkpoint_identity"],
        checkpoint_capabilities=tuple(manifest["checkpoint_capabilities"]),
        objective_identity=manifest["objective_identity"],
        grid_identity=manifest["grid_identity"],
        state_set_identity=manifest["state_set_identity"],
        records_identity=manifest["records_identity"],
        state_count=len(lines),
        outcome_count=available_count + unavailable_count,
        available_count=available_count,
        unavailable_count=unavailable_count,
    )


__all__ = [
    "COHORT_V2_HORIZONS",
    "COHORT_V2_PAIRS",
    "CohortV2EvaluationError",
    "CohortV2EvaluationReceipt",
    "CohortV2EvaluationResult",
    "CohortV2ExhaustiveEvaluator",
    "CohortV2BatchedPairObjectiveScorer",
    "CohortV2ParallelExhaustiveEvaluator",
    "CohortV2PairGrid",
    "CohortV2PairObjectiveScorer",
    "CohortV2PairOutcome",
    "CohortV2StateEvaluation",
    "cohort_v2_evaluation_state_set_identity",
    "validate_cohort_v2_evaluation",
    "write_cohort_v2_evaluation",
]
