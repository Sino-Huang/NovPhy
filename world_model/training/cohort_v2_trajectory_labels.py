"""Duration-weighted trajectory-optimal controller labels for cohort v2."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from world_model.model import ABSTRACTION_ORDER, PredictionPair, identity
from world_model.training.cohort_v2_evaluation import (
    CohortV2EvaluationResult,
    CohortV2PairOutcome,
    CohortV2StateEvaluation,
)
from world_model.training.cohort_v2_measurement import (
    CohortV2MeasurementResult,
    CohortV2PairMeasurement,
    CohortV2StateMeasurement,
)
from world_model.training.grid_artifacts import canonical_json_bytes


TRAJECTORY_LABEL_SCHEMA = "cohort_v2_trajectory_controller_labels_v1"
TIE_REL_TOL = 1e-6
TIE_ABS_TOL = 1e-12


class CohortV2TrajectoryLabelError(ValueError):
    """The controller label inputs or resulting artifact are invalid."""


def _nonnegative(value: float, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
        raise CohortV2TrajectoryLabelError(
            f"{field} must be a finite nonnegative number"
        )
    return float(value)


@dataclass(frozen=True, slots=True)
class CohortV2TrajectoryCostSpec:
    """Declared weights for the additive trajectory controller objective.

    Exhaustive evaluator objectives are already duration-weighted endpoint-quality
    costs. Physical endpoint incidence receives the same effective-duration weight;
    policy-dependent compute is charged once per selected decision.
    """

    physical_violation_weight: float
    compute_weight: float
    compute_reference: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "physical_violation_weight",
            _nonnegative(
                self.physical_violation_weight, "physical violation weight"
            ),
        )
        object.__setattr__(
            self, "compute_weight", _nonnegative(self.compute_weight, "compute weight")
        )
        reference = _nonnegative(self.compute_reference, "compute reference")
        if reference == 0.0:
            raise CohortV2TrajectoryLabelError("compute reference must be positive")
        object.__setattr__(self, "compute_reference", reference)

    @property
    def identity(self) -> str:
        return identity((
            "cohort-v2-trajectory-cost-v1",
            "duration-weighted-evaluator-objective",
            "source-endpoint-violation-incidence",
            "policy-dependent-compute-per-decision",
            self.physical_violation_weight,
            self.compute_weight,
            self.compute_reference,
        ))


@dataclass(frozen=True, slots=True)
class CohortV2ControllerLabel:
    state_id: str
    exposure_role: str
    attempt_id: str
    scenario_lineage_identity: str
    context_position: int
    context_fixed_step: int
    selected_pair: PredictionPair
    effective_horizon: int
    next_context_position: int
    segment_cost: float
    cost_to_go: float
    tied_pairs: tuple[PredictionPair, ...]


@dataclass(frozen=True, slots=True)
class CohortV2ControllerLabelResult:
    teacher: str
    evaluation_identity: str
    measurement_identity: str
    cost_spec_identity: str
    labels: tuple[CohortV2ControllerLabel, ...]


@dataclass(frozen=True, slots=True)
class CohortV2TrajectoryLabelReceipt:
    label_artifact_identity: str
    teacher: str
    evaluation_identity: str
    measurement_identity: str
    cost_spec_identity: str
    records_identity: str
    label_count: int


@dataclass(frozen=True, slots=True)
class _Candidate:
    pair: PredictionPair
    effective_horizon: int
    next_context_position: int
    segment_cost: float


def _paired_states(
    evaluation: CohortV2EvaluationResult,
    measurement: CohortV2MeasurementResult,
) -> tuple[tuple[CohortV2StateEvaluation, CohortV2StateMeasurement], ...]:
    if not isinstance(evaluation, CohortV2EvaluationResult) or not isinstance(
        measurement, CohortV2MeasurementResult
    ):
        raise CohortV2TrajectoryLabelError(
            "controller labels require exhaustive evaluation and measurement results"
        )
    if measurement.evaluation_identity != evaluation.identity:
        raise CohortV2TrajectoryLabelError(
            "pair measurements do not belong to the exhaustive evaluation"
        )
    if len(evaluation.states) != len(measurement.states):
        raise CohortV2TrajectoryLabelError("evaluation and measurement states differ")
    paired = tuple(zip(evaluation.states, measurement.states, strict=True))
    if any(
        state.state_id != measured.state_id
        or len(state.outcomes) != len(measured.outcomes)
        for state, measured in paired
    ):
        raise CohortV2TrajectoryLabelError("evaluation and measurement states differ")
    return paired


def _candidate(
    state: CohortV2StateEvaluation,
    outcome: CohortV2PairOutcome,
    measured: CohortV2PairMeasurement,
    spec: CohortV2TrajectoryCostSpec,
) -> _Candidate | None:
    if (
        outcome.pair != measured.pair
        or outcome.effective_horizon != measured.effective_horizon
        or outcome.target_frame_record_identity
        != measured.target_frame_record_identity
    ):
        raise CohortV2TrajectoryLabelError(
            "pair measurement differs from its exhaustive outcome"
        )
    if not outcome.available:
        if measured.compute is not None or measured.endpoint_plausibility is not None:
            raise CohortV2TrajectoryLabelError(
                "unavailable pair outcome has fabricated measurements"
            )
        return None
    plausibility = measured.endpoint_plausibility
    compute = measured.compute
    if plausibility is None or compute is None or plausibility.violation_rate is None:
        return None
    duration = state.frame_record_count - 1
    duration_weight = outcome.effective_horizon / duration
    segment_cost = (
        float(outcome.objective)
        + duration_weight
        * spec.physical_violation_weight
        * plausibility.violation_rate
        + spec.compute_weight
        * compute.policy_dependent_total
        / spec.compute_reference
    )
    return _Candidate(
        pair=outcome.pair,
        effective_horizon=outcome.effective_horizon,
        next_context_position=state.context_position + outcome.effective_horizon,
        segment_cost=segment_cost,
    )


def _candidates(
    state: CohortV2StateEvaluation,
    measured: CohortV2StateMeasurement,
    spec: CohortV2TrajectoryCostSpec,
) -> tuple[_Candidate, ...]:
    available = []
    for outcome, pair_measurement in zip(
        state.outcomes, measured.outcomes, strict=True
    ):
        candidate = _candidate(state, outcome, pair_measurement, spec)
        if candidate is not None:
            available.append(candidate)
    return tuple(available)


def _ordered_ties(
    scored: tuple[tuple[float, _Candidate], ...]
) -> tuple[_Candidate, ...]:
    minimum = min(value for value, _candidate_ in scored)
    tied = tuple(
        candidate
        for value, candidate in scored
        if math.isclose(value, minimum, rel_tol=TIE_REL_TOL, abs_tol=TIE_ABS_TOL)
    )
    return tuple(sorted(
        tied,
        key=lambda candidate: (
            candidate.pair.delta,
            ABSTRACTION_ORDER.index(candidate.pair.abstraction),
        ),
    ))


def _trajectory_labels(
    states: tuple[tuple[CohortV2StateEvaluation, CohortV2StateMeasurement], ...],
    spec: CohortV2TrajectoryCostSpec,
    *,
    myopic: bool,
) -> tuple[CohortV2ControllerLabel, ...]:
    ordered = tuple(sorted(states, key=lambda item: item[0].context_position))
    first = ordered[0][0]
    terminal_position = first.frame_record_count - 1
    if (
        tuple(item[0].context_position for item in ordered)
        != tuple(range(terminal_position))
        or any(
            state.attempt_id != first.attempt_id
            or state.exposure_role != first.exposure_role
            or state.scenario_lineage_identity != first.scenario_lineage_identity
            or state.frame_record_count != first.frame_record_count
            for state, _measured in ordered
        )
    ):
        raise CohortV2TrajectoryLabelError(
            "controller label trajectory membership is incomplete"
        )
    by_position = {state.context_position: (state, measured) for state, measured in ordered}
    selected: dict[int, tuple[_Candidate, tuple[_Candidate, ...]]] = {}
    if myopic:
        for position, (state, measured) in by_position.items():
            candidates = _candidates(state, measured, spec)
            if not candidates:
                raise CohortV2TrajectoryLabelError(
                    f"state {state.state_id} has no admissible pair evidence"
                )
            ties = _ordered_ties(tuple((item.segment_cost, item) for item in candidates))
            selected[position] = (ties[0], ties)
        values = {terminal_position: 0.0}
        for position in reversed(range(terminal_position)):
            choice, _ties = selected[position]
            values[position] = choice.segment_cost + values[choice.next_context_position]
    else:
        values = {terminal_position: 0.0}
        for position in reversed(range(terminal_position)):
            state, measured = by_position[position]
            candidates = tuple(
                item
                for item in _candidates(state, measured, spec)
                if item.next_context_position in values
            )
            if not candidates:
                raise CohortV2TrajectoryLabelError(
                    f"state {state.state_id} has no admissible complete path"
                )
            scored = tuple(
                (item.segment_cost + values[item.next_context_position], item)
                for item in candidates
            )
            ties = _ordered_ties(scored)
            choice = ties[0]
            selected[position] = (choice, ties)
            values[position] = choice.segment_cost + values[choice.next_context_position]
    return tuple(
        CohortV2ControllerLabel(
            state_id=state.state_id,
            exposure_role=state.exposure_role,
            attempt_id=state.attempt_id,
            scenario_lineage_identity=state.scenario_lineage_identity,
            context_position=state.context_position,
            context_fixed_step=state.context_fixed_step,
            selected_pair=selected[state.context_position][0].pair,
            effective_horizon=selected[state.context_position][0].effective_horizon,
            next_context_position=selected[state.context_position][0].next_context_position,
            segment_cost=selected[state.context_position][0].segment_cost,
            cost_to_go=values[state.context_position],
            tied_pairs=tuple(item.pair for item in selected[state.context_position][1]),
        )
        for state, _measured in ordered
    )


def _generate(
    evaluation: CohortV2EvaluationResult,
    measurement: CohortV2MeasurementResult,
    spec: CohortV2TrajectoryCostSpec,
    *,
    myopic: bool,
) -> CohortV2ControllerLabelResult:
    if not isinstance(spec, CohortV2TrajectoryCostSpec):
        raise CohortV2TrajectoryLabelError("trajectory cost specification is malformed")
    paired = _paired_states(evaluation, measurement)
    groups: dict[
        tuple[str, str, str],
        list[tuple[CohortV2StateEvaluation, CohortV2StateMeasurement]],
    ] = {}
    for state, measured in paired:
        key = (state.exposure_role, state.attempt_id, state.scenario_lineage_identity)
        groups.setdefault(key, []).append((state, measured))
    labels_by_state = {
        label.state_id: label
        for states in groups.values()
        for label in _trajectory_labels(tuple(states), spec, myopic=myopic)
    }
    if len(labels_by_state) != len(evaluation.states):
        raise CohortV2TrajectoryLabelError("controller label state identities are not unique")
    return CohortV2ControllerLabelResult(
        teacher="myopic_ablation" if myopic else "trajectory_optimal",
        evaluation_identity=evaluation.identity,
        measurement_identity=measurement.identity,
        cost_spec_identity=spec.identity,
        labels=tuple(labels_by_state[state.state_id] for state in evaluation.states),
    )


def generate_cohort_v2_trajectory_labels(
    evaluation: CohortV2EvaluationResult,
    measurement: CohortV2MeasurementResult,
    spec: CohortV2TrajectoryCostSpec,
) -> CohortV2ControllerLabelResult:
    """Generate the default non-myopic controller teacher."""
    return _generate(evaluation, measurement, spec, myopic=False)


def generate_cohort_v2_myopic_ablation_labels(
    evaluation: CohortV2EvaluationResult,
    measurement: CohortV2MeasurementResult,
    spec: CohortV2TrajectoryCostSpec,
) -> CohortV2ControllerLabelResult:
    """Generate the explicit per-state myopic controller ablation."""
    return _generate(evaluation, measurement, spec, myopic=True)


def _pair_payload(pair: PredictionPair) -> dict[str, str | int]:
    return {"abstraction": str(pair.abstraction), "requested_horizon": pair.delta}


def _label_payload(label: CohortV2ControllerLabel) -> dict[str, object]:
    return {
        "attempt_id": label.attempt_id,
        "context_fixed_step": label.context_fixed_step,
        "context_position": label.context_position,
        "cost_to_go": label.cost_to_go,
        "effective_horizon": label.effective_horizon,
        "exposure_role": label.exposure_role,
        "next_context_position": label.next_context_position,
        "record_type": "controller_label",
        "scenario_lineage_identity": label.scenario_lineage_identity,
        "schema": TRAJECTORY_LABEL_SCHEMA,
        "segment_cost": label.segment_cost,
        "selected_pair": _pair_payload(label.selected_pair),
        "state_id": label.state_id,
        "tied_pairs": [_pair_payload(pair) for pair in label.tied_pairs],
    }


def _records(result: CohortV2ControllerLabelResult) -> bytes:
    return b"".join(canonical_json_bytes(_label_payload(label)) for label in result.labels)


def _records_identity(records: bytes) -> str:
    return f"sha256:{hashlib.sha256(records).hexdigest()}"


def _manifest(
    result: CohortV2ControllerLabelResult,
    evaluation: CohortV2EvaluationResult,
    measurement: CohortV2MeasurementResult,
    records: bytes,
) -> dict[str, object]:
    records_identity = _records_identity(records)
    artifact_identity = identity((
        "cohort-v2-trajectory-controller-labels-v1",
        result.teacher,
        result.evaluation_identity,
        result.measurement_identity,
        result.cost_spec_identity,
        records_identity,
    ))
    roles = list(dict.fromkeys(label.exposure_role for label in result.labels))
    lineages = list(
        dict.fromkeys(label.scenario_lineage_identity for label in result.labels)
    )
    return {
        "artifact_type": "cohort_v2_controller_labels",
        "capability_declaration_identity": evaluation.capability_declaration_identity,
        "checkpoint_capabilities": list(evaluation.checkpoint_capabilities),
        "checkpoint_identity": evaluation.checkpoint_identity,
        "compute_calibration_identity": measurement.compute_calibration_identity,
        "cost_spec_identity": result.cost_spec_identity,
        "evaluation_identity": result.evaluation_identity,
        "execution_profile_identity": measurement.execution_profile_identity,
        "exposure_roles": roles,
        "grid_identity": evaluation.grid.identity,
        "label_artifact_identity": artifact_identity,
        "label_count": len(result.labels),
        "measurement_identity": result.measurement_identity,
        "objective_identity": evaluation.objective_identity,
        "partition_identity": evaluation.partition_identity,
        "records": "controller_labels.jsonl",
        "records_identity": records_identity,
        "release_identity": evaluation.release_identity,
        "scenario_lineage_identities": lineages,
        "schema": TRAJECTORY_LABEL_SCHEMA,
        "state_set_identity": evaluation.state_set_identity,
        "teacher": result.teacher,
    }


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _receipt(manifest: dict[str, object]) -> CohortV2TrajectoryLabelReceipt:
    return CohortV2TrajectoryLabelReceipt(
        label_artifact_identity=manifest["label_artifact_identity"],
        teacher=manifest["teacher"],
        evaluation_identity=manifest["evaluation_identity"],
        measurement_identity=manifest["measurement_identity"],
        cost_spec_identity=manifest["cost_spec_identity"],
        records_identity=manifest["records_identity"],
        label_count=manifest["label_count"],
    )


def validate_cohort_v2_trajectory_labels(
    root: Path,
    evaluation: CohortV2EvaluationResult,
    measurement: CohortV2MeasurementResult,
    spec: CohortV2TrajectoryCostSpec,
) -> CohortV2TrajectoryLabelReceipt:
    """Recompute the default trajectory teacher and compare exact artifact bytes."""
    try:
        manifest_raw = (Path(root) / "manifest.json").read_bytes()
        manifest = json.loads(manifest_raw)
        records_raw = (Path(root) / "controller_labels.jsonl").read_bytes()
    except (OSError, json.JSONDecodeError) as error:
        raise CohortV2TrajectoryLabelError(
            f"cannot load trajectory labels: {error}"
        ) from error
    result = generate_cohort_v2_trajectory_labels(evaluation, measurement, spec)
    expected_records = _records(result)
    expected_manifest = _manifest(result, evaluation, measurement, expected_records)
    if (
        canonical_json_bytes(manifest) != manifest_raw
        or manifest != expected_manifest
        or records_raw != expected_records
    ):
        raise CohortV2TrajectoryLabelError(
            "trajectory labels differ from their exhaustive pair costs"
        )
    return _receipt(manifest)


def write_cohort_v2_trajectory_labels(
    root: Path,
    evaluation: CohortV2EvaluationResult,
    measurement: CohortV2MeasurementResult,
    spec: CohortV2TrajectoryCostSpec,
) -> CohortV2TrajectoryLabelReceipt:
    """Write and source-validate the default trajectory-optimal teacher."""
    result = generate_cohort_v2_trajectory_labels(evaluation, measurement, spec)
    records = _records(result)
    manifest = _manifest(result, evaluation, measurement, records)
    root = Path(root)
    _atomic_write(root / "controller_labels.jsonl", records)
    _atomic_write(root / "manifest.json", canonical_json_bytes(manifest))
    return validate_cohort_v2_trajectory_labels(
        root, evaluation, measurement, spec
    )


__all__ = [
    "CohortV2ControllerLabel",
    "CohortV2ControllerLabelResult",
    "CohortV2TrajectoryCostSpec",
    "CohortV2TrajectoryLabelError",
    "CohortV2TrajectoryLabelReceipt",
    "generate_cohort_v2_myopic_ablation_labels",
    "generate_cohort_v2_trajectory_labels",
    "validate_cohort_v2_trajectory_labels",
    "write_cohort_v2_trajectory_labels",
]
