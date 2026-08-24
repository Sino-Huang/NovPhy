"""Matched source-endpoint plausibility and rollout-compute pair measurements."""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from world_model.data import CohortV2OracleWindowDataset, CohortV2ReleaseReader
from world_model.model import ABSTRACTION_ORDER, Abstraction, PredictionPair, identity
from world_model.training.cohort_v2 import (
    CohortV2EndpointPlausibility,
    PHYSICAL_VIOLATION_ENDPOINT_QUANTITIES,
    score_cohort_v2_endpoint_plausibility,
)
from world_model.training.cohort_v2_evaluation import (
    COHORT_V2_HORIZONS,
    CohortV2EvaluationResult,
)
from world_model.training.grid_artifacts import canonical_json_bytes


MEASUREMENT_SCHEMA = "cohort_v2_pair_measurements_v1"


class CohortV2MeasurementError(ValueError):
    """The pair measurement or its declared compute calibration is invalid."""


def _validated_cost(value: float, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
        raise CohortV2MeasurementError(f"{field} must be a finite nonnegative cost")
    return float(value)


@dataclass(frozen=True, slots=True)
class CohortV2ComputeCalibration:
    """Declared costs for every rollout operation controlled by a pair policy."""

    authority: str
    unit: str
    controller_per_decision: float
    continuous_adapter_per_decision: float
    micro_adapter_per_decision: float
    macro_adapter_per_decision: float
    micro_graph_base_per_decision: float
    micro_graph_per_entity: float
    micro_graph_per_contact: float
    micro_graph_per_support: float
    transition_per_decision: float
    continuous_readout_per_decision: float
    micro_readout_per_decision: float
    macro_readout_per_decision: float
    shared_initial_perception_per_rollout: float

    def __post_init__(self) -> None:
        if any(type(value) is not str or not value.strip() for value in (self.authority, self.unit)):
            raise CohortV2MeasurementError("compute authority and unit must be nonempty")
        for field in (
            "controller_per_decision",
            "continuous_adapter_per_decision",
            "micro_adapter_per_decision",
            "macro_adapter_per_decision",
            "micro_graph_base_per_decision",
            "micro_graph_per_entity",
            "micro_graph_per_contact",
            "micro_graph_per_support",
            "transition_per_decision",
            "continuous_readout_per_decision",
            "micro_readout_per_decision",
            "macro_readout_per_decision",
            "shared_initial_perception_per_rollout",
        ):
            object.__setattr__(
                self, field, _validated_cost(getattr(self, field), field)
            )

    @property
    def identity(self) -> str:
        return identity((
            "cohort-v2-compute-calibration-v1",
            self.authority,
            self.unit,
            self.controller_per_decision,
            self.continuous_adapter_per_decision,
            self.micro_adapter_per_decision,
            self.macro_adapter_per_decision,
            self.micro_graph_base_per_decision,
            self.micro_graph_per_entity,
            self.micro_graph_per_contact,
            self.micro_graph_per_support,
            self.transition_per_decision,
            self.continuous_readout_per_decision,
            self.micro_readout_per_decision,
            self.macro_readout_per_decision,
            self.shared_initial_perception_per_rollout,
        ))


@dataclass(frozen=True, slots=True)
class CohortV2ExecutionProfile:
    """Operations executed around each available pair transition."""

    controller_executed: bool
    shared_perception_executed: bool

    def __post_init__(self) -> None:
        if type(self.controller_executed) is not bool or type(
            self.shared_perception_executed
        ) is not bool:
            raise CohortV2MeasurementError("execution declarations must be boolean")

    @property
    def identity(self) -> str:
        return identity((
            "cohort-v2-execution-profile-v1",
            *self.canonical.values(),
        ))

    @property
    def canonical(self) -> dict[str, bool]:
        return {
            "controller_executed": self.controller_executed,
            "infiller_executed": False,
            "shared_perception_executed": self.shared_perception_executed,
        }


@dataclass(frozen=True, slots=True)
class CohortV2ComputeBreakdown:
    unit: str
    controller: float
    active_adapter: float
    graph_work: float
    transition: float
    active_readout: float
    infilling: float
    shared_perception: float
    simulated_frame_count: int
    policy_dependent_total: float
    full_end_to_end_total: float
    policy_dependent_per_simulated_frame: float
    full_end_to_end_per_simulated_frame: float


@dataclass(frozen=True, slots=True)
class CohortV2PairMeasurement:
    pair: PredictionPair
    effective_horizon: int
    target_frame_record_identity: str
    endpoint_plausibility: CohortV2EndpointPlausibility | None
    compute: CohortV2ComputeBreakdown | None
    unavailable_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CohortV2StateMeasurement:
    state_id: str
    outcomes: tuple[CohortV2PairMeasurement, ...]


@dataclass(frozen=True, slots=True)
class CohortV2MeasurementResult:
    evaluation_identity: str
    compute_calibration_identity: str
    execution_profile_identity: str
    states: tuple[CohortV2StateMeasurement, ...]

    @property
    def records_identity(self) -> str:
        return _records_identity(_records(self))

    @property
    def identity(self) -> str:
        return identity((
            "cohort-v2-pair-measurements-v1",
            self.evaluation_identity,
            self.compute_calibration_identity,
            self.execution_profile_identity,
            self.records_identity,
        ))


@dataclass(frozen=True, slots=True)
class CohortV2MeasurementReceipt:
    measurement_identity: str
    evaluation_identity: str
    compute_calibration_identity: str
    execution_profile_identity: str
    records_identity: str
    state_count: int
    outcome_count: int
    available_count: int
    unavailable_count: int


def _source_windows(
    readers: tuple[CohortV2ReleaseReader, ...],
) -> dict[tuple[str, int], object]:
    windows = {}
    for reader in readers:
        for window in CohortV2OracleWindowDataset(
            reader, requested_horizons=COHORT_V2_HORIZONS
        ):
            key = (window.context.identity, window.requested_horizon)
            if key in windows:
                raise CohortV2MeasurementError("source evaluation windows are not unique")
            windows[key] = window
    return windows


def _relations(label: object, predicate: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(label, Mapping) or label.get("availability") != "available":
        raise CohortV2MeasurementError(f"available micro pair lacks {predicate}")
    relations = label.get("relations")
    if not isinstance(relations, tuple) or any(
        not isinstance(item, tuple)
        or len(item) != 2
        or any(type(entity) is not str or not entity for entity in item)
        for item in relations
    ):
        raise CohortV2MeasurementError(f"{predicate} relations are malformed")
    return relations


def _graph_cost(window: object, calibration: CohortV2ComputeCalibration) -> float:
    contact = _relations(window.context.labels["contact"], "contact")
    supports = _relations(window.context.labels["supports"], "supports")
    entities = {entity for relation in (*contact, *supports) for entity in relation}
    return (
        calibration.micro_graph_base_per_decision
        + len(entities) * calibration.micro_graph_per_entity
        + len(contact) * calibration.micro_graph_per_contact
        + len(supports) * calibration.micro_graph_per_support
    )


def _compute(
    pair: PredictionPair,
    window: object,
    calibration: CohortV2ComputeCalibration,
    profile: CohortV2ExecutionProfile,
) -> CohortV2ComputeBreakdown:
    adapters = {
        Abstraction.CONTINUOUS: calibration.continuous_adapter_per_decision,
        Abstraction.MICRO: calibration.micro_adapter_per_decision,
        Abstraction.MACRO: calibration.macro_adapter_per_decision,
    }
    readouts = {
        Abstraction.CONTINUOUS: calibration.continuous_readout_per_decision,
        Abstraction.MICRO: calibration.micro_readout_per_decision,
        Abstraction.MACRO: calibration.macro_readout_per_decision,
    }
    controller = calibration.controller_per_decision if profile.controller_executed else 0.0
    graph = _graph_cost(window, calibration) if pair.abstraction is Abstraction.MICRO else 0.0
    perception = (
        calibration.shared_initial_perception_per_rollout
        if profile.shared_perception_executed and window.context_position == 0
        else 0.0
    )
    policy_total = (
        controller
        + adapters[pair.abstraction]
        + graph
        + calibration.transition_per_decision
        + readouts[pair.abstraction]
    )
    full_total = policy_total + perception
    frames = window.effective_horizon
    return CohortV2ComputeBreakdown(
        unit=calibration.unit,
        controller=controller,
        active_adapter=adapters[pair.abstraction],
        graph_work=graph,
        transition=calibration.transition_per_decision,
        active_readout=readouts[pair.abstraction],
        infilling=0.0,
        shared_perception=perception,
        simulated_frame_count=frames,
        policy_dependent_total=policy_total,
        full_end_to_end_total=full_total,
        policy_dependent_per_simulated_frame=policy_total / frames,
        full_end_to_end_per_simulated_frame=full_total / frames,
    )


def measure_cohort_v2_evaluation(
    evaluation: CohortV2EvaluationResult,
    readers: tuple[CohortV2ReleaseReader, ...],
    calibration: CohortV2ComputeCalibration,
    profile: CohortV2ExecutionProfile,
) -> CohortV2MeasurementResult:
    """Measure every available pair against its exact declared source endpoint.

    The target identity travels with the measurement so cross-mode summaries can
    compare only identical endpoints. A predicted carrier is deliberately not
    treated as an engine frame record with derivation-backed violation labels.
    """
    if not isinstance(evaluation, CohortV2EvaluationResult):
        raise CohortV2MeasurementError("measurement requires an exhaustive evaluation")
    if not isinstance(calibration, CohortV2ComputeCalibration) or not isinstance(
        profile, CohortV2ExecutionProfile
    ):
        raise CohortV2MeasurementError("measurement declarations are malformed")
    if (
        evaluation.release_identity != readers[0].release_identity
        or evaluation.partition_identity != readers[0].partition_identity
        or any(
            reader.release_identity != evaluation.release_identity
            or reader.partition_identity != evaluation.partition_identity
            for reader in readers
        )
    ):
        raise CohortV2MeasurementError("measurement readers differ from the evaluation")
    windows = _source_windows(readers)
    states = []
    for state in evaluation.states:
        outcomes = []
        for outcome in state.outcomes:
            window = windows.get((state.state_id, outcome.requested_horizon))
            if (
                window is None
                or window.target.identity != outcome.target_frame_record_identity
                or window.effective_horizon != outcome.effective_horizon
            ):
                raise CohortV2MeasurementError("pair endpoint differs from its source window")
            if not outcome.available:
                outcomes.append(CohortV2PairMeasurement(
                    pair=outcome.pair,
                    effective_horizon=outcome.effective_horizon,
                    target_frame_record_identity=outcome.target_frame_record_identity,
                    endpoint_plausibility=None,
                    compute=None,
                    unavailable_reasons=outcome.unavailable_reasons,
                ))
                continue
            outcomes.append(CohortV2PairMeasurement(
                pair=outcome.pair,
                effective_horizon=outcome.effective_horizon,
                target_frame_record_identity=outcome.target_frame_record_identity,
                endpoint_plausibility=score_cohort_v2_endpoint_plausibility(
                    window.target
                ),
                compute=_compute(outcome.pair, window, calibration, profile),
                unavailable_reasons=(),
            ))
        states.append(CohortV2StateMeasurement(state.state_id, tuple(outcomes)))
    if (
        evaluation.grid.horizons != COHORT_V2_HORIZONS
        or evaluation.grid.abstractions != ABSTRACTION_ORDER
        or len(windows) != len(evaluation.states) * len(evaluation.grid.horizons)
    ):
        raise CohortV2MeasurementError("evaluation state membership differs from readers")
    return CohortV2MeasurementResult(
        evaluation_identity=evaluation.identity,
        compute_calibration_identity=calibration.identity,
        execution_profile_identity=profile.identity,
        states=tuple(states),
    )


def _compute_payload(compute: CohortV2ComputeBreakdown) -> dict[str, object]:
    return {
        "active_adapter": compute.active_adapter,
        "active_readout": compute.active_readout,
        "controller": compute.controller,
        "full_end_to_end_per_simulated_frame": compute.full_end_to_end_per_simulated_frame,
        "full_end_to_end_total": compute.full_end_to_end_total,
        "graph_work": compute.graph_work,
        "infilling": compute.infilling,
        "policy_dependent_per_simulated_frame": compute.policy_dependent_per_simulated_frame,
        "policy_dependent_total": compute.policy_dependent_total,
        "shared_perception": compute.shared_perception,
        "simulated_frame_count": compute.simulated_frame_count,
        "transition": compute.transition,
        "unit": compute.unit,
    }


def _plausibility_payload(
    plausibility: CohortV2EndpointPlausibility,
) -> dict[str, object]:
    return {
        "available_value_count": plausibility.available_value_count,
        "declared_quantities": list(PHYSICAL_VIOLATION_ENDPOINT_QUANTITIES),
        "measurement_kind": "source_endpoint_violation_incidence",
        "unavailable_value_count": plausibility.unavailable_value_count,
        "violation_count": plausibility.violation_count,
        "violation_rate": plausibility.violation_rate,
    }


def _outcome_payload(outcome: CohortV2PairMeasurement) -> dict[str, object]:
    available = outcome.compute is not None
    return {
        "abstraction": str(outcome.pair.abstraction),
        "compute": None if outcome.compute is None else _compute_payload(outcome.compute),
        "endpoint_plausibility": (
            None
            if outcome.endpoint_plausibility is None
            else _plausibility_payload(outcome.endpoint_plausibility)
        ),
        "effective_horizon": outcome.effective_horizon,
        "requested_horizon": outcome.pair.delta,
        "status": "available" if available else "unavailable",
        "target_frame_record_identity": outcome.target_frame_record_identity,
        "unavailable_reasons": list(outcome.unavailable_reasons),
    }


def _records(result: CohortV2MeasurementResult) -> bytes:
    return b"".join(
        canonical_json_bytes({
            "outcomes": [_outcome_payload(outcome) for outcome in state.outcomes],
            "record_type": "state_pair_measurements",
            "schema": MEASUREMENT_SCHEMA,
            "state_id": state.state_id,
        })
        for state in result.states
    )


def _records_identity(records: bytes) -> str:
    return f"sha256:{hashlib.sha256(records).hexdigest()}"


def _manifest(
    result: CohortV2MeasurementResult, records: bytes
) -> dict[str, object]:
    available = sum(
        outcome.compute is not None
        for state in result.states
        for outcome in state.outcomes
    )
    outcome_count = sum(len(state.outcomes) for state in result.states)
    records_identity = _records_identity(records)
    measurement_identity = identity((
        "cohort-v2-pair-measurements-v1",
        result.evaluation_identity,
        result.compute_calibration_identity,
        result.execution_profile_identity,
        records_identity,
    ))
    return {
        "artifact_type": "cohort_v2_pair_measurements",
        "available_count": available,
        "compute_calibration_identity": result.compute_calibration_identity,
        "evaluation_identity": result.evaluation_identity,
        "execution_profile_identity": result.execution_profile_identity,
        "measurement_identity": measurement_identity,
        "outcome_count": outcome_count,
        "records": "pair_measurements.jsonl",
        "records_identity": records_identity,
        "schema": MEASUREMENT_SCHEMA,
        "state_count": len(result.states),
        "unavailable_count": outcome_count - available,
    }


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _receipt(manifest: Mapping[str, object]) -> CohortV2MeasurementReceipt:
    return CohortV2MeasurementReceipt(
        measurement_identity=manifest["measurement_identity"],
        evaluation_identity=manifest["evaluation_identity"],
        compute_calibration_identity=manifest["compute_calibration_identity"],
        execution_profile_identity=manifest["execution_profile_identity"],
        records_identity=manifest["records_identity"],
        state_count=manifest["state_count"],
        outcome_count=manifest["outcome_count"],
        available_count=manifest["available_count"],
        unavailable_count=manifest["unavailable_count"],
    )


def validate_cohort_v2_measurements(
    root: Path,
    evaluation: CohortV2EvaluationResult,
    *,
    readers: tuple[CohortV2ReleaseReader, ...],
    calibration: CohortV2ComputeCalibration,
    profile: CohortV2ExecutionProfile,
) -> CohortV2MeasurementReceipt:
    """Recompute pair measurements from their evaluation and v5 source readers."""
    try:
        manifest_raw = (Path(root) / "manifest.json").read_bytes()
        manifest = json.loads(manifest_raw)
        records_raw = (Path(root) / "pair_measurements.jsonl").read_bytes()
    except (OSError, json.JSONDecodeError) as error:
        raise CohortV2MeasurementError(
            f"cannot load pair measurements: {error}"
        ) from error
    expected_result = measure_cohort_v2_evaluation(
        evaluation, readers, calibration, profile
    )
    expected_records = _records(expected_result)
    expected_manifest = _manifest(expected_result, expected_records)
    if (
        canonical_json_bytes(manifest) != manifest_raw
        or manifest != expected_manifest
        or records_raw != expected_records
    ):
        raise CohortV2MeasurementError(
            "pair measurements differ from their evaluation or source readers"
        )
    return _receipt(manifest)


def write_cohort_v2_measurements(
    root: Path,
    evaluation: CohortV2EvaluationResult,
    *,
    readers: tuple[CohortV2ReleaseReader, ...],
    calibration: CohortV2ComputeCalibration,
    profile: CohortV2ExecutionProfile,
) -> CohortV2MeasurementReceipt:
    """Write deterministic measurements and validate them against their sources."""
    result = measure_cohort_v2_evaluation(evaluation, readers, calibration, profile)
    records = _records(result)
    manifest = _manifest(result, records)
    root = Path(root)
    _atomic_write(root / "pair_measurements.jsonl", records)
    _atomic_write(root / "manifest.json", canonical_json_bytes(manifest))
    return validate_cohort_v2_measurements(
        root,
        evaluation,
        readers=readers,
        calibration=calibration,
        profile=profile,
    )


__all__ = [
    "CohortV2ComputeBreakdown",
    "CohortV2ComputeCalibration",
    "CohortV2ExecutionProfile",
    "CohortV2MeasurementError",
    "CohortV2MeasurementReceipt",
    "CohortV2MeasurementResult",
    "CohortV2PairMeasurement",
    "CohortV2StateMeasurement",
    "measure_cohort_v2_evaluation",
    "validate_cohort_v2_measurements",
    "write_cohort_v2_measurements",
]
