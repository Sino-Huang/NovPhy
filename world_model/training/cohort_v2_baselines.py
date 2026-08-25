"""Comparable fixed and independent-axis policy baselines for cohort v2."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from world_model.model import ABSTRACTION_ORDER, Abstraction, PredictionPair, identity
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
from world_model.training.cohort_v2_trajectory_labels import (
    TIE_ABS_TOL,
    TIE_REL_TOL,
    CohortV2TrajectoryCostSpec,
)
from world_model.training.grid_artifacts import canonical_json_bytes


BASELINE_SCHEMA = "cohort_v2_policy_baselines_v1"
POLICY_ORDER = (
    "fixed_pair",
    "temporal_only",
    "description_only",
    "uniformly_marginalized_independent_axes",
)
ROLE_PERMISSIONS = {
    "training": ["learned_parameters"],
    "calibration": ["threshold_values", "tolerance_values"],
    "model_selection": ["configuration_selection"],
    "final_evaluation": ["frozen_final_metrics_after_authorization"],
}


class CohortV2BaselineError(ValueError):
    """The baseline evidence or resulting artifact is invalid."""


@dataclass(frozen=True, slots=True)
class CohortV2BaselineDecision:
    policy_id: str
    state_id: str
    exposure_role: str
    scenario_lineage_identity: str
    selected_pair: PredictionPair
    prediction_objective: float
    endpoint_violation_rate: float
    policy_compute_per_simulated_frame: float
    full_compute_per_simulated_frame: float
    segment_cost: float


@dataclass(frozen=True, slots=True)
class CohortV2BaselineScore:
    policy_id: str
    exposure_role: str
    state_count: int
    mean_prediction_objective: float
    mean_endpoint_violation_rate: float
    mean_policy_compute_per_simulated_frame: float
    mean_full_compute_per_simulated_frame: float
    mean_segment_cost: float


@dataclass(frozen=True, slots=True)
class CohortV2BaselineResult:
    evaluation_identity: str
    measurement_identity: str
    trajectory_label_artifact_identity: str
    cost_spec_identity: str
    derivation_index_identity: str
    comparison_state_set_identity: str
    selected_configurations: dict[str, object]
    decisions: tuple[CohortV2BaselineDecision, ...]
    scores: tuple[CohortV2BaselineScore, ...]
    frontiers: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class CohortV2BaselineReceipt:
    baseline_artifact_identity: str
    implementation_revision: str
    evaluation_identity: str
    measurement_identity: str
    trajectory_label_artifact_identity: str
    comparison_state_set_identity: str
    decision_count: int
    comparison_state_count: int


@dataclass(frozen=True, slots=True)
class _PairCost:
    pair: PredictionPair
    prediction_objective: float
    endpoint_violation_rate: float
    policy_compute_per_simulated_frame: float
    full_compute_per_simulated_frame: float
    segment_cost: float


def _require_identity(value: str, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise CohortV2BaselineError(f"{field} must be a nonempty identity")
    return value


def _paired_states(
    evaluation: CohortV2EvaluationResult,
    measurement: CohortV2MeasurementResult,
) -> tuple[tuple[CohortV2StateEvaluation, CohortV2StateMeasurement], ...]:
    if not isinstance(evaluation, CohortV2EvaluationResult) or not isinstance(
        measurement, CohortV2MeasurementResult
    ):
        raise CohortV2BaselineError(
            "baselines require exhaustive evaluation and measurement results"
        )
    if measurement.evaluation_identity != evaluation.identity:
        raise CohortV2BaselineError(
            "pair measurements do not belong to the exhaustive evaluation"
        )
    if len(evaluation.states) != len(measurement.states):
        raise CohortV2BaselineError("evaluation and measurement states differ")
    paired = tuple(zip(evaluation.states, measurement.states, strict=True))
    if any(
        state.state_id != measured.state_id
        or len(state.outcomes) != len(measured.outcomes)
        for state, measured in paired
    ):
        raise CohortV2BaselineError("evaluation and measurement states differ")
    return paired


def _pair_cost(
    state: CohortV2StateEvaluation,
    outcome: CohortV2PairOutcome,
    measured: CohortV2PairMeasurement,
    spec: CohortV2TrajectoryCostSpec,
) -> _PairCost | None:
    if (
        outcome.pair != measured.pair
        or outcome.effective_horizon != measured.effective_horizon
        or outcome.target_frame_record_identity
        != measured.target_frame_record_identity
    ):
        raise CohortV2BaselineError(
            "pair measurement differs from its exhaustive outcome"
        )
    if not outcome.available:
        return None
    plausibility = measured.endpoint_plausibility
    compute = measured.compute
    if plausibility is None or compute is None or plausibility.violation_rate is None:
        return None
    duration_weight = outcome.effective_horizon / (state.frame_record_count - 1)
    segment_cost = (
        float(outcome.objective)
        + duration_weight
        * spec.physical_violation_weight
        * plausibility.violation_rate
        + spec.compute_weight
        * compute.policy_dependent_total
        / spec.compute_reference
    )
    return _PairCost(
        pair=outcome.pair,
        prediction_objective=float(outcome.objective),
        endpoint_violation_rate=plausibility.violation_rate,
        policy_compute_per_simulated_frame=(
            compute.policy_dependent_per_simulated_frame
        ),
        full_compute_per_simulated_frame=compute.full_end_to_end_per_simulated_frame,
        segment_cost=segment_cost,
    )


def _complete_state_costs(
    evaluation: CohortV2EvaluationResult,
    measurement: CohortV2MeasurementResult,
    spec: CohortV2TrajectoryCostSpec,
) -> tuple[tuple[CohortV2StateEvaluation, dict[PredictionPair, _PairCost]], ...]:
    complete = []
    declared_pairs = tuple(evaluation.grid.pairs)
    for state, measured in _paired_states(evaluation, measurement):
        costs = {}
        for outcome, pair_measurement in zip(
            state.outcomes, measured.outcomes, strict=True
        ):
            cost = _pair_cost(state, outcome, pair_measurement, spec)
            if cost is not None:
                costs[cost.pair] = cost
        if tuple(costs) == declared_pairs:
            complete.append((state, costs))
    roles = tuple(dict.fromkeys(state.exposure_role for state, _costs in complete))
    required_roles = ("training", "calibration", "model_selection")
    if roles != required_roles or any(
        not any(state.exposure_role == role for state, _costs in complete)
        for role in required_roles
    ):
        raise CohortV2BaselineError(
            "complete-grid comparison scope must cover every public exposure role"
        )
    return tuple(complete)


def _pair_order(pair: PredictionPair) -> tuple[int, int]:
    return pair.delta, ABSTRACTION_ORDER.index(pair.abstraction)


def _tolerant_minimum(items: tuple[tuple[float, object], ...], order) -> object:
    minimum = min(value for value, _item in items)
    tied = tuple(
        item
        for value, item in items
        if math.isclose(value, minimum, rel_tol=TIE_REL_TOL, abs_tol=TIE_ABS_TOL)
    )
    return min(tied, key=order)


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values)


def _select_configurations(
    scoped: tuple[tuple[CohortV2StateEvaluation, dict[PredictionPair, _PairCost]], ...],
    pairs: tuple[PredictionPair, ...],
) -> tuple[PredictionPair, Abstraction, int]:
    selection_states = tuple(
        costs for state, costs in scoped if state.exposure_role == "model_selection"
    )
    fixed_pair = _tolerant_minimum(
        tuple(
            (_mean(tuple(costs[pair].segment_cost for costs in selection_states)), pair)
            for pair in pairs
        ),
        _pair_order,
    )
    temporal_mode = _tolerant_minimum(
        tuple(
            (
                _mean(tuple(
                    min(
                        costs[pair].segment_cost
                        for pair in pairs
                        if pair.abstraction is mode
                    )
                    for costs in selection_states
                )),
                mode,
            )
            for mode in ABSTRACTION_ORDER
        ),
        ABSTRACTION_ORDER.index,
    )
    horizons = tuple(dict.fromkeys(pair.delta for pair in pairs))
    description_horizon = _tolerant_minimum(
        tuple(
            (
                _mean(tuple(
                    min(
                        costs[pair].segment_cost
                        for pair in pairs
                        if pair.delta == horizon
                    )
                    for costs in selection_states
                )),
                horizon,
            )
            for horizon in horizons
        ),
        horizons.index,
    )
    return fixed_pair, temporal_mode, description_horizon


def _select_decision_cost(
    policy_id: str,
    costs: dict[PredictionPair, _PairCost],
    pairs: tuple[PredictionPair, ...],
    fixed_pair: PredictionPair,
    temporal_mode: Abstraction,
    description_horizon: int,
) -> _PairCost:
    if policy_id == "fixed_pair":
        return costs[fixed_pair]
    if policy_id == "temporal_only":
        candidates = tuple(
            costs[pair] for pair in pairs if pair.abstraction is temporal_mode
        )
        return _tolerant_minimum(
            tuple((cost.segment_cost, cost) for cost in candidates),
            lambda cost: _pair_order(cost.pair),
        )
    if policy_id == "description_only":
        candidates = tuple(
            costs[pair] for pair in pairs if pair.delta == description_horizon
        )
        return _tolerant_minimum(
            tuple((cost.segment_cost, cost) for cost in candidates),
            lambda cost: _pair_order(cost.pair),
        )
    if policy_id != "uniformly_marginalized_independent_axes":
        raise CohortV2BaselineError(f"unknown baseline policy {policy_id}")

    horizons = tuple(dict.fromkeys(pair.delta for pair in pairs))
    horizon = _tolerant_minimum(
        tuple(
            (
                _mean(tuple(
                    costs[pair].segment_cost
                    for pair in pairs
                    if pair.delta == candidate_horizon
                )),
                candidate_horizon,
            )
            for candidate_horizon in horizons
        ),
        horizons.index,
    )
    mode = _tolerant_minimum(
        tuple(
            (
                _mean(tuple(
                    costs[pair].segment_cost
                    for pair in pairs
                    if pair.abstraction is candidate_mode
                )),
                candidate_mode,
            )
            for candidate_mode in ABSTRACTION_ORDER
        ),
        ABSTRACTION_ORDER.index,
    )
    return costs[PredictionPair(horizon, mode)]


def _scores(
    decisions: tuple[CohortV2BaselineDecision, ...],
) -> tuple[CohortV2BaselineScore, ...]:
    roles = tuple(dict.fromkeys(decision.exposure_role for decision in decisions))
    scores = []
    for role in roles:
        for policy_id in POLICY_ORDER:
            rows = tuple(
                decision
                for decision in decisions
                if decision.exposure_role == role and decision.policy_id == policy_id
            )
            scores.append(CohortV2BaselineScore(
                policy_id=policy_id,
                exposure_role=role,
                state_count=len(rows),
                mean_prediction_objective=_mean(tuple(
                    row.prediction_objective for row in rows
                )),
                mean_endpoint_violation_rate=_mean(tuple(
                    row.endpoint_violation_rate for row in rows
                )),
                mean_policy_compute_per_simulated_frame=_mean(tuple(
                    row.policy_compute_per_simulated_frame for row in rows
                )),
                mean_full_compute_per_simulated_frame=_mean(tuple(
                    row.full_compute_per_simulated_frame for row in rows
                )),
                mean_segment_cost=_mean(tuple(row.segment_cost for row in rows)),
            ))
    return tuple(scores)


def _frontiers(
    scores: tuple[CohortV2BaselineScore, ...],
) -> dict[str, tuple[str, ...]]:
    frontiers = {}
    for role in dict.fromkeys(score.exposure_role for score in scores):
        role_scores = tuple(score for score in scores if score.exposure_role == role)
        frontier = []
        for candidate in role_scores:
            candidate_axes = (
                candidate.mean_prediction_objective,
                candidate.mean_endpoint_violation_rate,
                candidate.mean_policy_compute_per_simulated_frame,
            )
            dominated = False
            for other in role_scores:
                if other.policy_id == candidate.policy_id:
                    continue
                other_axes = (
                    other.mean_prediction_objective,
                    other.mean_endpoint_violation_rate,
                    other.mean_policy_compute_per_simulated_frame,
                )
                if all(left <= right for left, right in zip(other_axes, candidate_axes)) and any(
                    left < right for left, right in zip(other_axes, candidate_axes)
                ):
                    dominated = True
                    break
            if not dominated:
                frontier.append(candidate.policy_id)
        frontiers[role] = tuple(frontier)
    return frontiers


def generate_cohort_v2_policy_baselines(
    evaluation: CohortV2EvaluationResult,
    measurement: CohortV2MeasurementResult,
    spec: CohortV2TrajectoryCostSpec,
    *,
    trajectory_label_artifact_identity: str,
    derivation_index_identity: str,
) -> CohortV2BaselineResult:
    """Generate four directly comparable baselines on complete pair-grid states."""
    if not isinstance(spec, CohortV2TrajectoryCostSpec):
        raise CohortV2BaselineError("trajectory cost specification is malformed")
    trajectory_label_artifact_identity = _require_identity(
        trajectory_label_artifact_identity, "trajectory label artifact"
    )
    derivation_index_identity = _require_identity(
        derivation_index_identity, "derivation index"
    )
    scoped = _complete_state_costs(evaluation, measurement, spec)
    pairs = tuple(evaluation.grid.pairs)
    fixed_pair, temporal_mode, description_horizon = _select_configurations(
        scoped, pairs
    )
    decisions = []
    for state, costs in scoped:
        for policy_id in POLICY_ORDER:
            selected = _select_decision_cost(
                policy_id,
                costs,
                pairs,
                fixed_pair,
                temporal_mode,
                description_horizon,
            )
            decisions.append(CohortV2BaselineDecision(
                policy_id=policy_id,
                state_id=state.state_id,
                exposure_role=state.exposure_role,
                scenario_lineage_identity=state.scenario_lineage_identity,
                selected_pair=selected.pair,
                prediction_objective=selected.prediction_objective,
                endpoint_violation_rate=selected.endpoint_violation_rate,
                policy_compute_per_simulated_frame=(
                    selected.policy_compute_per_simulated_frame
                ),
                full_compute_per_simulated_frame=(
                    selected.full_compute_per_simulated_frame
                ),
                segment_cost=selected.segment_cost,
            ))
    state_ids = tuple(state.state_id for state, _costs in scoped)
    comparison_state_set_identity = identity((
        "cohort-v2-complete-pair-grid-baseline-scope-v1",
        evaluation.state_set_identity,
        state_ids,
    ))
    scores = _scores(tuple(decisions))
    return CohortV2BaselineResult(
        evaluation_identity=evaluation.identity,
        measurement_identity=measurement.identity,
        trajectory_label_artifact_identity=trajectory_label_artifact_identity,
        cost_spec_identity=spec.identity,
        derivation_index_identity=derivation_index_identity,
        comparison_state_set_identity=comparison_state_set_identity,
        selected_configurations={
            "configuration_selection_role": "model_selection",
            "description_only_fixed_requested_horizon": description_horizon,
            "fixed_pair": _pair_payload(fixed_pair),
            "temporal_only_fixed_abstraction": str(temporal_mode),
        },
        decisions=tuple(decisions),
        scores=scores,
        frontiers=_frontiers(scores),
    )


def _pair_payload(pair: PredictionPair) -> dict[str, str | int]:
    return {"abstraction": str(pair.abstraction), "requested_horizon": pair.delta}


def _decision_payload(decision: CohortV2BaselineDecision) -> dict[str, object]:
    return {
        "endpoint_violation_rate": decision.endpoint_violation_rate,
        "exposure_role": decision.exposure_role,
        "full_compute_per_simulated_frame": decision.full_compute_per_simulated_frame,
        "policy_compute_per_simulated_frame": (
            decision.policy_compute_per_simulated_frame
        ),
        "policy_id": decision.policy_id,
        "prediction_objective": decision.prediction_objective,
        "record_type": "baseline_policy_decision",
        "scenario_lineage_identity": decision.scenario_lineage_identity,
        "schema": BASELINE_SCHEMA,
        "segment_cost": decision.segment_cost,
        "selected_pair": _pair_payload(decision.selected_pair),
        "state_id": decision.state_id,
    }


def _score_payload(score: CohortV2BaselineScore) -> dict[str, object]:
    return {
        "exposure_role": score.exposure_role,
        "mean_endpoint_violation_rate": score.mean_endpoint_violation_rate,
        "mean_full_compute_per_simulated_frame": (
            score.mean_full_compute_per_simulated_frame
        ),
        "mean_policy_compute_per_simulated_frame": (
            score.mean_policy_compute_per_simulated_frame
        ),
        "mean_prediction_objective": score.mean_prediction_objective,
        "mean_segment_cost": score.mean_segment_cost,
        "policy_id": score.policy_id,
        "state_count": score.state_count,
    }


def _artifact_bytes(result: CohortV2BaselineResult) -> tuple[bytes, bytes, bytes]:
    decisions = b"".join(
        canonical_json_bytes(_decision_payload(decision))
        for decision in result.decisions
    )
    scores = canonical_json_bytes({
        "comparison_state_set_identity": result.comparison_state_set_identity,
        "schema": BASELINE_SCHEMA,
        "scores": [_score_payload(score) for score in result.scores],
    })
    frontiers = canonical_json_bytes({
        "axes": [
            "mean_prediction_objective",
            "mean_endpoint_violation_rate",
            "mean_policy_compute_per_simulated_frame",
        ],
        "comparison_state_set_identity": result.comparison_state_set_identity,
        "frontiers": [
            {"exposure_role": role, "policy_ids": list(policy_ids)}
            for role, policy_ids in result.frontiers.items()
        ],
        "schema": BASELINE_SCHEMA,
    })
    return decisions, scores, frontiers


def _bytes_identity(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _manifest(
    result: CohortV2BaselineResult,
    evaluation: CohortV2EvaluationResult,
    measurement: CohortV2MeasurementResult,
    artifact_bytes: tuple[bytes, bytes, bytes],
    implementation_revision: str,
) -> dict[str, object]:
    implementation_revision = _require_identity(
        implementation_revision, "implementation revision"
    )
    decisions, scores, frontiers = artifact_bytes
    artifact_identity = identity((
        "cohort-v2-policy-baselines-v1",
        result.evaluation_identity,
        result.measurement_identity,
        result.trajectory_label_artifact_identity,
        result.cost_spec_identity,
        result.derivation_index_identity,
        result.comparison_state_set_identity,
        implementation_revision,
        _bytes_identity(decisions),
        _bytes_identity(scores),
        _bytes_identity(frontiers),
    ))
    role_counts = {
        role: sum(
            decision.exposure_role == role and decision.policy_id == POLICY_ORDER[0]
            for decision in result.decisions
        )
        for role in ("training", "calibration", "model_selection")
    }
    return {
        "artifact_type": "cohort_v2_policy_baselines",
        "baseline_artifact_identity": artifact_identity,
        "capability_declaration_identity": evaluation.capability_declaration_identity,
        "checkpoint_capabilities": list(evaluation.checkpoint_capabilities),
        "checkpoint_identity": evaluation.checkpoint_identity,
        "comparison_scope": "states_with_all_declared_pairs_available",
        "comparison_state_count": sum(role_counts.values()),
        "comparison_state_count_by_role": role_counts,
        "comparison_state_set_identity": result.comparison_state_set_identity,
        "compute_calibration_identity": measurement.compute_calibration_identity,
        "cost_spec_identity": result.cost_spec_identity,
        "decision_count": len(result.decisions),
        "decisions": "baseline_decisions.jsonl",
        "decisions_identity": _bytes_identity(decisions),
        "derivation_index_identity": result.derivation_index_identity,
        "evaluation_identity": result.evaluation_identity,
        "execution_profile_identity": measurement.execution_profile_identity,
        "final_evaluation_consumed": False,
        "frontier": "frontier.json",
        "frontier_identity": _bytes_identity(frontiers),
        "grid_identity": evaluation.grid.identity,
        "implementation_revision": implementation_revision,
        "independent_axis_objective": (
            "uniform_expected_segment_cost_over_the_other_declared_axis"
        ),
        "independent_axes_are_output_factorization": False,
        "measurement_identity": result.measurement_identity,
        "objective_identity": evaluation.objective_identity,
        "partition_identity": evaluation.partition_identity,
        "policies": list(POLICY_ORDER),
        "policy_evidence_kind": "oracle_exhaustive_pair_costs",
        "policy_fitting": "none",
        "release_identity": evaluation.release_identity,
        "role_permissions": ROLE_PERMISSIONS,
        "roles_consumed": ["training", "calibration", "model_selection"],
        "schema": BASELINE_SCHEMA,
        "scores": "scores.json",
        "scores_identity": _bytes_identity(scores),
        "selected_configurations": result.selected_configurations,
        "state_set_identity": evaluation.state_set_identity,
        "trajectory_label_artifact_identity": (
            result.trajectory_label_artifact_identity
        ),
    }


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _receipt(manifest: dict[str, object]) -> CohortV2BaselineReceipt:
    return CohortV2BaselineReceipt(
        baseline_artifact_identity=manifest["baseline_artifact_identity"],
        implementation_revision=manifest["implementation_revision"],
        evaluation_identity=manifest["evaluation_identity"],
        measurement_identity=manifest["measurement_identity"],
        trajectory_label_artifact_identity=manifest[
            "trajectory_label_artifact_identity"
        ],
        comparison_state_set_identity=manifest["comparison_state_set_identity"],
        decision_count=manifest["decision_count"],
        comparison_state_count=manifest["comparison_state_count"],
    )


def validate_cohort_v2_policy_baselines(
    root: Path,
    evaluation: CohortV2EvaluationResult,
    measurement: CohortV2MeasurementResult,
    spec: CohortV2TrajectoryCostSpec,
    *,
    trajectory_label_artifact_identity: str,
    derivation_index_identity: str,
    implementation_revision: str,
) -> CohortV2BaselineReceipt:
    """Recompute the baseline suite and compare every artifact byte."""
    root = Path(root)
    try:
        manifest_raw = (root / "manifest.json").read_bytes()
        manifest = json.loads(manifest_raw)
        actual = tuple(
            (root / name).read_bytes()
            for name in ("baseline_decisions.jsonl", "scores.json", "frontier.json")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise CohortV2BaselineError(f"cannot load policy baselines: {error}") from error
    result = generate_cohort_v2_policy_baselines(
        evaluation,
        measurement,
        spec,
        trajectory_label_artifact_identity=trajectory_label_artifact_identity,
        derivation_index_identity=derivation_index_identity,
    )
    expected = _artifact_bytes(result)
    expected_manifest = _manifest(
        result, evaluation, measurement, expected, implementation_revision
    )
    if (
        actual != expected
        or manifest != expected_manifest
        or canonical_json_bytes(manifest) != manifest_raw
    ):
        raise CohortV2BaselineError(
            "policy baselines differ from their exhaustive pair evidence"
        )
    return _receipt(manifest)


def write_cohort_v2_policy_baselines(
    root: Path,
    evaluation: CohortV2EvaluationResult,
    measurement: CohortV2MeasurementResult,
    spec: CohortV2TrajectoryCostSpec,
    *,
    trajectory_label_artifact_identity: str,
    derivation_index_identity: str,
    implementation_revision: str,
) -> CohortV2BaselineReceipt:
    """Write and source-validate the baseline decision, score, and frontier suite."""
    result = generate_cohort_v2_policy_baselines(
        evaluation,
        measurement,
        spec,
        trajectory_label_artifact_identity=trajectory_label_artifact_identity,
        derivation_index_identity=derivation_index_identity,
    )
    artifacts = _artifact_bytes(result)
    manifest = _manifest(
        result, evaluation, measurement, artifacts, implementation_revision
    )
    root = Path(root)
    for name, data in zip(
        ("baseline_decisions.jsonl", "scores.json", "frontier.json"),
        artifacts,
        strict=True,
    ):
        _atomic_write(root / name, data)
    _atomic_write(root / "manifest.json", canonical_json_bytes(manifest))
    return validate_cohort_v2_policy_baselines(
        root,
        evaluation,
        measurement,
        spec,
        trajectory_label_artifact_identity=trajectory_label_artifact_identity,
        derivation_index_identity=derivation_index_identity,
        implementation_revision=implementation_revision,
    )


__all__ = [
    "CohortV2BaselineDecision",
    "CohortV2BaselineError",
    "CohortV2BaselineReceipt",
    "CohortV2BaselineResult",
    "CohortV2BaselineScore",
    "generate_cohort_v2_policy_baselines",
    "validate_cohort_v2_policy_baselines",
    "write_cohort_v2_policy_baselines",
]
