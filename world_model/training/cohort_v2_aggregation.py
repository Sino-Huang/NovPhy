"""Closed-loop controller dataset aggregation for cohort v2."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path

import torch

from world_model.data import (
    CohortV2OracleWindow,
    CohortV2ReleaseReader,
    CohortV2Rollout,
)
from world_model.data.cohort_v2 import (
    CAPABILITY_DECLARATION_IDENTITY,
    COHORT_V2_RELEASE_IDENTITY,
)
from world_model.model import (
    ABSTRACTION_ORDER,
    Abstraction,
    DualOutputPredictor,
    PredictionPair,
    identity,
)
from world_model.training.cohort_v2 import build_cohort_v2_transition_request
from world_model.training.cohort_v2_controller import (
    CohortV2ControllerConfig,
    CohortV2ControllerExample,
    CohortV2JointPairController,
    CohortV2TwoHeadController,
    build_cohort_v2_controller_examples,
    select_cohort_v2_controller_pairs,
    train_cohort_v2_controllers,
)
from world_model.training.cohort_v2_evaluation import (
    CohortV2EvaluationResult,
    CohortV2PairOutcome,
    CohortV2StateEvaluation,
)
from world_model.training.cohort_v2_macro import macro_readout_loss
from world_model.training.cohort_v2_measurement import (
    CohortV2MeasurementResult,
    CohortV2PairMeasurement,
)
from world_model.training.cohort_v2_micro import (
    CohortV2StateCodec,
    cohort_v2_action,
    micro_predicate_loss,
    micro_relation_loss,
)
from world_model.training.cohort_v2_trajectory_labels import (
    CohortV2ControllerLabel,
    CohortV2ControllerLabelResult,
    CohortV2TrajectoryCostSpec,
)
from world_model.training.grid_artifacts import canonical_json_bytes


AGGREGATION_SCHEMA = "cohort_v2_closed_loop_controller_aggregation_v1"
TIE_REL_TOL = 1e-6
TIE_ABS_TOL = 1e-12


class CohortV2AggregationError(ValueError):
    """Closed-loop membership, relabelling, or artifacts are invalid."""


@dataclass(frozen=True, slots=True)
class CohortV2AggregationConfig:
    rounds: int = 1

    def __post_init__(self) -> None:
        if self.rounds not in (1, 2):
            raise CohortV2AggregationError("aggregation requires one or two declared rounds")

    @property
    def identity(self) -> str:
        return identity((
            "cohort-v2-closed-loop-aggregation-config-v1",
            self.rounds,
            "training-role-rollouts-only",
            "aligned-ground-truth-expert-continuation",
        ))


@dataclass(frozen=True, slots=True)
class CohortV2AggregatedState:
    round_index: int
    state_id: str
    attempt_id: str
    scenario_lineage_identity: str
    context_position: int
    context_fixed_step: int
    carrier: tuple[float, ...]
    selected_pair: PredictionPair
    relabelled_pair: PredictionPair
    selected_segment_cost: float
    relabelled_segment_cost: float
    selected_cost_to_go: float
    relabelled_cost_to_go: float


@dataclass(frozen=True, slots=True)
class CohortV2ClosedLoopScore:
    name: str
    exposure_role: str
    rollout_count: int
    decision_count: int
    mean_terminal_carrier_mse: float
    mean_endpoint_violation_rate: float
    mean_selected_segment_cost: float
    mean_pair_regret: float


@dataclass(frozen=True, slots=True)
class CohortV2AggregationResult:
    states: tuple[CohortV2AggregatedState, ...]
    scores: tuple[CohortV2ClosedLoopScore, ...]
    round_training_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CohortV2AggregationRun:
    result: CohortV2AggregationResult
    round_models: tuple[
        tuple[CohortV2JointPairController, CohortV2TwoHeadController], ...
    ]


@dataclass(frozen=True, slots=True)
class _Candidate:
    pair: PredictionPair
    segment_cost: float
    cost_to_go: float
    endpoint_violation_rate: float
    successor: torch.Tensor


@dataclass(frozen=True, slots=True)
class _RolloutResult:
    states: tuple[CohortV2AggregatedState, ...]
    decision_count: int
    terminal_carrier_mse: float
    endpoint_violation_rate: float
    mean_selected_segment_cost: float
    mean_pair_regret: float


def _window(
    rollout: CohortV2Rollout,
    state: CohortV2StateEvaluation,
    outcome: CohortV2PairOutcome,
) -> CohortV2OracleWindow:
    target_position = state.context_position + outcome.effective_horizon
    return CohortV2OracleWindow(
        source_release_identity=COHORT_V2_RELEASE_IDENTITY,
        capability_declaration_identity=CAPABILITY_DECLARATION_IDENTITY,
        exposure_role=rollout.exposure_role,
        attempt_id=rollout.attempt_id,
        scenario_lineage_identity=rollout.scenario_lineage_identity,
        intervention=rollout.intervention,
        context_position=state.context_position,
        requested_horizon=outcome.requested_horizon,
        effective_horizon=outcome.effective_horizon,
        context=rollout.frame_records[state.context_position],
        target=rollout.frame_records[target_position],
        agent_observation=b"",
    )


def _candidate(
    carrier: torch.Tensor,
    predictor: DualOutputPredictor,
    codec: CohortV2StateCodec,
    rollout: CohortV2Rollout,
    state: CohortV2StateEvaluation,
    outcome: CohortV2PairOutcome,
    measured: CohortV2PairMeasurement,
    spec: CohortV2TrajectoryCostSpec,
    labels_by_position: Mapping[int, CohortV2ControllerLabel],
) -> _Candidate | None:
    if not outcome.available:
        return None
    plausibility = measured.endpoint_plausibility
    compute = measured.compute
    if plausibility is None or compute is None or plausibility.violation_rate is None:
        return None
    window = _window(rollout, state, outcome)
    device = carrier.device
    action = cohort_v2_action(window).unsqueeze(0).to(device)
    request = build_cohort_v2_transition_request(outcome.pair, (window,))
    with torch.no_grad():
        successor = predictor.carrier(carrier.unsqueeze(0), action, request)
        target = codec.encode(window.target).unsqueeze(0).to(device)
        quality = (successor - target).pow(2).mean()
        if outcome.pair.abstraction is Abstraction.MICRO:
            relation = micro_relation_loss(
                predictor.micro_head, successor, (window.target,)
            ).loss
            predicate = micro_predicate_loss(
                predictor.micro_head, successor, (window.target,)
            ).loss
            quality = quality + (relation + predicate) / 2.0
        elif outcome.pair.abstraction is Abstraction.MACRO:
            quality = quality + macro_readout_loss(
                predictor.macro_head, successor, (window,)
            ).loss
    duration_weight = outcome.effective_horizon / (state.frame_record_count - 1)
    segment_cost = (
        duration_weight * float(quality)
        + duration_weight
        * spec.physical_violation_weight
        * plausibility.violation_rate
        + spec.compute_weight
        * compute.policy_dependent_total
        / spec.compute_reference
    )
    next_position = state.context_position + outcome.effective_horizon
    future = 0.0
    if next_position < state.frame_record_count - 1:
        future = labels_by_position[next_position].cost_to_go
    return _Candidate(
        pair=outcome.pair,
        segment_cost=segment_cost,
        cost_to_go=segment_cost + future,
        endpoint_violation_rate=plausibility.violation_rate,
        successor=successor.squeeze(0),
    )


def _best(candidates: tuple[_Candidate, ...]) -> _Candidate:
    minimum = min(item.cost_to_go for item in candidates)
    ties = tuple(
        item
        for item in candidates
        if math.isclose(
            item.cost_to_go, minimum, rel_tol=TIE_REL_TOL, abs_tol=TIE_ABS_TOL
        )
    )
    return min(
        ties,
        key=lambda item: (
            item.pair.delta,
            ABSTRACTION_ORDER.index(item.pair.abstraction),
        ),
    )


def _rollout(
    *,
    name: str,
    round_index: int,
    model: CohortV2JointPairController,
    rollout: CohortV2Rollout,
    predictor: DualOutputPredictor,
    codec: CohortV2StateCodec,
    evaluation: CohortV2EvaluationResult,
    measurement: CohortV2MeasurementResult,
    labels: CohortV2ControllerLabelResult,
    spec: CohortV2TrajectoryCostSpec,
    examples_by_state: Mapping[str, CohortV2ControllerExample],
    collect_states: bool,
    progress: Callable[[str], None] | None,
) -> _RolloutResult:
    states_by_position = {
        item.context_position: item
        for item in evaluation.states
        if item.attempt_id == rollout.attempt_id
    }
    measured_by_state = {item.state_id: item for item in measurement.states}
    labels_by_position = {
        item.context_position: item
        for item in labels.labels
        if item.attempt_id == rollout.attempt_id
    }
    terminal_position = len(rollout.frame_records) - 1
    if set(states_by_position) != set(range(terminal_position)):
        raise CohortV2AggregationError("closed-loop rollout lacks aligned pair evidence")
    carrier = codec.encode(rollout.frame_records[0]).to(next(predictor.parameters()).device)
    position = 0
    records = []
    segment_costs = []
    regrets = []
    endpoint_violation = 0.0
    while position < terminal_position:
        state = states_by_position[position]
        example = examples_by_state[state.state_id]
        pair = select_cohort_v2_controller_pairs(
            "joint_pair", model, example.features.unsqueeze(0), evaluation.grid.pairs
        )[0]
        measured = measured_by_state[state.state_id]
        candidates = tuple(
            candidate
            for outcome, pair_measurement in zip(
                state.outcomes, measured.outcomes, strict=True
            )
            if (
                candidate := _candidate(
                    carrier,
                    predictor,
                    codec,
                    rollout,
                    state,
                    outcome,
                    pair_measurement,
                    spec=spec,
                    labels_by_position=labels_by_position,
                )
            )
            is not None
        )
        if not candidates:
            raise CohortV2AggregationError(f"state {state.state_id} has no relabelling evidence")
        by_pair = {item.pair: item for item in candidates}
        selected = by_pair.get(pair)
        if selected is None:
            raise CohortV2AggregationError(
                f"controller selected unavailable pair {pair.identity}"
            )
        relabelled = _best(candidates)
        if collect_states and position > 0:
            records.append(CohortV2AggregatedState(
                round_index=round_index,
                state_id=state.state_id,
                attempt_id=rollout.attempt_id,
                scenario_lineage_identity=rollout.scenario_lineage_identity,
                context_position=position,
                context_fixed_step=state.context_fixed_step,
                carrier=tuple(float(value) for value in carrier.detach().cpu().tolist()),
                selected_pair=selected.pair,
                relabelled_pair=relabelled.pair,
                selected_segment_cost=selected.segment_cost,
                relabelled_segment_cost=relabelled.segment_cost,
                selected_cost_to_go=selected.cost_to_go,
                relabelled_cost_to_go=relabelled.cost_to_go,
            ))
        segment_costs.append(selected.segment_cost)
        regrets.append(selected.cost_to_go - relabelled.cost_to_go)
        endpoint_violation = selected.endpoint_violation_rate
        carrier = selected.successor
        position += next(
            outcome.effective_horizon
            for outcome in state.outcomes
            if outcome.pair == selected.pair
        )
        if progress is not None and (
            len(segment_costs) == 1
            or len(segment_costs) % 50 == 0
            or position == terminal_position
        ):
            progress(
                f"[rollout {name}] decisions={len(segment_costs)} "
                f"position={position}/{terminal_position} "
                f"pair=h{selected.pair.delta}/{selected.pair.abstraction}"
            )
    target = codec.encode(rollout.frame_records[-1]).to(carrier.device)
    terminal_mse = float((carrier - target).pow(2).mean())
    return _RolloutResult(
        states=tuple(records),
        decision_count=len(segment_costs),
        terminal_carrier_mse=terminal_mse,
        endpoint_violation_rate=endpoint_violation,
        mean_selected_segment_cost=sum(segment_costs) / len(segment_costs),
        mean_pair_regret=sum(regrets) / len(regrets),
    )


def _score(name: str, role: str, rollouts: tuple[_RolloutResult, ...]) -> CohortV2ClosedLoopScore:
    if not rollouts:
        raise CohortV2AggregationError(f"{name} closed-loop evaluation is empty")
    return CohortV2ClosedLoopScore(
        name=name,
        exposure_role=role,
        rollout_count=len(rollouts),
        decision_count=sum(item.decision_count for item in rollouts),
        mean_terminal_carrier_mse=(
            sum(item.terminal_carrier_mse for item in rollouts) / len(rollouts)
        ),
        mean_endpoint_violation_rate=(
            sum(item.endpoint_violation_rate for item in rollouts) / len(rollouts)
        ),
        mean_selected_segment_cost=(
            sum(item.mean_selected_segment_cost for item in rollouts) / len(rollouts)
        ),
        mean_pair_regret=sum(item.mean_pair_regret for item in rollouts) / len(rollouts),
    )


def run_cohort_v2_controller_aggregation(
    readers: tuple[CohortV2ReleaseReader, ...],
    evaluation: CohortV2EvaluationResult,
    measurement: CohortV2MeasurementResult,
    labels: CohortV2ControllerLabelResult,
    spec: CohortV2TrajectoryCostSpec,
    predictor: DualOutputPredictor,
    codec: CohortV2StateCodec,
    baseline_models: tuple[CohortV2JointPairController, CohortV2TwoHeadController],
    controller_config: CohortV2ControllerConfig,
    aggregation_config: CohortV2AggregationConfig,
    *,
    progress: Callable[[str], None] | None = None,
    rollout_limit: int | None = None,
) -> CohortV2AggregationRun:
    """Collect predicted training carriers, relabel them, and retrain cumulatively."""
    if measurement.evaluation_identity != evaluation.identity:
        raise CohortV2AggregationError("aggregation measurements belong to another evaluation")
    if tuple(reader.rollouts[0].exposure_role for reader in readers) != (
        "training", "calibration", "model_selection"
    ):
        raise CohortV2AggregationError("aggregation readers must preserve public role order")
    base_examples = build_cohort_v2_controller_examples(readers, labels, controller_config)
    examples_by_state = {item.state_id: item for item in base_examples}
    cumulative = list(base_examples)
    current = baseline_models
    round_models = []
    aggregated_states = []
    training_rollouts = readers[0].rollouts[:rollout_limit]
    for round_index in range(1, aggregation_config.rounds + 1):
        visits = []
        for index, rollout in enumerate(training_rollouts, start=1):
            if progress is not None:
                progress(
                    f"[aggregate round={round_index} rollout={index}/{len(training_rollouts)}] "
                    f"attempt={rollout.attempt_id}"
                )
            result = _rollout(
                name=f"round_{round_index}",
                round_index=round_index,
                model=current[0],
                rollout=rollout,
                predictor=predictor,
                codec=codec,
                evaluation=evaluation,
                measurement=measurement,
                labels=labels,
                spec=spec,
                examples_by_state=examples_by_state,
                collect_states=True,
                progress=progress,
            )
            visits.extend(result.states)
        if not visits:
            raise CohortV2AggregationError("learned rollouts visited no predicted carrier states")
        for item in visits:
            source = examples_by_state[item.state_id]
            cumulative.append(CohortV2ControllerExample(
                state_id=f"closed-loop-round-{round_index}:{item.state_id}",
                exposure_role="training",
                scenario_lineage_identity=item.scenario_lineage_identity,
                features=source.features,
                oracle_pair=item.relabelled_pair,
                oracle_segment_cost=item.relabelled_segment_cost,
            ))
        aggregated_states.extend(visits)
        if progress is not None:
            progress(
                f"[aggregate round={round_index}] added={len(visits)} "
                "cumulative_training="
                f"{sum(item.exposure_role == 'training' for item in cumulative)}"
            )
        current = train_cohort_v2_controllers(
            tuple(cumulative),
            evaluation.grid.pairs,
            controller_config,
            progress=progress,
        )
        round_models.append(current)

    evaluation_rollouts = readers[2].rollouts[:rollout_limit]
    scores = []
    for name, models in (("oracle_state_baseline", baseline_models),) + tuple(
        (f"aggregation_round_{index}", models)
        for index, models in enumerate(round_models, start=1)
    ):
        results = []
        for index, rollout in enumerate(evaluation_rollouts, start=1):
            if progress is not None:
                progress(
                    f"[evaluate {name} rollout={index}/{len(evaluation_rollouts)}] "
                    f"attempt={rollout.attempt_id}"
                )
            results.append(_rollout(
                name=name,
                round_index=0,
                model=models[0],
                rollout=rollout,
                predictor=predictor,
                codec=codec,
                evaluation=evaluation,
                measurement=measurement,
                labels=labels,
                spec=spec,
                examples_by_state=examples_by_state,
                collect_states=False,
                progress=progress,
            ))
        scores.append(_score(name, "model_selection", tuple(results)))
    base_training_count = sum(
        item.exposure_role == "training" for item in base_examples
    )
    counts = tuple(
        base_training_count
        + sum(item.round_index <= round_index for item in aggregated_states)
        for round_index in range(1, aggregation_config.rounds + 1)
    )
    return CohortV2AggregationRun(
        CohortV2AggregationResult(tuple(aggregated_states), tuple(scores), counts),
        tuple(round_models),
    )


def _pair_payload(pair: PredictionPair) -> dict[str, object]:
    return {"requested_horizon": pair.delta, "abstraction": str(pair.abstraction)}


def aggregation_result_bytes(result: CohortV2AggregationResult) -> tuple[bytes, bytes]:
    states = b"".join(
        canonical_json_bytes(
            asdict(item)
            | {
                "schema": AGGREGATION_SCHEMA,
                "record_type": "closed_loop_predicted_carrier_state",
                "selected_pair": _pair_payload(item.selected_pair),
                "relabelled_pair": _pair_payload(item.relabelled_pair),
            }
        )
        for item in result.states
    )
    baseline = result.scores[0]
    rounds = []
    for score in result.scores:
        value = asdict(score)
        value["endpoint_mse_change_from_baseline"] = (
            score.mean_terminal_carrier_mse - baseline.mean_terminal_carrier_mse
        )
        value["pair_utility_change_from_baseline"] = (
            score.mean_selected_segment_cost - baseline.mean_selected_segment_cost
        )
        rounds.append(value)
    scores = canonical_json_bytes({
        "schema": AGGREGATION_SCHEMA,
        "round_training_counts": list(result.round_training_counts),
        "scores": rounds,
    })
    return states, scores


def _bytes_identity(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _checkpoint_bytes(run: CohortV2AggregationRun, config: CohortV2ControllerConfig) -> bytes:
    payload = {
        "schema": AGGREGATION_SCHEMA,
        "config": asdict(config),
        "rounds": [
            {
                "round_index": index,
                "joint_model_state": models[0].state_dict(),
                "two_head_model_state": models[1].state_dict(),
            }
            for index, models in enumerate(run.round_models, start=1)
        ],
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def write_cohort_v2_controller_aggregation(
    root: Path,
    run: CohortV2AggregationRun,
    controller_config: CohortV2ControllerConfig,
    aggregation_config: CohortV2AggregationConfig,
    evaluation: CohortV2EvaluationResult,
    measurement: CohortV2MeasurementResult,
    spec: CohortV2TrajectoryCostSpec,
    *,
    source_controller_artifact_identity: str,
    source_controller_checkpoint_identity: str,
    source_predictor_checkpoint_identity: str,
    trajectory_label_artifact_identity: str,
    derivation_index_identity: str,
    implementation_revision: str,
) -> dict[str, object]:
    root = Path(root)
    if root.exists():
        raise CohortV2AggregationError(f"immutable aggregation output already exists: {root}")
    artifacts = aggregation_result_bytes(run.result)
    checkpoint = _checkpoint_bytes(run, controller_config)
    manifest = {
        "artifact_type": "cohort_v2_closed_loop_controller_aggregation",
        "schema": AGGREGATION_SCHEMA,
        "aggregation_config": asdict(aggregation_config),
        "aggregation_config_identity": aggregation_config.identity,
        "capability_declaration_identity": evaluation.capability_declaration_identity,
        "controller_config": asdict(controller_config),
        "derivation_index_identity": derivation_index_identity,
        "evaluation_identity": evaluation.identity,
        "final_evaluation_consumed": False,
        "implementation_revision": implementation_revision,
        "measurement_identity": measurement.identity,
        "objective_identity": evaluation.objective_identity,
        "partition_identity": evaluation.partition_identity,
        "release_identity": evaluation.release_identity,
        "role_permissions": {"aggregation": ["training"], "evaluation": ["model_selection"]},
        "source_cohort_mutated": False,
        "source_controller_artifact_identity": source_controller_artifact_identity,
        "source_controller_checkpoint_identity": source_controller_checkpoint_identity,
        "source_predictor_checkpoint_identity": source_predictor_checkpoint_identity,
        "trajectory_cost_spec_identity": spec.identity,
        "trajectory_label_artifact_identity": trajectory_label_artifact_identity,
        "predicted_carrier_membership": "closed_loop_visited_states_after_first_transition",
        "relabel_authority": "aligned_future_ground_truth_plus_oracle_cost_to_go",
        "controller_inputs": [
            "agent_observation", "declared_intervention", "elapsed_fixed_step_position"
        ],
        "predicted_carrier_is_controller_input": False,
        "artifacts": {
            "states": {
                "path": "aggregation_states.jsonl",
                "identity": _bytes_identity(artifacts[0]),
            },
            "scores": {"path": "scores.json", "identity": _bytes_identity(artifacts[1])},
            "checkpoint": {"path": "checkpoint.pt", "identity": _bytes_identity(checkpoint)},
        },
    }
    manifest["aggregation_artifact_identity"] = identity((
        "cohort-v2-closed-loop-controller-aggregation-v1",
        aggregation_config.identity,
        evaluation.identity,
        measurement.identity,
        spec.identity,
        source_controller_artifact_identity,
        source_predictor_checkpoint_identity,
        implementation_revision,
        tuple(item["identity"] for item in manifest["artifacts"].values()),
    ))
    root.mkdir(parents=True)
    for name, data in (
        ("aggregation_states.jsonl", artifacts[0]),
        ("scores.json", artifacts[1]),
        ("checkpoint.pt", checkpoint),
        ("manifest.json", canonical_json_bytes(manifest)),
    ):
        path = root / name
        with open(path, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    return manifest


def validate_cohort_v2_controller_aggregation(root: Path) -> dict[str, object]:
    root = Path(root)
    try:
        raw = (root / "manifest.json").read_bytes()
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise CohortV2AggregationError(f"cannot load aggregation manifest: {error}") from error
    if manifest.get("schema") != AGGREGATION_SCHEMA or canonical_json_bytes(manifest) != raw:
        raise CohortV2AggregationError("aggregation manifest is malformed or noncanonical")
    for reference in manifest["artifacts"].values():
        data = (root / reference["path"]).read_bytes()
        if _bytes_identity(data) != reference["identity"]:
            raise CohortV2AggregationError(
                "aggregation artifact identity differs from its manifest"
            )
    if (
        manifest.get("final_evaluation_consumed") is not False
        or manifest.get("source_cohort_mutated") is not False
    ):
        raise CohortV2AggregationError("aggregation crossed its declared data boundary")
    return manifest


def load_cohort_v2_aggregated_controllers(
    root: Path,
) -> tuple[
    tuple[tuple[CohortV2JointPairController, CohortV2TwoHeadController], ...],
    CohortV2ControllerConfig,
]:
    """Load every declared aggregation-round controller checkpoint."""
    root = Path(root)
    manifest = validate_cohort_v2_controller_aggregation(root)
    try:
        payload = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=True)
        config = CohortV2ControllerConfig(**payload["config"])
        rounds = payload["rounds"]
    except (OSError, KeyError, TypeError, RuntimeError, ValueError) as error:
        raise CohortV2AggregationError(
            f"cannot load aggregated controller checkpoint: {error}"
        ) from error
    if (
        type(payload) is not dict
        or set(payload) != {"schema", "config", "rounds"}
        or payload["schema"] != AGGREGATION_SCHEMA
        or payload["config"] != manifest["controller_config"]
        or type(rounds) is not list
        or len(rounds) != manifest["aggregation_config"]["rounds"]
    ):
        raise CohortV2AggregationError("aggregated controller checkpoint is malformed")
    models = []
    for index, item in enumerate(rounds, start=1):
        if type(item) is not dict or set(item) != {
            "round_index", "joint_model_state", "two_head_model_state"
        } or item["round_index"] != index:
            raise CohortV2AggregationError("aggregated controller round is malformed")
        joint = CohortV2JointPairController(config)
        two_head = CohortV2TwoHeadController(config)
        try:
            joint.load_state_dict(item["joint_model_state"], strict=True)
            two_head.load_state_dict(item["two_head_model_state"], strict=True)
        except (TypeError, RuntimeError) as error:
            raise CohortV2AggregationError(
                f"aggregated controller model state is invalid: {error}"
            ) from error
        joint.eval()
        two_head.eval()
        models.append((joint, two_head))
    return tuple(models), config


__all__ = [
    "AGGREGATION_SCHEMA",
    "CohortV2AggregatedState",
    "CohortV2AggregationConfig",
    "CohortV2AggregationError",
    "CohortV2AggregationResult",
    "CohortV2AggregationRun",
    "CohortV2ClosedLoopScore",
    "aggregation_result_bytes",
    "load_cohort_v2_aggregated_controllers",
    "run_cohort_v2_controller_aggregation",
    "validate_cohort_v2_controller_aggregation",
    "write_cohort_v2_controller_aggregation",
]
