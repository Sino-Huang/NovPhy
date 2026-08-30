"""CEM/MPC gameplay planning through the Science Birds shot interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import torch
from PIL import Image

from world_model.data.deployment_temporal import (
    AgentObservation,
    TemporalObservationContext,
    TemporalVisualCarrierAdapter,
)
from world_model.model import (
    Abstraction,
    MacroTransitionBatch,
    MacroTransitionInput,
    MicroTransitionBatch,
    MicroTransitionInput,
    PredictionPair,
    TransitionRequest,
)
from world_model.training.cohort_v2_controller import (
    CohortV2ControllerFeatureCodec,
    select_cohort_v2_controller_pairs,
)
from world_model.training.cohort_v2_macro import MACRO_PAIRS
from world_model.training.cohort_v2_measurement import CohortV2ComputeCalibration
from world_model.training.grid_artifacts import canonical_json_bytes


@dataclass(frozen=True, slots=True)
class SlingshotAction:
    """One slingshot-relative drag and optional post-release tap."""

    drag_x: int
    drag_y: int
    tap_time_ms: int

    def to_interface_action(
        self,
        slingshot_anchor: tuple[int, int],
        bounds: "SlingshotActionBounds",
    ) -> dict[str, Any]:
        if not bounds.contains(self):
            raise ValueError("slingshot action is outside the declared bounds")
        return {
            "action_type": "drag_release",
            "coordinate_frame": "slingshot_relative",
            "drag_start": [int(slingshot_anchor[0]), int(slingshot_anchor[1])],
            "drag_release": [self.drag_x, self.drag_y],
            "tapTime": self.tap_time_ms,
            "releaseTime": bounds.release_time_ms,
        }

    @classmethod
    def from_interface_action(cls, value: Mapping[str, Any]) -> "SlingshotAction":
        if value.get("coordinate_frame", "slingshot_relative") != "slingshot_relative":
            raise ValueError("gameplay actions must use slingshot-relative coordinates")
        release = value.get("drag_release")
        if not isinstance(release, (list, tuple)) or len(release) != 2:
            raise ValueError("gameplay action requires a two-value drag_release")
        return cls(
            int(release[0]),
            int(release[1]),
            int(value.get("tapTime", value.get("tap_time", 0))),
        )


@dataclass(frozen=True, slots=True)
class SlingshotActionBounds:
    drag_x: tuple[int, int]
    drag_y: tuple[int, int]
    tap_time_ms: tuple[int, int]
    release_time_ms: int = 600

    def contains(self, action: SlingshotAction) -> bool:
        return (
            self.drag_x[0] <= action.drag_x <= self.drag_x[1]
            and self.drag_y[0] <= action.drag_y <= self.drag_y[1]
            and self.tap_time_ms[0] <= action.tap_time_ms <= self.tap_time_ms[1]
        )


class TerminalStatus(StrEnum):
    ONGOING = "ongoing"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True, slots=True)
class GameplayCostConfig:
    goal_progress_weight: float
    terminal_success_cost: float
    terminal_failure_cost: float
    illegal_action_cost: float
    physical_penalty_weight: float
    rollout_penalty_weight: float
    compute_weight: float
    structure_unstable_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class GameplayCostTerms:
    goal_progress: float
    terminal_status: TerminalStatus
    legal_action: bool
    physical_penalty: float
    rollout_penalty: float
    compute: float
    structure_unstable_probability: float


@dataclass(frozen=True, slots=True)
class GameplayCostBreakdown:
    total: float
    goal_progress_cost: float
    terminal_cost: float
    legal_action_cost: float
    physical_cost: float
    rollout_cost: float
    compute_cost: float
    structure_unstable_cost: float
    structure_unstable_affects_cost: bool


class GameplayCost:
    """Declared additive cost used for every model-based candidate."""

    def __init__(self, config: GameplayCostConfig) -> None:
        self.config = config

    def evaluate(self, terms: GameplayCostTerms) -> GameplayCostBreakdown:
        goal = -self.config.goal_progress_weight * terms.goal_progress
        terminal = {
            TerminalStatus.ONGOING: 0.0,
            TerminalStatus.SUCCESS: self.config.terminal_success_cost,
            TerminalStatus.FAILURE: self.config.terminal_failure_cost,
        }[terms.terminal_status]
        legality = 0.0 if terms.legal_action else self.config.illegal_action_cost
        physical = self.config.physical_penalty_weight * terms.physical_penalty
        rollout = self.config.rollout_penalty_weight * terms.rollout_penalty
        compute = self.config.compute_weight * terms.compute
        unstable = (
            self.config.structure_unstable_weight
            * terms.structure_unstable_probability
        )
        return GameplayCostBreakdown(
            total=goal + terminal + legality + physical + rollout + compute + unstable,
            goal_progress_cost=goal,
            terminal_cost=terminal,
            legal_action_cost=legality,
            physical_cost=physical,
            rollout_cost=rollout,
            compute_cost=compute,
            structure_unstable_cost=unstable,
            structure_unstable_affects_cost=(
                self.config.structure_unstable_weight != 0.0
            ),
        )


@dataclass(frozen=True, slots=True)
class PlanningObservation:
    identity: str
    carrier: torch.Tensor
    pig_slots: tuple[int, ...]
    slingshot_anchor: tuple[int, int]
    agent_rgb: bytes | None = None
    terminal_status: TerminalStatus = TerminalStatus.ONGOING
    parser_diagnostics: Mapping[str, Any] | None = None
    symbols: Any | None = None
    frame_height: int = 480


class VisualPlanningObservationAdapter(TemporalVisualCarrierAdapter):
    """Expose the shared deployment temporal carrier to gameplay planning."""

    carrier_adapter_identity = TemporalVisualCarrierAdapter.identity

    def from_temporal_context(
        self,
        context: TemporalObservationContext,
        *,
        slingshot_anchor: tuple[int, int],
        terminal_status: TerminalStatus,
    ) -> PlanningObservation:
        result = self.build(context)
        with Image.open(BytesIO(context.current.png)) as opened:
            frame_height = opened.height
        pig_slots = tuple(
            index
            for index, value in enumerate(self.model.object_vocabulary)
            if value.startswith("pig:")
        )
        return PlanningObservation(
            identity=context.current.identity,
            carrier=result.tensor,
            pig_slots=pig_slots,
            slingshot_anchor=slingshot_anchor,
            agent_rgb=context.current.png,
            terminal_status=terminal_status,
            parser_diagnostics=result.diagnostics,
            symbols=result.symbols,
            frame_height=frame_height,
        )

    def from_agent_rgb(
        self,
        *,
        identity: str,
        png: bytes,
        slingshot_anchor: tuple[int, int],
        terminal_status: TerminalStatus,
    ) -> PlanningObservation:
        return self.from_temporal_context(
            TemporalObservationContext(
                None,
                AgentObservation(
                    identity,
                    None,
                    None,
                    png,
                    observation_role="agent",
                ),
            ),
            slingshot_anchor=slingshot_anchor,
            terminal_status=terminal_status,
        )


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    actions: tuple[SlingshotAction, ...]
    total_cost: float
    predicted_carriers: tuple[torch.Tensor, ...] = ()
    requested_horizons: tuple[int, ...] = ()
    effective_horizons: tuple[int, ...] = ()
    model_rollout_count: int = 0
    failure: str | None = None
    cost_breakdown: GameplayCostBreakdown | None = None
    model_compute: float = 0.0
    requested_abstractions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PredictedTransition:
    carrier: torch.Tensor
    requested_horizons: tuple[int, ...]
    effective_horizons: tuple[int, ...]
    compute: float
    physical_penalty: float
    rollout_penalty: float
    requested_abstractions: tuple[str, ...] = ()


class ActionConditionedWorldModel(Protocol):
    def rollout(
        self,
        observation: PlanningObservation,
        carrier: torch.Tensor,
        action: SlingshotAction,
    ) -> PredictedTransition: ...


class FrozenCohortV2WorldModel:
    """Frozen issue-15 predictor plus its deployed joint pair controller."""

    def __init__(
        self,
        *,
        predictor: torch.nn.Module,
        pair_controller: torch.nn.Module,
        controller_codec: CohortV2ControllerFeatureCodec,
        compute: CohortV2ComputeCalibration,
        fixed_steps_per_shot: int,
        release_time_ms: int,
        fixed_pair: PredictionPair | None = None,
    ) -> None:
        if fixed_steps_per_shot <= 0:
            raise ValueError("fixed_steps_per_shot must be positive")
        self.predictor = predictor
        self.pair_controller = pair_controller
        self.controller_codec = controller_codec
        self.compute = compute
        self.fixed_steps_per_shot = fixed_steps_per_shot
        self.release_time_ms = release_time_ms
        self.fixed_pair = fixed_pair

    @staticmethod
    def _transition_request(pair, symbols: Any | None) -> TransitionRequest:
        if pair.abstraction is Abstraction.CONTINUOUS:
            return TransitionRequest(pair, None)
        if symbols is None:
            raise ValueError("selected symbolic pair has no deployment parser input")
        if pair.abstraction is Abstraction.MICRO:
            return TransitionRequest(pair, MicroTransitionBatch((
                MicroTransitionInput(
                    symbols.frame_record_identity,
                    symbols.contact,
                    symbols.supports,
                ),
            )))
        return TransitionRequest(pair, MacroTransitionBatch((
            MacroTransitionInput(
                symbols.frame_record_identity,
                symbols.steady_state,
                symbols.structure_unstable,
            ),
        )))

    def _decision_compute(self, pair, symbols: Any | None) -> float:
        adapters = {
            Abstraction.CONTINUOUS: self.compute.continuous_adapter_per_decision,
            Abstraction.MICRO: self.compute.micro_adapter_per_decision,
            Abstraction.MACRO: self.compute.macro_adapter_per_decision,
        }
        readouts = {
            Abstraction.CONTINUOUS: self.compute.continuous_readout_per_decision,
            Abstraction.MICRO: self.compute.micro_readout_per_decision,
            Abstraction.MACRO: self.compute.macro_readout_per_decision,
        }
        graph = 0.0
        if pair.abstraction is Abstraction.MICRO and symbols is not None:
            contact = symbols.contact.relations or ()
            supports = symbols.supports.relations or ()
            entities = {entity for relation in (*contact, *supports) for entity in relation}
            graph = (
                self.compute.micro_graph_base_per_decision
                + len(entities) * self.compute.micro_graph_per_entity
                + len(contact) * self.compute.micro_graph_per_contact
                + len(supports) * self.compute.micro_graph_per_support
            )
        return (
            (
                0.0
                if self.fixed_pair is not None
                else self.compute.controller_per_decision
            )
            + adapters[pair.abstraction]
            + graph
            + self.compute.transition_per_decision
            + readouts[pair.abstraction]
        )

    def rollout(
        self,
        observation: PlanningObservation,
        carrier: torch.Tensor,
        action: SlingshotAction,
    ) -> PredictedTransition:
        if observation.agent_rgb is None:
            raise ValueError("frozen controller requires a deployment RGB observation")
        if self.fixed_pair is None:
            features = self.controller_codec.encode(
                observation.agent_rgb,
                elapsed_fixed_steps=0,
                intervention={"interface_action": {
                    "drag_release": (action.drag_x, action.drag_y),
                    "frame_height": observation.frame_height,
                    "releaseTime": self.release_time_ms,
                    "tapTime": action.tap_time_ms,
                }},
            ).unsqueeze(0)
            pair = select_cohort_v2_controller_pairs(
                "joint_pair", self.pair_controller, features, MACRO_PAIRS
            )[0]
        else:
            pair = self.fixed_pair
        request = self._transition_request(pair, observation.symbols)
        device = next(self.predictor.parameters()).device
        action_tensor = torch.tensor((
            action.drag_x / float(observation.frame_height),
            action.drag_y / float(observation.frame_height),
            self.release_time_ms / 1000.0,
            action.tap_time_ms / 1000.0,
            1.0,
        ), dtype=torch.float32, device=device).unsqueeze(0)
        current = carrier.to(device).unsqueeze(0)
        requested = []
        effective = []
        abstractions = []
        total_compute = 0.0
        physical_penalty = 0.0
        rollout_penalty = 0.0
        elapsed = 0
        with torch.no_grad():
            while elapsed < self.fixed_steps_per_shot:
                step = min(pair.delta, self.fixed_steps_per_shot - elapsed)
                predicted = self.predictor.carrier(current, action_tensor, request)
                if not bool(torch.isfinite(predicted).all()):
                    raise RuntimeError("world model produced a nonfinite carrier")
                rollout_penalty += float((predicted - current).pow(2).mean())
                physical_penalty += float(
                    torch.relu(predicted.abs() - 2.0).pow(2).mean()
                )
                current = predicted
                elapsed += step
                requested.append(pair.delta)
                effective.append(step)
                abstractions.append(pair.abstraction.value)
                total_compute += self._decision_compute(pair, observation.symbols)
        return PredictedTransition(
            carrier=current.squeeze(0).detach().cpu(),
            requested_horizons=tuple(requested),
            effective_horizons=tuple(effective),
            compute=total_compute,
            physical_penalty=physical_penalty,
            rollout_penalty=rollout_penalty,
            requested_abstractions=tuple(abstractions),
        )


class ContinuousCheckpointWorldModel:
    """Run a retrained continuous checkpoint under one fixed or adaptive horizon."""

    def __init__(
        self,
        *,
        predictor: torch.nn.Module,
        fixed_steps_per_shot: int,
        release_time_ms: int,
        transition_compute: float,
        fixed_horizon: int | None = None,
        horizon_selector: Callable[[PlanningObservation, SlingshotAction], int] | None = None,
        controller_compute: float = 0.0,
    ) -> None:
        if (
            fixed_steps_per_shot <= 0
            or release_time_ms < 0
            or not math.isfinite(transition_compute)
            or transition_compute < 0.0
            or not math.isfinite(controller_compute)
            or controller_compute < 0.0
            or (fixed_horizon is None) == (horizon_selector is None)
            or (fixed_horizon is not None and fixed_horizon not in (1, 15))
        ):
            raise ValueError("continuous checkpoint world-model configuration is invalid")
        self.predictor = predictor
        self.fixed_steps_per_shot = fixed_steps_per_shot
        self.release_time_ms = release_time_ms
        self.transition_compute = float(transition_compute)
        self.fixed_horizon = fixed_horizon
        self.horizon_selector = horizon_selector
        self.controller_compute = float(controller_compute)

    def rollout(
        self,
        observation: PlanningObservation,
        carrier: torch.Tensor,
        action: SlingshotAction,
    ) -> PredictedTransition:
        horizon = (
            self.fixed_horizon
            if self.fixed_horizon is not None
            else self.horizon_selector(observation, action)  # type: ignore[misc]
        )
        if horizon not in (1, 15):
            raise ValueError("adaptive continuous horizon must be 1 or 15")
        pair = PredictionPair(horizon, Abstraction.CONTINUOUS)
        device = next(self.predictor.parameters()).device
        action_tensor = torch.tensor((
            action.drag_x / float(observation.frame_height),
            action.drag_y / float(observation.frame_height),
            self.release_time_ms / 1000.0,
            action.tap_time_ms / 1000.0,
            1.0,
        ), dtype=torch.float32, device=device).unsqueeze(0)
        current = carrier.to(device).unsqueeze(0)
        requested = []
        effective = []
        rollout_penalty = 0.0
        physical_penalty = 0.0
        elapsed = 0
        with torch.no_grad():
            while elapsed < self.fixed_steps_per_shot:
                step = min(horizon, self.fixed_steps_per_shot - elapsed)
                predicted = self.predictor.carrier(current, action_tensor, pair)
                if not bool(torch.isfinite(predicted).all()):
                    raise RuntimeError("world model produced a nonfinite carrier")
                rollout_penalty += float((predicted - current).pow(2).mean())
                physical_penalty += float(
                    torch.relu(predicted.abs() - 2.0).pow(2).mean()
                )
                current = predicted
                elapsed += step
                requested.append(horizon)
                effective.append(step)
        return PredictedTransition(
            carrier=current.squeeze(0).detach().cpu(),
            requested_horizons=tuple(requested),
            effective_horizons=tuple(effective),
            compute=(
                len(requested) * self.transition_compute
                + (self.controller_compute if self.fixed_horizon is None else 0.0)
            ),
            physical_penalty=physical_penalty,
            rollout_penalty=rollout_penalty,
            requested_abstractions=("continuous",) * len(requested),
        )


class WorldModelCandidateEvaluator:
    """Evaluate complete shot sequences without reading future engine state."""

    def __init__(
        self,
        model: ActionConditionedWorldModel,
        bounds: SlingshotActionBounds,
        cost: GameplayCost,
        *,
        pig_success_threshold: float = 0.1,
    ) -> None:
        self.model = model
        self.bounds = bounds
        self.cost = cost
        self.pig_success_threshold = pig_success_threshold

    @staticmethod
    def _pig_activity(carrier: torch.Tensor, slots: tuple[int, ...]) -> float:
        values = [
            float(carrier[2 + 13 * slot].detach().cpu().clamp(0.0, 1.0))
            for slot in slots
        ]
        return sum(values)

    def evaluate(
        self,
        observation: PlanningObservation,
        actions: tuple[SlingshotAction, ...],
    ) -> CandidateEvaluation:
        legal = bool(actions) and all(self.bounds.contains(action) for action in actions)
        unstable = float(
            (observation.parser_diagnostics or {}).get(
                "structure_unstable_probability", 0.0
            )
        )
        if not legal:
            breakdown = self.cost.evaluate(GameplayCostTerms(
                goal_progress=0.0,
                terminal_status=TerminalStatus.ONGOING,
                legal_action=False,
                physical_penalty=0.0,
                rollout_penalty=0.0,
                compute=0.0,
                structure_unstable_probability=unstable,
            ))
            return CandidateEvaluation(
                actions=actions,
                total_cost=breakdown.total,
                failure="illegal_action",
                cost_breakdown=breakdown,
            )

        initial_activity = self._pig_activity(
            observation.carrier, observation.pig_slots
        )
        current = observation.carrier
        carriers = []
        requested = []
        effective = []
        abstractions = []
        compute = 0.0
        physical = 0.0
        rollout = 0.0
        rollout_count = 0
        for action in actions:
            transition = self.model.rollout(observation, current, action)
            current = transition.carrier
            carriers.append(current)
            requested.extend(transition.requested_horizons)
            effective.extend(transition.effective_horizons)
            abstractions.extend(transition.requested_abstractions)
            compute += transition.compute
            physical += transition.physical_penalty
            rollout += transition.rollout_penalty
            rollout_count += len(transition.requested_horizons)
        final_activity = self._pig_activity(current, observation.pig_slots)
        progress = (
            (initial_activity - final_activity) / initial_activity
            if initial_activity > 0.0
            else 0.0
        )
        terminal = (
            TerminalStatus.SUCCESS
            if final_activity <= self.pig_success_threshold
            else TerminalStatus.ONGOING
        )
        breakdown = self.cost.evaluate(GameplayCostTerms(
            goal_progress=progress,
            terminal_status=terminal,
            legal_action=True,
            physical_penalty=physical,
            rollout_penalty=rollout,
            compute=compute,
            structure_unstable_probability=unstable,
        ))
        return CandidateEvaluation(
            actions=actions,
            total_cost=breakdown.total,
            predicted_carriers=tuple(carriers),
            requested_horizons=tuple(requested),
            effective_horizons=tuple(effective),
            model_rollout_count=rollout_count,
            cost_breakdown=breakdown,
            model_compute=compute,
            requested_abstractions=tuple(abstractions),
        )


class CandidateEvaluator(Protocol):
    def evaluate(
        self,
        observation: PlanningObservation,
        actions: tuple[SlingshotAction, ...],
    ) -> CandidateEvaluation: ...


@dataclass(frozen=True, slots=True)
class CEMConfig:
    population_size: int
    elite_count: int
    iterations: int
    sequence_length: int
    seed: int
    minimum_std: float = 1.0

    def __post_init__(self) -> None:
        if (
            self.population_size <= 0
            or not 0 < self.elite_count <= self.population_size
            or self.iterations <= 0
            or self.sequence_length <= 0
            or self.seed < 0
            or self.minimum_std <= 0.0
        ):
            raise ValueError("CEM configuration is invalid")


@dataclass(frozen=True, slots=True)
class CEMIteration:
    index: int
    candidate_costs: tuple[float, ...]
    elite_indices: tuple[int, ...]
    updated_mean: tuple[tuple[float, float, float], ...]
    updated_std: tuple[tuple[float, float, float], ...]
    invalid_candidate_count: int
    candidate_failures: tuple[str | None, ...] = ()


@dataclass(frozen=True, slots=True)
class PlanResult:
    planner_id: str
    seed: int
    actions: tuple[SlingshotAction, ...]
    selected_evaluation: CandidateEvaluation
    iterations: tuple[CEMIteration, ...] = ()
    candidate_count: int = 0
    invalid_candidate_count: int = 0
    model_rollout_count: int = 0
    planner_compute: float = 0.0
    goal_evaluation_count: int = 0
    wall_clock_seconds: float = 0.0


class CEMPlanner:
    """Seeded Cross-Entropy Method over bounded shot sequences."""

    planner_id = "cem_world_model"

    def __init__(
        self,
        config: CEMConfig,
        bounds: SlingshotActionBounds,
        evaluator: CandidateEvaluator,
        *,
        progress: Any | None = None,
    ) -> None:
        self.config = config
        self.bounds = bounds
        self.evaluator = evaluator
        self.progress = progress
        self._plan_index = 0

    def plan(self, observation: PlanningObservation) -> PlanResult:
        started = time.monotonic()
        seed = self.config.seed + self._plan_index
        self._plan_index += 1
        rng = np.random.default_rng(seed)
        lower = np.asarray((
            self.bounds.drag_x[0],
            self.bounds.drag_y[0],
            self.bounds.tap_time_ms[0],
        ), dtype=np.float64)
        upper = np.asarray((
            self.bounds.drag_x[1],
            self.bounds.drag_y[1],
            self.bounds.tap_time_ms[1],
        ), dtype=np.float64)
        mean = np.broadcast_to((lower + upper) / 2.0, (self.config.sequence_length, 3)).copy()
        std = np.broadcast_to((upper - lower) / 2.0, mean.shape).copy()
        summaries = []
        selected: CandidateEvaluation | None = None
        total_invalid = 0
        total_model_rollouts = 0
        total_model_compute = 0.0
        for iteration in range(self.config.iterations):
            if self.progress is not None:
                self.progress(
                    f"[cem seed={seed}] iteration {iteration + 1}/{self.config.iterations} "
                    f"population={self.config.population_size}"
                )
            sampled = rng.normal(
                mean,
                std,
                size=(self.config.population_size, self.config.sequence_length, 3),
            )
            sampled = np.rint(np.clip(sampled, lower, upper)).astype(np.int64)
            evaluations = []
            costs = []
            invalid = 0
            progress_every = max(1, self.config.population_size // 4)
            for candidate_index, row in enumerate(sampled, start=1):
                actions = tuple(
                    SlingshotAction(int(value[0]), int(value[1]), int(value[2]))
                    for value in row
                )
                try:
                    evaluation = self.evaluator.evaluate(observation, actions)
                except Exception as error:
                    evaluation = CandidateEvaluation(
                        actions=actions,
                        total_cost=math.inf,
                        failure=f"{type(error).__name__}: {error}",
                    )
                selection_cost = evaluation.total_cost
                if evaluation.failure is not None or not math.isfinite(selection_cost):
                    invalid += 1
                    selection_cost = math.inf
                evaluations.append(evaluation)
                costs.append(selection_cost)
                total_model_rollouts += evaluation.model_rollout_count
                total_model_compute += evaluation.model_compute
                if self.progress is not None and (
                    candidate_index % progress_every == 0
                    or candidate_index == self.config.population_size
                ):
                    self.progress(
                        f"[cem seed={seed}] iteration {iteration + 1}/{self.config.iterations} "
                        f"candidate {candidate_index}/{self.config.population_size}"
                    )
            elite_indices = tuple(
                int(value)
                for value in np.argsort(np.asarray(costs), kind="stable")[: self.config.elite_count]
            )
            if not all(math.isfinite(costs[index]) for index in elite_indices):
                raise RuntimeError("CEM has too few valid candidates for elite selection")
            elites = sampled[np.asarray(elite_indices)]
            mean = elites.mean(axis=0)
            std = np.maximum(elites.std(axis=0), self.config.minimum_std)
            selected = evaluations[elite_indices[0]]
            total_invalid += invalid
            summaries.append(CEMIteration(
                index=iteration + 1,
                candidate_costs=tuple(float(value) for value in costs),
                elite_indices=elite_indices,
                updated_mean=tuple(tuple(float(value) for value in row) for row in mean),
                updated_std=tuple(tuple(float(value) for value in row) for row in std),
                invalid_candidate_count=invalid,
                candidate_failures=tuple(value.failure for value in evaluations),
            ))
            if self.progress is not None:
                self.progress(
                    f"[cem seed={seed}] iteration {iteration + 1} complete "
                    f"elite_cost={costs[elite_indices[0]]:.6f} invalid={invalid}"
                )
        assert selected is not None
        return PlanResult(
            planner_id=self.planner_id,
            seed=seed,
            actions=selected.actions,
            selected_evaluation=selected,
            iterations=tuple(summaries),
            candidate_count=self.config.population_size * self.config.iterations,
            invalid_candidate_count=total_invalid,
            model_rollout_count=total_model_rollouts,
            planner_compute=total_model_compute,
            goal_evaluation_count=self.config.population_size * self.config.iterations,
            wall_clock_seconds=time.monotonic() - started,
        )


class RandomLegalPlanner:
    planner_id = "random_legal"

    def __init__(
        self,
        bounds: SlingshotActionBounds,
        *,
        sequence_length: int,
        seed: int,
    ) -> None:
        self.bounds = bounds
        self.sequence_length = sequence_length
        self.seed = seed
        self._plan_index = 0

    def plan(self, observation: PlanningObservation) -> PlanResult:
        started = time.monotonic()
        seed = self.seed + self._plan_index
        self._plan_index += 1
        rng = np.random.default_rng(seed)
        actions = tuple(
            SlingshotAction(
                int(rng.integers(self.bounds.drag_x[0], self.bounds.drag_x[1] + 1)),
                int(rng.integers(self.bounds.drag_y[0], self.bounds.drag_y[1] + 1)),
                int(rng.integers(
                    self.bounds.tap_time_ms[0], self.bounds.tap_time_ms[1] + 1
                )),
            )
            for _ in range(self.sequence_length)
        )
        evaluation = CandidateEvaluation(actions, 0.0)
        return PlanResult(
            self.planner_id,
            seed,
            actions,
            evaluation,
            candidate_count=len(actions),
            wall_clock_seconds=time.monotonic() - started,
        )


class HeuristicNoModelPlanner:
    """Declared medium-strength, rising-angle no-model shot baseline."""

    planner_id = "heuristic_no_model"

    def __init__(
        self,
        bounds: SlingshotActionBounds,
        *,
        sequence_length: int,
    ) -> None:
        self.bounds = bounds
        self.sequence_length = sequence_length
        self._plan_index = 0

    def plan(self, observation: PlanningObservation) -> PlanResult:
        started = time.monotonic()
        self._plan_index += 1
        drag_x = round((self.bounds.drag_x[0] + self.bounds.drag_x[1]) / 2)
        drag_y = round((self.bounds.drag_y[0] + self.bounds.drag_y[1]) / 2)
        tap = round((self.bounds.tap_time_ms[0] + self.bounds.tap_time_ms[1]) / 2)
        actions = tuple(
            SlingshotAction(drag_x, drag_y, tap)
            for _ in range(self.sequence_length)
        )
        evaluation = CandidateEvaluation(actions, 0.0)
        return PlanResult(
            self.planner_id,
            self._plan_index,
            actions,
            evaluation,
            candidate_count=len(actions),
            wall_clock_seconds=time.monotonic() - started,
        )


class ControlMode(StrEnum):
    OPEN_LOOP = "open_loop"
    MPC = "mpc"


@dataclass(frozen=True, slots=True)
class ControlConfig:
    mode: ControlMode
    max_shots: int
    max_planner_compute: float
    max_wall_clock_seconds: float | None = None

    def __post_init__(self) -> None:
        if (
            type(self.mode) is not ControlMode
            or self.max_shots <= 0
            or not math.isfinite(self.max_planner_compute)
            or self.max_planner_compute <= 0.0
            or (
                self.max_wall_clock_seconds is not None
                and (
                    not math.isfinite(self.max_wall_clock_seconds)
                    or self.max_wall_clock_seconds <= 0.0
                )
            )
        ):
            raise ValueError("gameplay control configuration is invalid")


class GameplayEnvironment(Protocol):
    def observe(self) -> PlanningObservation: ...

    def execute(self, action: SlingshotAction) -> PlanningObservation: ...


class GameplayPlanner(Protocol):
    planner_id: str

    def plan(self, observation: PlanningObservation) -> PlanResult: ...


@dataclass(frozen=True, slots=True)
class ControlStep:
    shot_index: int
    observation_before: str
    observation_after: str
    plan: PlanResult
    executed_action: SlingshotAction
    recursive_rollout_error: float | None
    observation_before_diagnostics: Mapping[str, Any] | None = None
    observation_after_diagnostics: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ControlResult:
    mode: ControlMode
    planner_id: str
    steps: tuple[ControlStep, ...]
    termination_reason: str
    success: bool
    replan_count: int
    candidate_count: int
    invalid_candidate_count: int
    model_rollout_count: int
    planner_compute: float
    goal_evaluation_count: int
    planner_wall_clock_seconds: float
    game_interface_wall_clock_seconds: float
    wall_clock_seconds: float
    failures: tuple[str, ...]


def run_gameplay_control(
    planner: GameplayPlanner,
    environment: GameplayEnvironment,
    config: ControlConfig,
    *,
    progress: Any | None = None,
) -> ControlResult:
    """Execute an open-loop sequence or one-shot receding-horizon MPC."""
    started = time.monotonic()
    steps = []
    failures = []
    replans = 0
    candidate_count = 0
    invalid_count = 0
    model_rollouts = 0
    planner_compute = 0.0
    goal_evaluations = 0
    planner_wall_clock = 0.0
    game_interface_wall_clock = 0.0
    termination = "shot_limit"
    interface_started = time.monotonic()
    try:
        observation = environment.observe()
    except Exception as error:
        failures.append(f"game_interface_observe: {type(error).__name__}: {error}")
        observation = None
        termination = "game_interface_failure"
    game_interface_wall_clock += time.monotonic() - interface_started

    while observation is not None and len(steps) < config.max_shots:
        if (
            config.max_wall_clock_seconds is not None
            and time.monotonic() - started >= config.max_wall_clock_seconds
        ):
            termination = "timeout"
            break
        if observation.terminal_status is TerminalStatus.SUCCESS:
            termination = "success"
            break
        if observation.terminal_status is TerminalStatus.FAILURE:
            termination = "terminal_failure"
            break
        try:
            plan = planner.plan(observation)
        except Exception as error:
            failures.append(f"planner: {type(error).__name__}: {error}")
            termination = "planner_failure"
            break
        replans += 1
        candidate_count += plan.candidate_count
        invalid_count += plan.invalid_candidate_count
        model_rollouts += plan.model_rollout_count
        planner_compute += plan.planner_compute
        goal_evaluations += plan.goal_evaluation_count
        planner_wall_clock += plan.wall_clock_seconds
        if (
            config.max_wall_clock_seconds is not None
            and time.monotonic() - started >= config.max_wall_clock_seconds
        ):
            termination = "timeout"
            break
        if progress is not None:
            progress(
                f"[plan {replans}] mode={config.mode} candidates={plan.candidate_count} "
                f"model_rollouts={plan.model_rollout_count} selected_cost="
                f"{plan.selected_evaluation.total_cost:.6f}"
            )
        if planner_compute > config.max_planner_compute:
            termination = "compute_limit"
            break
        planned_actions = plan.actions if config.mode is ControlMode.OPEN_LOOP else plan.actions[:1]
        if not planned_actions:
            failures.append("planner: selected no actions")
            termination = "planner_failure"
            break
        for action_index, action in enumerate(planned_actions):
            if len(steps) >= config.max_shots:
                termination = "shot_limit"
                break
            before = observation
            interface_started = time.monotonic()
            try:
                observation = environment.execute(action)
            except Exception as error:
                failures.append(f"game_interface_execute: {type(error).__name__}: {error}")
                termination = "game_interface_failure"
                observation = None
                game_interface_wall_clock += time.monotonic() - interface_started
                break
            game_interface_wall_clock += time.monotonic() - interface_started
            error_value = None
            predicted = plan.selected_evaluation.predicted_carriers
            if action_index < len(predicted) and predicted[action_index].shape == observation.carrier.shape:
                error_value = float(
                    (predicted[action_index] - observation.carrier).pow(2).mean()
                )
            steps.append(ControlStep(
                shot_index=len(steps) + 1,
                observation_before=before.identity,
                observation_after=observation.identity,
                plan=plan,
                executed_action=action,
                recursive_rollout_error=error_value,
                observation_before_diagnostics=before.parser_diagnostics,
                observation_after_diagnostics=observation.parser_diagnostics,
            ))
            if (
                config.max_wall_clock_seconds is not None
                and time.monotonic() - started >= config.max_wall_clock_seconds
            ):
                termination = "timeout"
                break
            if progress is not None:
                progress(
                    f"[shot {len(steps)}/{config.max_shots}] action={action} "
                    f"state={observation.terminal_status} rollout_error={error_value}"
                )
            if observation.terminal_status is TerminalStatus.SUCCESS:
                termination = "success"
                break
            if observation.terminal_status is TerminalStatus.FAILURE:
                termination = "terminal_failure"
                break
        if observation is None or termination in {
            "success", "terminal_failure", "game_interface_failure", "timeout"
        }:
            break
        if config.mode is ControlMode.OPEN_LOOP:
            termination = "open_loop_sequence_complete"
            break

    return ControlResult(
        mode=config.mode,
        planner_id=planner.planner_id,
        steps=tuple(steps),
        termination_reason=termination,
        success=termination == "success",
        replan_count=replans,
        candidate_count=candidate_count,
        invalid_candidate_count=invalid_count,
        model_rollout_count=model_rollouts,
        planner_compute=planner_compute,
        goal_evaluation_count=goal_evaluations,
        planner_wall_clock_seconds=planner_wall_clock,
        game_interface_wall_clock_seconds=game_interface_wall_clock,
        wall_clock_seconds=time.monotonic() - started,
        failures=tuple(failures),
    )


@dataclass(frozen=True, slots=True)
class GameplayEvidenceBindings:
    implementation_revision: str
    world_model_checkpoint_identity: str
    controller_checkpoint_identity: str
    visual_parser_checkpoint_identity: str
    observation_adapter_identity: str
    goal_cost_version: str
    goal_cost_config: GameplayCostConfig
    action_bounds: SlingshotActionBounds
    cem_config: CEMConfig
    control_config: ControlConfig
    seed: int
    level_identity: str
    environment_version: str
    rerun_commands: tuple[str, ...] = ()


def _action_payload(action: SlingshotAction) -> dict[str, int]:
    return {
        "drag_x": action.drag_x,
        "drag_y": action.drag_y,
        "tap_time_ms": action.tap_time_ms,
    }


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _bindings_payload(bindings: GameplayEvidenceBindings) -> dict[str, Any]:
    return {
        "implementation_revision": bindings.implementation_revision,
        "world_model_checkpoint_identity": bindings.world_model_checkpoint_identity,
        "controller_checkpoint_identity": bindings.controller_checkpoint_identity,
        "visual_parser_checkpoint_identity": bindings.visual_parser_checkpoint_identity,
        "observation_adapter_identity": bindings.observation_adapter_identity,
        "goal_cost_version": bindings.goal_cost_version,
        "goal_cost_config": asdict(bindings.goal_cost_config),
        "action_bounds": asdict(bindings.action_bounds),
        "cem_config": asdict(bindings.cem_config),
        "control_config": {
            "mode": str(bindings.control_config.mode),
            "max_shots": bindings.control_config.max_shots,
            "max_planner_compute": bindings.control_config.max_planner_compute,
            "max_wall_clock_seconds": bindings.control_config.max_wall_clock_seconds,
        },
        "seed": bindings.seed,
        "level_identity": bindings.level_identity,
        "environment_version": bindings.environment_version,
        "rerun_commands": list(bindings.rerun_commands),
    }


def _plan_payload(plan: PlanResult) -> dict[str, Any]:
    selected = plan.selected_evaluation
    return {
        "planner_id": plan.planner_id,
        "seed": plan.seed,
        "selected_actions": [_action_payload(value) for value in plan.actions],
        "selected_cost": selected.total_cost,
        "selected_cost_breakdown": (
            None if selected.cost_breakdown is None else asdict(selected.cost_breakdown)
        ),
        "requested_horizons": list(selected.requested_horizons),
        "effective_horizons": list(selected.effective_horizons),
        "requested_abstractions": list(selected.requested_abstractions),
        "selected_model_rollout_count": selected.model_rollout_count,
        "candidate_count": plan.candidate_count,
        "invalid_candidate_count": plan.invalid_candidate_count,
        "model_rollout_count": plan.model_rollout_count,
        "planner_compute": plan.planner_compute,
        "goal_evaluation_count": plan.goal_evaluation_count,
        "wall_clock_seconds": plan.wall_clock_seconds,
        "iterations": [
            {
                "index": item.index,
                "candidate_costs": [
                    _finite_or_none(value) for value in item.candidate_costs
                ],
                "elite_indices": list(item.elite_indices),
                "updated_mean": [list(row) for row in item.updated_mean],
                "updated_std": [list(row) for row in item.updated_std],
                "invalid_candidate_count": item.invalid_candidate_count,
                "candidate_failures": list(item.candidate_failures),
            }
            for item in plan.iterations
        ],
    }


def _result_payload(result: ControlResult) -> dict[str, Any]:
    errors = [
        item.recursive_rollout_error
        for item in result.steps
        if item.recursive_rollout_error is not None
    ]
    return {
        "mode": str(result.mode),
        "planner_id": result.planner_id,
        "termination_reason": result.termination_reason,
        "success": result.success,
        "replan_count": result.replan_count,
        "executed_shot_count": len(result.steps),
        "candidate_count": result.candidate_count,
        "invalid_candidate_count": result.invalid_candidate_count,
        "model_rollout_count": result.model_rollout_count,
        "planner_compute": result.planner_compute,
        "goal_evaluation_count": result.goal_evaluation_count,
        "planner_wall_clock_seconds": result.planner_wall_clock_seconds,
        "game_interface_wall_clock_seconds": result.game_interface_wall_clock_seconds,
        "wall_clock_seconds": result.wall_clock_seconds,
        "failures": list(result.failures),
        "accumulated_recursive_rollout_error": sum(errors),
        "steps": [
            {
                "shot_index": item.shot_index,
                "observation_before": item.observation_before,
                "observation_after": item.observation_after,
                "observation_before_diagnostics": item.observation_before_diagnostics,
                "observation_after_diagnostics": item.observation_after_diagnostics,
                "executed_action": _action_payload(item.executed_action),
                "recursive_rollout_error": item.recursive_rollout_error,
                "requested_horizons": list(
                    item.plan.selected_evaluation.requested_horizons
                ),
                "effective_horizons": list(
                    item.plan.selected_evaluation.effective_horizons
                ),
                "requested_abstractions": list(
                    item.plan.selected_evaluation.requested_abstractions
                ),
                "plan": _plan_payload(item.plan),
            }
            for item in result.steps
        ],
    }


def _evidence_document(
    result: ControlResult,
    bindings: GameplayEvidenceBindings,
) -> dict[str, Any]:
    payload = {
        "schema": "cohort_v2_gameplay_planning_evidence_v1",
        "source_bindings": _bindings_payload(bindings),
        "control_result": _result_payload(result),
    }
    artifact_identity = "cohort-v2-gameplay-planning-evidence-v1:sha256:" + hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return {**payload, "artifact_identity": artifact_identity}


def write_gameplay_evidence(
    root: Path,
    result: ControlResult,
    bindings: GameplayEvidenceBindings,
) -> dict[str, Any]:
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    document = _evidence_document(result, bindings)
    path = target / "evidence.json"
    temporary = path.with_name(path.name + ".tmp")
    encoded = canonical_json_bytes(document)
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return json.loads(encoded)


def validate_gameplay_evidence(
    root: Path,
    *,
    expected_bindings: GameplayEvidenceBindings | None = None,
) -> dict[str, Any]:
    path = Path(root) / "evidence.json"
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"gameplay evidence is unreadable: {error}") from error
    if canonical_json_bytes(document) != raw:
        raise ValueError("gameplay evidence is not canonical")
    payload = {key: value for key, value in document.items() if key != "artifact_identity"}
    expected_identity = "cohort-v2-gameplay-planning-evidence-v1:sha256:" + hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    if (
        document.get("schema") != "cohort_v2_gameplay_planning_evidence_v1"
        or document.get("artifact_identity") != expected_identity
        or (
            expected_bindings is not None
            and canonical_json_bytes(document.get("source_bindings"))
            != canonical_json_bytes(_bindings_payload(expected_bindings))
        )
    ):
        raise ValueError("gameplay evidence source binding or identity differs")
    return document
