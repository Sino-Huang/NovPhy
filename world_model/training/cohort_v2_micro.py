"""Oracle micro-relation training and scoring for the cohort-v2 pair grid."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
from torch import nn
from torch.nn import functional as F

from world_model.data import (
    CohortV2CentralFrameRecord,
    CohortV2OracleWindow,
    CohortV2OracleWindowDataset,
    CohortV2ReleaseReader,
)
from world_model.data.cohort_v2 import (
    CAPABILITY_DECLARATION_IDENTITY,
    COHORT_V2_RELEASE_IDENTITY,
    MICRO_SPEC_IDENTITY,
)
from world_model.model import (
    Abstraction,
    DualOutputPredictor,
    MicroReadoutHead,
    PredictionPair,
    PredictorConfig,
    RelationTransitionValue,
    identity,
)
from world_model.training.cohort_v2 import build_cohort_v2_transition_request
from world_model.training.cohort_v2_evaluation import (
    COHORT_V2_HORIZONS,
    CohortV2EvaluationResult,
)
from world_model.training.grid_artifacts import canonical_json_bytes


MICRO_CHECKPOINT_SCHEMA: Final = "cohort_v2_micro_checkpoint_v2"
MICRO_FRONTIER_SCHEMA: Final = "cohort_v2_micro_frontier_input_v1"
MICRO_RELATION_AUTHORITY: Final = MICRO_SPEC_IDENTITY
MICRO_CAPABILITIES: Final = frozenset({
    "transition.continuous",
    "transition.micro",
})
MICRO_PAIRS: Final = tuple(
    PredictionPair(horizon, abstraction)
    for horizon in COHORT_V2_HORIZONS
    for abstraction in (Abstraction.CONTINUOUS, Abstraction.MICRO)
)
MICRO_PREDICATES: Final = ("contact", "supports")
STATE_FEATURES_PER_ENTITY: Final = 13


class CohortV2MicroError(ValueError):
    """The oracle micro training, checkpoint, or frontier input is invalid."""


@dataclass(frozen=True, slots=True)
class CohortV2MicroConfig:
    seed: int = 20260824
    steps: int = 1200
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    symbolic_weight: float = 1.0
    latent_dim: int = 192
    hidden_dim: int = 384
    depth: int = 3
    max_entities: int = 12
    device: str = "cuda"

    def __post_init__(self) -> None:
        for field in (
            "seed", "steps", "batch_size", "latent_dim", "hidden_dim",
            "depth", "max_entities",
        ):
            value = getattr(self, field)
            minimum = 0 if field == "seed" else 1
            if type(value) is not int or value < minimum:
                raise CohortV2MicroError(f"{field} must be an integer >= {minimum}")
        if self.latent_dim < 2 + self.max_entities * STATE_FEATURES_PER_ENTITY:
            raise CohortV2MicroError("latent_dim cannot hold the declared entity slots")
        if (
            self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or not 0.0 <= self.grad_clip <= 1e3
            or self.symbolic_weight < 0.0
        ):
            raise CohortV2MicroError("optimizer or symbolic-loss configuration is invalid")
        if type(self.device) is not str or not self.device.strip():
            raise CohortV2MicroError("device must be nonempty")

    @property
    def predictor_config(self) -> PredictorConfig:
        return PredictorConfig(
            latent_dim=self.latent_dim,
            action_dim=5,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
        )

    @property
    def identity(self) -> str:
        return identity((
            "cohort-v2-micro-training-config-v1",
            self.seed,
            self.steps,
            self.batch_size,
            self.learning_rate,
            self.weight_decay,
            self.grad_clip,
            self.symbolic_weight,
            self.latent_dim,
            self.hidden_dim,
            self.depth,
            self.max_entities,
            tuple(pair.identity for pair in MICRO_PAIRS),
        ))


class CohortV2StateCodec:
    """Map an oracle engine frame record to the fixed-width continuous carrier."""

    def __init__(self, *, latent_dim: int, max_entities: int) -> None:
        if (
            type(latent_dim) is not int
            or type(max_entities) is not int
            or max_entities <= 0
            or latent_dim < 2 + max_entities * STATE_FEATURES_PER_ENTITY
        ):
            raise CohortV2MicroError("state codec dimensions are invalid")
        self.latent_dim = latent_dim
        self.max_entities = max_entities

    @property
    def identity(self) -> str:
        return identity((
            "cohort-v2-oracle-continuous-carrier-v1",
            self.latent_dim,
            self.max_entities,
            STATE_FEATURES_PER_ENTITY,
            "scenario-object-id-order",
        ))

    @staticmethod
    def _number(value: Any, field: str) -> float:
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise CohortV2MicroError(f"engine state {field} is not finite")
        return float(value)

    def encode(self, frame: CohortV2CentralFrameRecord) -> torch.Tensor:
        if not isinstance(frame, CohortV2CentralFrameRecord):
            raise CohortV2MicroError("state codec requires a central frame record")
        state = frame.engine_state
        entities = state.get("entities")
        world = state.get("world")
        if not isinstance(entities, tuple) or not isinstance(world, Mapping):
            raise CohortV2MicroError("engine state lacks immutable entities or world")
        if len(entities) > self.max_entities:
            raise CohortV2MicroError("engine state exceeds the declared entity slots")
        gravity = world.get("gravity_vector")
        if not isinstance(gravity, tuple) or len(gravity) != 2:
            raise CohortV2MicroError("engine state gravity is malformed")
        values = [
            self._number(gravity[0], "gravity x") / 10.0,
            self._number(gravity[1], "gravity y") / 10.0,
        ]
        ordered = sorted(
            entities,
            key=lambda item: (str(item.get("scenario_object_id")), str(item.get("entity_id"))),
        )
        body_types = {"static": -1.0, "kinematic": 0.0, "dynamic": 1.0}
        for entity in ordered:
            if not isinstance(entity, Mapping):
                raise CohortV2MicroError("engine state entity is malformed")
            active = float(entity.get("lifecycle") == "active")
            body_present = entity.get("body_present") is True
            body = entity.get("body")
            features = [active, float(body_present)]
            if body_present:
                if not isinstance(body, Mapping):
                    raise CohortV2MicroError("present engine body is malformed")
                position = body.get("position")
                velocity = body.get("velocity")
                if (
                    not isinstance(position, tuple)
                    or len(position) != 2
                    or not isinstance(velocity, tuple)
                    or len(velocity) != 2
                    or body.get("body_type") not in body_types
                ):
                    raise CohortV2MicroError("engine body kinematics are malformed")
                rotation = math.radians(self._number(body.get("rotation_degrees"), "rotation"))
                features.extend((
                    body_types[str(body["body_type"])],
                    float(body.get("simulated") is True),
                    float(body.get("gravity_applicable") is True),
                    self._number(body.get("gravity_scale"), "gravity scale"),
                    self._number(position[0], "position x") / 20.0,
                    self._number(position[1], "position y") / 20.0,
                    self._number(velocity[0], "velocity x") / 20.0,
                    self._number(velocity[1], "velocity y") / 20.0,
                    math.sin(rotation),
                    math.cos(rotation),
                    self._number(
                        body.get("angular_velocity_degrees_per_second"),
                        "angular velocity",
                    ) / 360.0,
                ))
            else:
                features.extend((0.0,) * (STATE_FEATURES_PER_ENTITY - 2))
            if len(features) != STATE_FEATURES_PER_ENTITY:
                raise AssertionError("entity feature declaration drifted")
            values.extend(features)
        values.extend(
            (0.0,) * ((self.max_entities - len(ordered)) * STATE_FEATURES_PER_ENTITY)
        )
        values.extend((0.0,) * (self.latent_dim - len(values)))
        return torch.tensor(values, dtype=torch.float32)

    def batch(self, frames: tuple[CohortV2CentralFrameRecord, ...]) -> torch.Tensor:
        if type(frames) is not tuple or not frames:
            raise CohortV2MicroError("state codec batch must be nonempty")
        return torch.stack(tuple(self.encode(frame) for frame in frames))


def cohort_v2_action(window: CohortV2OracleWindow) -> torch.Tensor:
    """Return the declared five-component intervention input."""
    action = window.intervention.get("engine_relative_action")
    if not isinstance(action, Mapping):
        raise CohortV2MicroError("window lacks its engine-relative intervention")
    drag = action.get("drag_delta_canvas_pixels")
    if not isinstance(drag, tuple) or len(drag) != 2:
        raise CohortV2MicroError("intervention drag delta is malformed")
    values = (
        float(drag[0]) / 480.0,
        float(drag[1]) / 480.0,
        float(action.get("hold_milliseconds")) / 1000.0,
        float(action.get("tap_time_milliseconds")) / 1000.0,
        1.0,
    )
    if not all(math.isfinite(value) for value in values):
        raise CohortV2MicroError("intervention contains a nonfinite value")
    return torch.tensor(values, dtype=torch.float32)


def _relation_value(frame: CohortV2CentralFrameRecord, predicate: str) -> RelationTransitionValue:
    label = frame.labels[predicate]
    availability = label.get("availability")
    relations = label.get("relations")
    if availability == "available":
        return RelationTransitionValue(availability, relations)
    if isinstance(availability, str) and availability.startswith("unavailable_"):
        if relations not in (None, ()):
            raise CohortV2MicroError(f"unavailable {predicate} contains relations")
        return RelationTransitionValue(availability, None)
    raise CohortV2MicroError(f"{predicate} availability is malformed")


def _active_entity_ids(frame: CohortV2CentralFrameRecord) -> tuple[str, ...]:
    entities = frame.engine_state.get("entities")
    if not isinstance(entities, tuple):
        raise CohortV2MicroError("engine state entities are malformed")
    result = tuple(sorted(
        entity["entity_id"]
        for entity in entities
        if isinstance(entity, Mapping) and entity.get("lifecycle") == "active"
    ))
    if any(type(entity) is not str or not entity for entity in result):
        raise CohortV2MicroError("active entity id is malformed")
    return result


def _queries(entity_ids: tuple[str, ...], predicate: str) -> tuple[tuple[str, str], ...]:
    if predicate == "contact":
        return tuple(
            (first, second)
            for index, first in enumerate(entity_ids)
            for second in entity_ids[index + 1:]
        )
    return tuple(
        (first, second)
        for first in entity_ids
        for second in entity_ids
        if first != second
    )


def _canonical_contact(relation: tuple[str, str]) -> tuple[str, str]:
    return tuple(sorted(relation))  # type: ignore[return-value]


def _balanced_binary_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    groups = []
    for value in (0.0, 1.0):
        selected = losses[labels == value]
        if selected.numel():
            groups.append(selected.mean())
    if not groups:
        return logits.sum() * 0.0
    return torch.stack(groups).mean()


@dataclass(frozen=True, slots=True)
class MicroRelationLoss:
    loss: torch.Tensor
    per_example: torch.Tensor
    available_sample_mask: torch.Tensor
    available_predicate_count: int
    relation_query_count: int


@dataclass(frozen=True, slots=True)
class MicroPredicateLoss:
    loss: torch.Tensor
    per_example: torch.Tensor
    available_sample_mask: torch.Tensor
    available_predicate_count: int


def micro_predicate_loss(
    head: MicroReadoutHead,
    carrier: torch.Tensor,
    targets: tuple[CohortV2CentralFrameRecord, ...],
    *,
    weights: torch.Tensor | None = None,
) -> MicroPredicateLoss:
    """Supervise the public contact/support logits from oracle relation sets."""
    if not isinstance(head, MicroReadoutHead):
        raise CohortV2MicroError("micro loss requires the model micro head")
    if (
        not isinstance(carrier, torch.Tensor)
        or carrier.ndim != 2
        or type(targets) is not tuple
        or len(targets) != carrier.shape[0]
    ):
        raise CohortV2MicroError("micro targets must match the carrier batch")
    if weights is None:
        weights = torch.ones(carrier.shape[0], device=carrier.device)
    if not isinstance(weights, torch.Tensor) or weights.shape != (carrier.shape[0],):
        raise CohortV2MicroError("micro loss weights must match the carrier batch")
    logits = head(carrier)
    labels = torch.zeros_like(logits)
    availability = torch.zeros_like(logits, dtype=torch.bool)
    for sample_index, target in enumerate(targets):
        for predicate_index, predicate in enumerate(MICRO_PREDICATES):
            value = _relation_value(target, predicate)
            if value.available:
                assert value.relations is not None
                labels[sample_index, predicate_index] = float(bool(value.relations))
                availability[sample_index, predicate_index] = True
    element_losses = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    counts = availability.sum(dim=1)
    available_samples = counts > 0
    per_example = (
        (element_losses * availability).sum(dim=1)
        / counts.clamp_min(1)
    )
    if not bool(available_samples.any()):
        loss = carrier.sum() * 0.0
    else:
        selected_weights = weights * available_samples
        loss = (per_example * selected_weights).sum() / selected_weights.sum()
    return MicroPredicateLoss(
        loss,
        per_example,
        available_samples,
        int(availability.sum().item()),
    )


def micro_relation_loss(
    head: MicroReadoutHead,
    carrier: torch.Tensor,
    targets: tuple[CohortV2CentralFrameRecord, ...],
    *,
    weights: torch.Tensor | None = None,
) -> MicroRelationLoss:
    """Supervise exact target relation sets, masking unavailable predicates."""
    if not isinstance(head, MicroReadoutHead):
        raise CohortV2MicroError("micro loss requires the model micro head")
    if (
        not isinstance(carrier, torch.Tensor)
        or carrier.ndim != 2
        or type(targets) is not tuple
        or len(targets) != carrier.shape[0]
    ):
        raise CohortV2MicroError("micro targets must match the carrier batch")
    if weights is None:
        weights = torch.ones(carrier.shape[0], device=carrier.device)
    if not isinstance(weights, torch.Tensor) or weights.shape != (carrier.shape[0],):
        raise CohortV2MicroError("micro loss weights must match the carrier batch")
    sample_terms: list[list[torch.Tensor]] = [[] for _ in targets]
    available_predicates = 0
    query_count = 0
    for predicate in MICRO_PREDICATES:
        values = tuple(_relation_value(target, predicate) for target in targets)
        query_batches = tuple(
            _queries(_active_entity_ids(target), predicate) if value.available else ()
            for target, value in zip(targets, values, strict=True)
        )
        logits = head.relation_logits(carrier, predicate, query_batches)
        for index, (value, queries, sample_logits) in enumerate(
            zip(values, query_batches, logits, strict=True)
        ):
            if not value.available:
                continue
            assert value.relations is not None
            positives = (
                {_canonical_contact(relation) for relation in value.relations}
                if predicate == "contact"
                else set(value.relations)
            )
            labels = torch.tensor(
                [
                    float(
                        (_canonical_contact(query) if predicate == "contact" else query)
                        in positives
                    )
                    for query in queries
                ],
                dtype=carrier.dtype,
                device=carrier.device,
            )
            sample_terms[index].append(_balanced_binary_loss(sample_logits, labels))
            available_predicates += 1
            query_count += len(queries)
    per_example_terms = []
    available_samples = []
    for index, terms in enumerate(sample_terms):
        if terms:
            per_example_terms.append(torch.stack(terms).mean())
            available_samples.append(True)
        else:
            per_example_terms.append(carrier[index].sum() * 0.0)
            available_samples.append(False)
    per_example = torch.stack(per_example_terms)
    available_sample_mask = torch.tensor(
        available_samples, dtype=torch.bool, device=carrier.device
    )
    if not any(available_samples):
        loss = carrier.sum() * 0.0
    else:
        selected_weights = weights * available_sample_mask
        loss = (per_example * selected_weights).sum() / selected_weights.sum()
    return MicroRelationLoss(
        loss,
        per_example,
        available_sample_mask,
        available_predicates,
        query_count,
    )


def _micro_available(window: CohortV2OracleWindow) -> bool:
    return all(
        window.context.labels[predicate].get("availability") == "available"
        and window.target.labels[predicate].get("availability") == "available"
        for predicate in MICRO_PREDICATES
    )


class CohortV2MicroTrainingData:
    """Balanced strict-horizon pools from the training exposure role only."""

    def __init__(
        self,
        reader: CohortV2ReleaseReader,
        config: CohortV2MicroConfig,
        *,
        included_attempt_ids: frozenset[str] | None = None,
    ) -> None:
        if not isinstance(reader, CohortV2ReleaseReader):
            raise CohortV2MicroError("micro training requires a validated release reader")
        if not reader.rollouts or {rollout.exposure_role for rollout in reader.rollouts} != {"training"}:
            raise CohortV2MicroError("learned parameters may use only the training exposure role")
        if reader.release_identity != COHORT_V2_RELEASE_IDENTITY:
            raise CohortV2MicroError("micro training reader targets another release")
        available_attempt_ids = frozenset(
            rollout.attempt_id for rollout in reader.rollouts
        )
        if included_attempt_ids is None:
            included_attempt_ids = available_attempt_ids
        if (
            type(included_attempt_ids) is not frozenset
            or not included_attempt_ids
            or not included_attempt_ids <= available_attempt_ids
        ):
            raise CohortV2MicroError(
                "included training attempt ids must be a nonempty reader subset"
            )
        dataset = CohortV2OracleWindowDataset(
            reader, requested_horizons=COHORT_V2_HORIZONS
        )
        pools: dict[PredictionPair, list[CohortV2OracleWindow]] = {
            pair: [] for pair in MICRO_PAIRS
        }
        for window in dataset:
            if window.attempt_id not in included_attempt_ids:
                continue
            if window.effective_horizon != window.requested_horizon:
                continue
            continuous = PredictionPair(window.requested_horizon, Abstraction.CONTINUOUS)
            pools[continuous].append(window)
            if _micro_available(window):
                pools[PredictionPair(window.requested_horizon, Abstraction.MICRO)].append(window)
        empty = tuple(pair.identity for pair, values in pools.items() if not values)
        if empty:
            raise CohortV2MicroError(f"training has no eligible windows for pairs: {empty}")
        self.reader = reader
        self.config = config
        self.included_attempt_ids = included_attempt_ids
        self.pools = {pair: tuple(values) for pair, values in pools.items()}
        self.frame_counts = {
            rollout.attempt_id: len(rollout.frame_records) for rollout in reader.rollouts
            if rollout.attempt_id in included_attempt_ids
        }

    def schedule_at(self, step: int) -> PredictionPair:
        if type(step) is not int or step < 0:
            raise CohortV2MicroError("step must be nonnegative")
        keys = list(MICRO_PAIRS)
        cycle, offset = divmod(step, len(keys))
        generator = random.Random(self.config.seed)
        for _ in range(cycle + 1):
            generator.shuffle(keys)
        return keys[offset]

    def batch_at(
        self, pair: PredictionPair, step: int
    ) -> tuple[CohortV2OracleWindow, ...]:
        pool = self.pools[pair]
        pair_offset = MICRO_PAIRS.index(pair)
        generator = random.Random(self.config.seed + step * len(MICRO_PAIRS) + pair_offset)
        return tuple(
            pool[generator.randrange(len(pool))] for _ in range(self.config.batch_size)
        )

    def duration_weights(
        self, windows: tuple[CohortV2OracleWindow, ...]
    ) -> torch.Tensor:
        return torch.tensor(
            [
                window.effective_horizon / (self.frame_counts[window.attempt_id] - 1)
                for window in windows
            ],
            dtype=torch.float32,
        )


@dataclass(frozen=True, slots=True)
class CohortV2MicroStepResult:
    step: int
    pair: PredictionPair
    total_loss: float
    carrier_loss: float
    micro_loss: float
    available_predicate_count: int
    relation_query_count: int
    learning_rate: float


class CohortV2MicroTrainer:
    def __init__(
        self,
        data: CohortV2MicroTrainingData,
        config: CohortV2MicroConfig,
        *,
        symbolic_gate: Callable[
            [tuple[CohortV2OracleWindow, ...]], torch.Tensor
        ]
        | None = None,
    ) -> None:
        self.data = data
        self.config = config
        self.symbolic_gate = symbolic_gate
        self.device = torch.device(config.device)
        self.codec = CohortV2StateCodec(
            latent_dim=config.latent_dim, max_entities=config.max_entities
        )
        self.predictor = DualOutputPredictor(config.predictor_config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.predictor.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.step_count = 0
        self.pair_counts = {pair: 0 for pair in MICRO_PAIRS}

    def _learning_rate(self, step: int) -> float:
        progress = min(step, self.config.steps - 1) / max(1, self.config.steps - 1)
        return self.config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    def train_step(self) -> CohortV2MicroStepResult:
        step = self.step_count
        pair = self.data.schedule_at(step)
        windows = self.data.batch_at(pair, step)
        contexts = self.codec.batch(tuple(window.context for window in windows)).to(self.device)
        targets = self.codec.batch(tuple(window.target for window in windows)).to(self.device)
        actions = torch.stack(tuple(cohort_v2_action(window) for window in windows)).to(self.device)
        weights = self.data.duration_weights(windows).to(self.device)
        request = build_cohort_v2_transition_request(pair, windows)
        carrier = self.predictor.carrier(contexts, actions, request)
        per_example = (carrier - targets).pow(2).mean(dim=1)
        carrier_loss = (per_example * weights).sum() / weights.sum()
        if pair.abstraction is Abstraction.MICRO:
            symbolic_weights = weights
            if self.symbolic_gate is not None:
                gate = self.symbolic_gate(windows)
                if (
                    not isinstance(gate, torch.Tensor)
                    or gate.shape != weights.shape
                    or not bool(torch.isfinite(gate).all())
                    or bool((gate < 0.0).any())
                    or bool((gate > 1.0).any())
                ):
                    raise CohortV2MicroError(
                        "symbolic gate must return one finite [0, 1] weight per window"
                    )
                symbolic_weights = weights * gate.to(self.device)
            relations = micro_relation_loss(
                self.predictor.micro_head,
                carrier,
                tuple(window.target for window in windows),
                weights=symbolic_weights,
            )
            predicates = micro_predicate_loss(
                self.predictor.micro_head,
                carrier,
                tuple(window.target for window in windows),
                weights=symbolic_weights,
            )
            micro_loss = (relations.loss + predicates.loss) / 2.0
        else:
            relations = MicroRelationLoss(
                carrier.sum() * 0.0,
                torch.zeros(carrier.shape[0], device=carrier.device),
                torch.zeros(
                    carrier.shape[0], dtype=torch.bool, device=carrier.device
                ),
                0,
                0,
            )
            micro_loss = carrier.sum() * 0.0
        total = carrier_loss + self.config.symbolic_weight * micro_loss
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(self.predictor.parameters(), self.config.grad_clip)
        learning_rate = self._learning_rate(step)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.optimizer.step()
        self.step_count += 1
        self.pair_counts[pair] += 1
        return CohortV2MicroStepResult(
            step=step,
            pair=pair,
            total_loss=float(total.detach()),
            carrier_loss=float(carrier_loss.detach()),
            micro_loss=float(micro_loss.detach()),
            available_predicate_count=relations.available_predicate_count,
            relation_query_count=relations.relation_query_count,
            learning_rate=learning_rate,
        )


@dataclass(frozen=True, slots=True)
class CohortV2MicroCheckpoint:
    path: Path
    identity: str
    step: int
    pair_counts: tuple[tuple[str, int], ...]


def _checkpoint_identity(
    reader: CohortV2ReleaseReader,
    config: CohortV2MicroConfig,
    codec: CohortV2StateCodec,
    model_state_identity: str,
    step: int,
    pair_counts: tuple[tuple[str, int], ...],
) -> str:
    return identity((
        "cohort-v2-micro-checkpoint-v2",
        reader.release_identity,
        reader.partition_identity,
        CAPABILITY_DECLARATION_IDENTITY,
        config.identity,
        config.predictor_config.identity,
        codec.identity,
        model_state_identity,
        step,
        pair_counts,
        tuple(sorted(MICRO_CAPABILITIES)),
        MICRO_RELATION_AUTHORITY,
        "training",
    ))


def cohort_v2_model_state_identity(model_state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(model_state):
        tensor = model_state[name]
        if type(name) is not str or not isinstance(tensor, torch.Tensor):
            raise CohortV2MicroError("micro checkpoint model state is malformed")
        contiguous = tensor.detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return f"sha256:{digest.hexdigest()}"


def save_cohort_v2_micro_checkpoint(
    path: Path,
    trainer: CohortV2MicroTrainer,
) -> CohortV2MicroCheckpoint:
    counts = tuple(
        (f"horizon={pair.delta},abstraction={pair.abstraction}", trainer.pair_counts[pair])
        for pair in MICRO_PAIRS
    )
    model_state = trainer.predictor.state_dict()
    model_state_identity = cohort_v2_model_state_identity(model_state)
    checkpoint_identity = _checkpoint_identity(
        trainer.data.reader,
        trainer.config,
        trainer.codec,
        model_state_identity,
        trainer.step_count,
        counts,
    )
    payload = {
        "capabilities": list(sorted(MICRO_CAPABILITIES)),
        "capability_declaration_identity": CAPABILITY_DECLARATION_IDENTITY,
        "checkpoint_identity": checkpoint_identity,
        "codec_identity": trainer.codec.identity,
        "config_identity": trainer.config.identity,
        "exposure_role": "training",
        "model_state": model_state,
        "model_state_identity": model_state_identity,
        "micro_relation_authority": MICRO_RELATION_AUTHORITY,
        "pair_counts": dict(counts),
        "partition_identity": trainer.data.reader.partition_identity,
        "predictor_config_identity": trainer.config.predictor_config.identity,
        "release_identity": trainer.data.reader.release_identity,
        "schema": MICRO_CHECKPOINT_SCHEMA,
        "step": trainer.step_count,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return CohortV2MicroCheckpoint(target, checkpoint_identity, trainer.step_count, counts)


def load_cohort_v2_micro_checkpoint(
    path: Path,
    *,
    reader: CohortV2ReleaseReader,
    config: CohortV2MicroConfig,
    device: str | None = None,
) -> tuple[DualOutputPredictor, CohortV2StateCodec, CohortV2MicroCheckpoint]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise CohortV2MicroError(f"cannot load micro checkpoint: {error}") from error
    required = {
        "capabilities", "capability_declaration_identity", "checkpoint_identity",
        "codec_identity", "config_identity", "exposure_role", "model_state",
        "model_state_identity", "micro_relation_authority", "pair_counts",
        "partition_identity", "predictor_config_identity",
        "release_identity", "schema", "step",
    }
    if type(payload) is not dict or set(payload) != required:
        raise CohortV2MicroError("micro checkpoint envelope is malformed")
    codec = CohortV2StateCodec(
        latent_dim=config.latent_dim, max_entities=config.max_entities
    )
    pair_counts_value = payload["pair_counts"]
    if type(pair_counts_value) is not dict:
        raise CohortV2MicroError("micro checkpoint pair counts are malformed")
    pair_counts = tuple(
        (f"horizon={pair.delta},abstraction={pair.abstraction}", pair_counts_value.get(
            f"horizon={pair.delta},abstraction={pair.abstraction}"
        ))
        for pair in MICRO_PAIRS
    )
    step = payload["step"]
    if not isinstance(payload["model_state"], Mapping):
        raise CohortV2MicroError("micro checkpoint model state is malformed")
    model_state_identity = cohort_v2_model_state_identity(payload["model_state"])
    expected_identity = _checkpoint_identity(
        reader,
        config,
        codec,
        model_state_identity,
        step,
        pair_counts,
    )
    if (
        payload["schema"] != MICRO_CHECKPOINT_SCHEMA
        or payload["release_identity"] != reader.release_identity
        or payload["partition_identity"] != reader.partition_identity
        or payload["capability_declaration_identity"] != CAPABILITY_DECLARATION_IDENTITY
        or payload["config_identity"] != config.identity
        or payload["predictor_config_identity"] != config.predictor_config.identity
        or payload["codec_identity"] != codec.identity
        or payload["model_state_identity"] != model_state_identity
        or payload["exposure_role"] != "training"
        or payload["capabilities"] != list(sorted(MICRO_CAPABILITIES))
        or payload["micro_relation_authority"] != MICRO_RELATION_AUTHORITY
        or type(step) is not int
        or step != config.steps
        or any(type(count) is not int or count < 0 for _, count in pair_counts)
        or sum(count for _, count in pair_counts) != step
        or payload["checkpoint_identity"] != expected_identity
    ):
        raise CohortV2MicroError("micro checkpoint provenance is stale or malformed")
    predictor = DualOutputPredictor(config.predictor_config)
    try:
        predictor.load_state_dict(payload["model_state"], strict=True)
    except (RuntimeError, TypeError) as error:
        raise CohortV2MicroError(f"micro checkpoint model state is invalid: {error}") from error
    predictor.to(torch.device(device or config.device)).eval()
    return predictor, codec, CohortV2MicroCheckpoint(
        Path(path), expected_identity, step, pair_counts
    )


class CohortV2MicroPairScorer:
    """Duration-weighted carrier and exact-relation objective for #3's seam."""

    capabilities = MICRO_CAPABILITIES

    def __init__(
        self,
        predictor: DualOutputPredictor,
        codec: CohortV2StateCodec,
        checkpoint: CohortV2MicroCheckpoint,
        config: CohortV2MicroConfig,
        readers: tuple[CohortV2ReleaseReader, ...],
        *,
        progress_every: int = 0,
        progress_total: int | None = None,
        worker_name: str | None = None,
    ) -> None:
        self.predictor = predictor
        self.codec = codec
        self.checkpoint_identity = checkpoint.identity
        self.config = config
        self.device = next(predictor.parameters()).device
        self.frame_counts = {
            rollout.attempt_id: len(rollout.frame_records)
            for reader in readers
            for rollout in reader.rollouts
        }
        self.objective_identity = identity((
            "cohort-v2-micro-pair-objective-v2",
            "duration-weighted-carrier-mse+public-micro-bce+exact-micro-bce",
            config.symbolic_weight,
            MICRO_RELATION_AUTHORITY,
            codec.identity,
            checkpoint.identity,
        ))
        self.progress_every = progress_every
        self.progress_total = progress_total
        self.worker_name = worker_name or str(self.device)
        self.call_count = 0
        self._next_progress = progress_every

    def objective(self, window: CohortV2OracleWindow, pair: PredictionPair) -> float:
        return self.objective_batch((window,), pair)[0]

    def objective_batch(
        self,
        windows: tuple[CohortV2OracleWindow, ...],
        pair: PredictionPair,
    ) -> tuple[float, ...]:
        if type(windows) is not tuple or not windows:
            raise CohortV2MicroError("scoring batch must be nonempty")
        with torch.no_grad():
            context = self.codec.batch(
                tuple(window.context for window in windows)
            ).to(self.device)
            target = self.codec.batch(
                tuple(window.target for window in windows)
            ).to(self.device)
            action = torch.stack(
                tuple(cohort_v2_action(window) for window in windows)
            ).to(self.device)
            request = build_cohort_v2_transition_request(pair, windows)
            carrier = self.predictor.carrier(context, action, request)
            carrier_loss = (carrier - target).pow(2).mean(dim=1)
            if pair.abstraction is Abstraction.MICRO:
                relation = micro_relation_loss(
                    self.predictor.micro_head,
                    carrier,
                    tuple(window.target for window in windows),
                ).per_example
                predicate = micro_predicate_loss(
                    self.predictor.micro_head,
                    carrier,
                    tuple(window.target for window in windows),
                ).per_example
                symbolic = (relation + predicate) / 2.0
            else:
                symbolic = torch.zeros_like(carrier_loss)
            duration_weight = torch.tensor(
                [
                    window.effective_horizon
                    / (self.frame_counts[window.attempt_id] - 1)
                    for window in windows
                ],
                dtype=carrier.dtype,
                device=self.device,
            )
            values = duration_weight * (
                carrier_loss + self.config.symbolic_weight * symbolic
            )
        self.call_count += len(windows)
        if self.progress_every and self.call_count >= self._next_progress:
            suffix = (
                f"/{self.progress_total}" if self.progress_total is not None else ""
            )
            print(
                f"[score {self.worker_name} {self.call_count}{suffix}] latest="
                f"h{pair.delta}/{pair.abstraction} "
                f"mean_objective={float(values.mean()):.6f}",
                flush=True,
            )
            while self._next_progress <= self.call_count:
                self._next_progress += self.progress_every
        return tuple(float(value) for value in values.detach().cpu().tolist())


def _frontier_document(
    *,
    evaluation_identity: str,
    checkpoint_identity: str,
    objective_identity: str,
    grid_identity: str,
    state_set_identity: str,
    release_identity: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    fields = (
        evaluation_identity,
        checkpoint_identity,
        objective_identity,
        grid_identity,
        state_set_identity,
        tuple(
            (
                row["requested_horizon"], row["abstraction"], row["status"],
                row["available_state_count"], row["mean_objective"],
                tuple(row["unavailable_reasons"]),
            )
            for row in rows
        ),
    )
    return {
        "artifact_type": "pair_frontier_input",
        "checkpoint_identity": checkpoint_identity,
        "evaluation_identity": evaluation_identity,
        "exposure_role": "model_selection",
        "frontier_identity": identity(("cohort-v2-micro-frontier-input-v1", *fields)),
        "grid_identity": grid_identity,
        "micro_relation_authority": MICRO_RELATION_AUTHORITY,
        "objective_identity": objective_identity,
        "pairs": rows,
        "release_identity": release_identity,
        "schema": MICRO_FRONTIER_SCHEMA,
        "state_set_identity": state_set_identity,
    }


def _frontier_payload(result: CohortV2EvaluationResult) -> dict[str, object]:
    states = tuple(
        state for state in result.states if state.exposure_role == "model_selection"
    )
    rows = []
    for pair_index, pair in enumerate(result.grid.pairs):
        outcomes = tuple(state.outcomes[pair_index] for state in states)
        values = tuple(float(outcome.objective) for outcome in outcomes if outcome.available)
        reasons = sorted({
            reason for outcome in outcomes for reason in outcome.unavailable_reasons
        })
        rows.append({
            "abstraction": str(pair.abstraction),
            "available_state_count": len(values),
            "mean_objective": sum(values) / len(values) if values else None,
            "requested_horizon": pair.delta,
            "status": "available" if values else "unavailable",
            "unavailable_reasons": reasons,
        })
    return _frontier_document(
        evaluation_identity=result.identity,
        checkpoint_identity=result.checkpoint_identity,
        objective_identity=result.objective_identity,
        grid_identity=result.grid.identity,
        state_set_identity=result.state_set_identity,
        release_identity=result.release_identity,
        rows=rows,
    )


def write_cohort_v2_micro_frontier_input(
    path: Path, result: CohortV2EvaluationResult
) -> Path:
    payload = _frontier_payload(result)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(payload))
    os.replace(temporary, target)
    return target


def validate_cohort_v2_micro_frontier_input(
    path: Path, result: CohortV2EvaluationResult
) -> None:
    expected = canonical_json_bytes(_frontier_payload(result))
    try:
        actual = Path(path).read_bytes()
    except OSError as error:
        raise CohortV2MicroError(f"cannot load micro frontier input: {error}") from error
    if actual != expected:
        raise CohortV2MicroError("micro frontier input differs from its evaluation")


def validate_cohort_v2_micro_frontier_artifacts(
    path: Path, evaluation_root: Path
) -> None:
    try:
        actual = Path(path).read_bytes()
        manifest = json.loads((Path(evaluation_root) / "manifest.json").read_bytes())
        records = tuple(
            json.loads(line)
            for line in (Path(evaluation_root) / manifest["records"]).read_bytes().splitlines()
        )
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise CohortV2MicroError(
            f"cannot load micro frontier source artifacts: {error}"
        ) from error
    model_selection = tuple(
        record for record in records if record.get("exposure_role") == "model_selection"
    )
    rows = []
    for pair_index, pair in enumerate(manifest["pairs"]):
        outcomes = tuple(record["outcomes"][pair_index] for record in model_selection)
        values = tuple(
            float(outcome["objective"])
            for outcome in outcomes
            if outcome.get("status") == "available"
        )
        reasons = sorted({
            reason
            for outcome in outcomes
            for reason in outcome.get("unavailable_reasons", ())
        })
        rows.append({
            "abstraction": pair["abstraction"],
            "available_state_count": len(values),
            "mean_objective": sum(values) / len(values) if values else None,
            "requested_horizon": pair["requested_horizon"],
            "status": "available" if values else "unavailable",
            "unavailable_reasons": reasons,
        })
    expected = canonical_json_bytes(_frontier_document(
        evaluation_identity=manifest["evaluation_identity"],
        checkpoint_identity=manifest["checkpoint_identity"],
        objective_identity=manifest["objective_identity"],
        grid_identity=manifest["grid_identity"],
        state_set_identity=manifest["state_set_identity"],
        release_identity=manifest["release_identity"],
        rows=rows,
    ))
    if actual != expected:
        raise CohortV2MicroError(
            "micro frontier input differs from persisted evaluation records"
        )


__all__ = [
    "MICRO_CAPABILITIES",
    "MICRO_FRONTIER_SCHEMA",
    "MICRO_PAIRS",
    "MICRO_RELATION_AUTHORITY",
    "CohortV2MicroCheckpoint",
    "CohortV2MicroConfig",
    "CohortV2MicroError",
    "CohortV2MicroPairScorer",
    "CohortV2MicroStepResult",
    "CohortV2MicroTrainer",
    "CohortV2MicroTrainingData",
    "CohortV2StateCodec",
    "MicroRelationLoss",
    "MicroPredicateLoss",
    "cohort_v2_action",
    "cohort_v2_model_state_identity",
    "load_cohort_v2_micro_checkpoint",
    "micro_relation_loss",
    "micro_predicate_loss",
    "save_cohort_v2_micro_checkpoint",
    "validate_cohort_v2_micro_frontier_input",
    "validate_cohort_v2_micro_frontier_artifacts",
    "write_cohort_v2_micro_frontier_input",
]
