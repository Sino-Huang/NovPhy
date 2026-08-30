"""Matched-capacity oracle symbolic-interface comparison for issue #13."""
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Final

import torch
from torch import nn

from world_model.data import (
    CohortV2CentralFrameRecord,
    CohortV2OracleWindow,
    CohortV2OracleWindowDataset,
    CohortV2ReleaseReader,
)
from world_model.data.cohort_v2 import (
    CAPABILITY_DECLARATION_IDENTITY,
    MICRO_SPEC_IDENTITY,
)
from world_model.model import (
    Abstraction,
    DualOutputPredictor,
    MicroTransitionBatch,
    PredictionPair,
    identity,
)
from world_model.training.cohort_v2 import build_cohort_v2_transition_request
from world_model.training.cohort_v2_micro import (
    MICRO_CAPABILITIES,
    MICRO_PAIRS,
    CohortV2MicroConfig,
    CohortV2MicroError,
    CohortV2MicroTrainer,
    CohortV2MicroTrainingData,
    CohortV2StateCodec,
    cohort_v2_action,
    cohort_v2_model_state_identity,
)


INTERFACE_CHECKPOINT_SCHEMA: Final = "cohort_v2_symbolic_interface_checkpoint_v1"
MATERIAL_RELATION_F1_GAIN: Final = 0.02
PREDICATES: Final = ("contact", "supports")


@unique
class SymbolicInterface(StrEnum):
    NO_SYMBOL = "no_symbol"
    ORDERED_FLAT = "ordered_flat_predicate"
    DIRECTED_GNN = "directed_gnn"
    SPSG = "spsg"


INTERFACE_ORDER: Final = tuple(SymbolicInterface)


def _entity_embedding(
    table: nn.Embedding, entity_identity: str, device: torch.device
) -> torch.Tensor:
    encoded = torch.tensor(tuple(entity_identity.encode("utf-8")), device=device)
    return table(encoded).mean(dim=0)


class MatchedMicroInterfaceAdapter(nn.Module):
    """Four interfaces with an identical learned parameter inventory.

    ``ordered_flat_predicate`` serializes grounded positive predicates with
    separate first/second role projections. ``directed_gnn`` aggregates those
    messages at their destination nodes. ``spsg`` replaces each ordinary edge
    message with a tensor-product role/filler binding before directed message
    passing. The SPSG branch deliberately has no contrastive-negative loss.
    """

    def __init__(self, hidden_dim: int, interface: SymbolicInterface) -> None:
        super().__init__()
        if type(interface) is not SymbolicInterface:
            raise CohortV2MicroError("symbolic interface is invalid")
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise CohortV2MicroError("symbolic interface hidden width is invalid")
        self.interface = interface
        self.hidden_dim = hidden_dim
        self.entity_embedding = nn.Embedding(256, hidden_dim)
        self.predicate_embedding = nn.Embedding(len(PREDICATES), hidden_dim)
        self.sender_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.receiver_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.message_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.node_update = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.availability_projection = nn.Linear(len(PREDICATES), hidden_dim, bias=False)
        factor = math.isqrt(hidden_dim)
        while hidden_dim % factor:
            factor -= 1
        self.role_dim = factor
        self.filler_dim = hidden_dim // factor

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _entity(self, value: str, device: torch.device) -> torch.Tensor:
        return _entity_embedding(self.entity_embedding, value, device)

    @staticmethod
    def _edges(sample) -> tuple[tuple[int, str, str], ...]:
        edges = []
        for predicate_index, predicate in enumerate(PREDICATES):
            relation_value = getattr(sample, predicate)
            if not relation_value.available:
                continue
            assert relation_value.relations is not None
            for first, second in relation_value.relations:
                if predicate == "contact" and second < first:
                    first, second = second, first
                edges.append((predicate_index, first, second))
        return tuple(sorted(edges))

    def _ordinary_message(
        self, predicate: int, first: str, second: str, device: torch.device
    ) -> torch.Tensor:
        return self.message_projection(torch.nn.functional.silu(
            self.sender_projection(self._entity(first, device))
            + self.receiver_projection(self._entity(second, device))
            + self.predicate_embedding.weight[predicate]
        ))

    def _bound_message(
        self, predicate: int, first: str, second: str, device: torch.device
    ) -> torch.Tensor:
        sender = self.sender_projection(self._entity(first, device)).reshape(
            self.role_dim, self.filler_dim
        )
        receiver = self.receiver_projection(self._entity(second, device)).reshape(
            self.role_dim, self.filler_dim
        )
        role = sender.mean(dim=1)
        filler = receiver.mean(dim=0)
        binding = torch.outer(role, filler).reshape(self.hidden_dim)
        return self.message_projection(torch.nn.functional.silu(
            binding + self.predicate_embedding.weight[predicate]
        ))

    def _constant(self, device: torch.device) -> torch.Tensor:
        # A learned, sample-independent adapter gives the no-symbol baseline the
        # same capacity without exposing predicate content or availability.
        entity = self.entity_embedding.weight.mean(dim=0)
        predicate = self.predicate_embedding.weight.mean(dim=0)
        message = self.message_projection(torch.nn.functional.silu(
            self.sender_projection(entity) + self.receiver_projection(entity) + predicate
        ))
        constant_availability = torch.ones(len(PREDICATES), device=device)
        return self.output_projection(torch.nn.functional.silu(
            self.node_update(torch.nn.functional.silu(message))
            + self.availability_projection(constant_availability)
        ))

    def _flat(self, sample, device: torch.device) -> torch.Tensor:
        edges = self._edges(sample)
        aggregate = torch.zeros(self.hidden_dim, device=device)
        for predicate, first, second in edges:
            aggregate = aggregate + self._ordinary_message(
                predicate, first, second, device
            )
        availability = torch.tensor(
            tuple(getattr(sample, predicate).available for predicate in PREDICATES),
            dtype=aggregate.dtype,
            device=device,
        )
        aggregate = aggregate + self.availability_projection(availability)
        return self.output_projection(torch.nn.functional.silu(
            self.node_update(torch.nn.functional.silu(aggregate))
        ))

    def _graph(self, sample, device: torch.device, *, tensor_product: bool) -> torch.Tensor:
        declared_edges = self._edges(sample)
        directed_edges = []
        for predicate, first, second in declared_edges:
            directed_edges.append((predicate, first, second))
            if predicate == 0:
                directed_edges.append((predicate, second, first))
        nodes = tuple(sorted({
            entity
            for _, first, second in directed_edges
            for entity in (first, second)
        }))
        node_values = {node: self._entity(node, device) for node in nodes}
        incoming = {
            node: torch.zeros(self.hidden_dim, device=device) for node in nodes
        }
        message = self._bound_message if tensor_product else self._ordinary_message
        for predicate, first, second in directed_edges:
            incoming[second] = incoming[second] + message(
                predicate, first, second, device
            )
        if nodes:
            aggregate = torch.stack(tuple(
                self.node_update(torch.nn.functional.silu(node_values[node] + incoming[node]))
                for node in nodes
            )).mean(dim=0)
        else:
            aggregate = self.node_update(torch.zeros(self.hidden_dim, device=device))
        availability = torch.tensor(
            tuple(getattr(sample, predicate).available for predicate in PREDICATES),
            dtype=aggregate.dtype,
            device=device,
        )
        return self.output_projection(torch.nn.functional.silu(
            aggregate + self.availability_projection(availability)
        ))

    def forward(self, hidden: torch.Tensor, mode_input: object) -> torch.Tensor:
        if type(mode_input) is not MicroTransitionBatch:
            raise CohortV2MicroError("symbolic interface requires a micro transition batch")
        if len(mode_input.samples) != hidden.shape[0]:
            raise CohortV2MicroError("symbolic interface batch does not match the carrier")
        encoded = []
        for sample in mode_input.samples:
            if self.interface is SymbolicInterface.NO_SYMBOL:
                value = self._constant(hidden.device)
            elif self.interface is SymbolicInterface.ORDERED_FLAT:
                value = self._flat(sample, hidden.device)
            elif self.interface is SymbolicInterface.DIRECTED_GNN:
                value = self._graph(sample, hidden.device, tensor_product=False)
            else:
                value = self._graph(sample, hidden.device, tensor_product=True)
            encoded.append(value)
        return hidden + torch.stack(encoded)


def build_interface_predictor(
    config: CohortV2MicroConfig, interface: SymbolicInterface
) -> DualOutputPredictor:
    # Construct the default model first so every seeded variant receives the
    # exact same shared-trunk and readout initialization.
    predictor = DualOutputPredictor(config.predictor_config)
    predictor.micro_adapter = MatchedMicroInterfaceAdapter(
        config.hidden_dim, interface
    )
    return predictor


class CohortV2SymbolicInterfaceTrainer(CohortV2MicroTrainer):
    def __init__(
        self,
        data: CohortV2MicroTrainingData,
        config: CohortV2MicroConfig,
        interface: SymbolicInterface,
    ) -> None:
        self.data = data
        self.config = config
        self.interface = interface
        self.device = torch.device(config.device)
        self.codec = CohortV2StateCodec(
            latent_dim=config.latent_dim, max_entities=config.max_entities
        )
        self.predictor = build_interface_predictor(config, interface).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.predictor.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.step_count = 0
        self.pair_counts = {pair: 0 for pair in MICRO_PAIRS}


@dataclass(frozen=True, slots=True)
class SymbolicInterfaceCheckpoint:
    path: Path
    identity: str
    interface: SymbolicInterface
    step: int
    trainable_parameter_count: int
    adapter_parameter_count: int


def _checkpoint_identity(
    reader: CohortV2ReleaseReader,
    config: CohortV2MicroConfig,
    interface: SymbolicInterface,
    model_state_identity: str,
    step: int,
    pair_counts: tuple[tuple[str, int], ...],
) -> str:
    return identity((
        INTERFACE_CHECKPOINT_SCHEMA,
        reader.release_identity,
        reader.partition_identity,
        CAPABILITY_DECLARATION_IDENTITY,
        MICRO_SPEC_IDENTITY,
        interface,
        config.identity,
        model_state_identity,
        step,
        pair_counts,
    ))


def save_symbolic_interface_checkpoint(
    path: Path, trainer: CohortV2SymbolicInterfaceTrainer
) -> SymbolicInterfaceCheckpoint:
    pair_counts = tuple(
        (pair.identity, trainer.pair_counts[pair]) for pair in MICRO_PAIRS
    )
    model_state = trainer.predictor.state_dict()
    model_state_identity = cohort_v2_model_state_identity(model_state)
    checkpoint_identity = _checkpoint_identity(
        trainer.data.reader,
        trainer.config,
        trainer.interface,
        model_state_identity,
        trainer.step_count,
        pair_counts,
    )
    adapter = trainer.predictor.micro_adapter
    payload = {
        "adapter_parameter_count": sum(p.numel() for p in adapter.parameters()),
        "capabilities": sorted(MICRO_CAPABILITIES),
        "capability_declaration_identity": CAPABILITY_DECLARATION_IDENTITY,
        "checkpoint_identity": checkpoint_identity,
        "config_identity": trainer.config.identity,
        "exposure_role": "training",
        "interface": str(trainer.interface),
        "micro_relation_authority": MICRO_SPEC_IDENTITY,
        "model_state": model_state,
        "model_state_identity": model_state_identity,
        "pair_counts": dict(pair_counts),
        "partition_identity": trainer.data.reader.partition_identity,
        "release_identity": trainer.data.reader.release_identity,
        "schema": INTERFACE_CHECKPOINT_SCHEMA,
        "step": trainer.step_count,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in trainer.predictor.parameters()
        ),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return SymbolicInterfaceCheckpoint(
        target,
        checkpoint_identity,
        trainer.interface,
        trainer.step_count,
        payload["trainable_parameter_count"],
        payload["adapter_parameter_count"],
    )


def load_symbolic_interface_checkpoint(
    path: Path,
    *,
    reader: CohortV2ReleaseReader,
    config: CohortV2MicroConfig,
    interface: SymbolicInterface,
    device: str,
) -> tuple[DualOutputPredictor, CohortV2StateCodec, SymbolicInterfaceCheckpoint]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise CohortV2MicroError(f"cannot load symbolic-interface checkpoint: {error}") from error
    if type(payload) is not dict or payload.get("schema") != INTERFACE_CHECKPOINT_SCHEMA:
        raise CohortV2MicroError("symbolic-interface checkpoint envelope is malformed")
    pair_counts = tuple(
        (pair.identity, payload.get("pair_counts", {}).get(pair.identity))
        for pair in MICRO_PAIRS
    )
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping):
        raise CohortV2MicroError("symbolic-interface model state is malformed")
    model_state_identity = cohort_v2_model_state_identity(model_state)
    expected_identity = _checkpoint_identity(
        reader,
        config,
        interface,
        model_state_identity,
        payload.get("step"),
        pair_counts,
    )
    if (
        payload.get("checkpoint_identity") != expected_identity
        or payload.get("release_identity") != reader.release_identity
        or payload.get("partition_identity") != reader.partition_identity
        or payload.get("capability_declaration_identity") != CAPABILITY_DECLARATION_IDENTITY
        or payload.get("micro_relation_authority") != MICRO_SPEC_IDENTITY
        or payload.get("config_identity") != config.identity
        or payload.get("interface") != str(interface)
        or payload.get("exposure_role") != "training"
        or payload.get("capabilities") != sorted(MICRO_CAPABILITIES)
        or payload.get("step") != config.steps
        or any(type(count) is not int or count < 0 for _, count in pair_counts)
        or sum(count for _, count in pair_counts) != config.steps
        or payload.get("model_state_identity") != model_state_identity
    ):
        raise CohortV2MicroError("symbolic-interface checkpoint provenance is stale or malformed")
    predictor = build_interface_predictor(config, interface)
    try:
        predictor.load_state_dict(model_state, strict=True)
    except RuntimeError as error:
        raise CohortV2MicroError(f"symbolic-interface model state is invalid: {error}") from error
    predictor.to(device).eval()
    trainable = sum(parameter.numel() for parameter in predictor.parameters())
    adapter = sum(parameter.numel() for parameter in predictor.micro_adapter.parameters())
    if (
        payload.get("trainable_parameter_count") != trainable
        or payload.get("adapter_parameter_count") != adapter
    ):
        raise CohortV2MicroError("symbolic-interface parameter accounting differs")
    codec = CohortV2StateCodec(
        latent_dim=config.latent_dim, max_entities=config.max_entities
    )
    return predictor, codec, SymbolicInterfaceCheckpoint(
        Path(path), expected_identity, interface, config.steps, trainable, adapter
    )


@dataclass(frozen=True, slots=True)
class RelationCounts:
    threshold: float
    true_positive: int
    false_positive: int
    false_negative: int
    query_count: int

    @property
    def f1(self) -> float:
        denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        return 0.0 if denominator == 0 else 2 * self.true_positive / denominator


def _active_entity_ids(frame: CohortV2CentralFrameRecord) -> tuple[str, ...]:
    entities = frame.engine_state.get("entities")
    if not isinstance(entities, tuple):
        raise CohortV2MicroError("relation metric engine entities are malformed")
    return tuple(sorted(
        str(entity["entity_id"])
        for entity in entities
        if isinstance(entity, Mapping) and entity.get("lifecycle") == "active"
    ))


def _queries_and_labels(
    frame: CohortV2CentralFrameRecord, predicate: str
) -> tuple[tuple[tuple[str, str], ...], tuple[bool, ...]]:
    ids = _active_entity_ids(frame)
    if predicate == "contact":
        queries = tuple(
            (first, second)
            for index, first in enumerate(ids)
            for second in ids[index + 1:]
        )
        positives = {tuple(sorted(item)) for item in frame.labels[predicate]["relations"]}
        labels = tuple(tuple(sorted(query)) in positives for query in queries)
    else:
        queries = tuple(
            (first, second) for first in ids for second in ids if first != second
        )
        positives = set(frame.labels[predicate]["relations"])
        labels = tuple(query in positives for query in queries)
    return queries, labels


def collect_relation_probabilities(
    predictor: DualOutputPredictor,
    codec: CohortV2StateCodec,
    reader: CohortV2ReleaseReader,
    *,
    batch_size: int,
    limit: int | None = None,
    progress_label: str | None = None,
) -> dict[str, tuple[tuple[float, bool], ...]]:
    windows = tuple(
        window
        for window in CohortV2OracleWindowDataset(reader, requested_horizons=(1, 5, 15))
        if all(
            window.context.labels[predicate].get("availability") == "available"
            and window.target.labels[predicate].get("availability") == "available"
            for predicate in PREDICATES
        )
    )
    if limit is not None:
        windows = windows[:limit]
    device = next(predictor.parameters()).device
    collected: dict[str, list[tuple[float, bool]]] = {predicate: [] for predicate in PREDICATES}
    with torch.no_grad():
        for offset in range(0, len(windows), batch_size):
            batch = windows[offset:offset + batch_size]
            contexts = codec.batch(tuple(window.context for window in batch)).to(device)
            actions = torch.stack(tuple(cohort_v2_action(window) for window in batch)).to(device)
            pair = PredictionPair(batch[0].requested_horizon, Abstraction.MICRO)
            # Horizon batches are regrouped because the source dataset is state-major.
            by_horizon: dict[int, list[int]] = {}
            for index, window in enumerate(batch):
                by_horizon.setdefault(window.requested_horizon, []).append(index)
            for horizon, indices in by_horizon.items():
                selected = tuple(batch[index] for index in indices)
                selected_contexts = contexts[indices]
                selected_actions = actions[indices]
                request = build_cohort_v2_transition_request(
                    PredictionPair(horizon, Abstraction.MICRO), selected
                )
                carrier = predictor.carrier(selected_contexts, selected_actions, request)
                for predicate in PREDICATES:
                    query_batches = []
                    label_batches = []
                    for window in selected:
                        queries, labels = _queries_and_labels(window.target, predicate)
                        query_batches.append(queries)
                        label_batches.append(labels)
                    logits = predictor.micro_head.relation_logits(
                        carrier, predicate, tuple(query_batches)
                    )
                    for sample_logits, labels in zip(logits, label_batches, strict=True):
                        probabilities = torch.sigmoid(sample_logits).cpu().tolist()
                        collected[predicate].extend(zip(probabilities, labels, strict=True))
            completed = offset + len(batch)
            if progress_label and (
                offset == 0
                or completed == len(windows)
                or completed % 512 < batch_size
            ):
                print(
                    f"[relations {progress_label} {completed}/{len(windows)}]",
                    flush=True,
                )
    return {predicate: tuple(values) for predicate, values in collected.items()}


def relation_counts(
    values: tuple[tuple[float, bool], ...], threshold: float
) -> RelationCounts:
    true_positive = sum(probability >= threshold and label for probability, label in values)
    false_positive = sum(probability >= threshold and not label for probability, label in values)
    false_negative = sum(probability < threshold and label for probability, label in values)
    return RelationCounts(
        threshold, true_positive, false_positive, false_negative, len(values)
    )


def calibrate_relation_thresholds(
    values: dict[str, tuple[tuple[float, bool], ...]]
) -> dict[str, float]:
    thresholds = tuple(index / 20 for index in range(1, 20))
    return {
        predicate: max(
            thresholds,
            key=lambda threshold: (
                relation_counts(values[predicate], threshold).f1,
                -abs(threshold - 0.5),
            ),
        )
        for predicate in PREDICATES
    }


def score_relations(
    values: dict[str, tuple[tuple[float, bool], ...]],
    thresholds: dict[str, float],
) -> dict[str, RelationCounts]:
    return {
        predicate: relation_counts(values[predicate], thresholds[predicate])
        for predicate in PREDICATES
    }


def select_symbolic_interface(
    macro_f1: Mapping[SymbolicInterface, float],
    *,
    material_gain: float = MATERIAL_RELATION_F1_GAIN,
) -> tuple[SymbolicInterface, tuple[dict[str, object], ...]]:
    if set(macro_f1) != set(INTERFACE_ORDER):
        raise CohortV2MicroError("interface decision requires all four variants")
    retained = INTERFACE_ORDER[0]
    comparisons = []
    for candidate in INTERFACE_ORDER[1:]:
        gain = float(macro_f1[candidate]) - float(macro_f1[retained])
        promoted = gain >= material_gain
        comparisons.append({
            "candidate": str(candidate),
            "gain_over_retained": gain,
            "material_gain": material_gain,
            "promoted": promoted,
            "retained_before": str(retained),
        })
        if promoted:
            retained = candidate
    return retained, tuple(comparisons)


def interface_compute_macs(
    interface: SymbolicInterface,
    hidden_dim: int,
    contact_count: int,
    support_count: int,
    entity_count: int,
) -> int:
    h2 = hidden_dim * hidden_dim
    if interface is SymbolicInterface.NO_SYMBOL:
        return 5 * h2 + 2 * hidden_dim
    if interface is SymbolicInterface.ORDERED_FLAT:
        return 3 * h2 * (contact_count + support_count) + 2 * h2 + 2 * hidden_dim
    directed_edges = 2 * contact_count + support_count
    result = 3 * h2 * directed_edges + h2 * entity_count + h2 + 2 * hidden_dim
    if interface is SymbolicInterface.SPSG:
        result += hidden_dim * directed_edges
    return result


__all__ = [
    "INTERFACE_ORDER",
    "MATERIAL_RELATION_F1_GAIN",
    "MatchedMicroInterfaceAdapter",
    "RelationCounts",
    "SymbolicInterface",
    "SymbolicInterfaceCheckpoint",
    "CohortV2SymbolicInterfaceTrainer",
    "build_interface_predictor",
    "calibrate_relation_thresholds",
    "collect_relation_probabilities",
    "interface_compute_macs",
    "load_symbolic_interface_checkpoint",
    "relation_counts",
    "save_symbolic_interface_checkpoint",
    "score_relations",
    "select_symbolic_interface",
]
