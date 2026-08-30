"""Training-role feature parser and learned-symbol transition inputs for issue #16."""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from world_model.data import (
    CohortV2CentralFrameRecord,
    CohortV2OracleWindow,
    CohortV2ReleaseReader,
)
from world_model.model import (
    Abstraction,
    BooleanTransitionValue,
    MacroTransitionBatch,
    MacroTransitionInput,
    MicroTransitionBatch,
    MicroTransitionInput,
    PredictionPair,
    RelationTransitionValue,
    TransitionRequest,
    identity,
)
from world_model.training.cohort_v2_micro import cohort_v2_model_state_identity
from world_model.training.grid_artifacts import canonical_json_bytes


FEATURE_PARSER_SCHEMA: Final = "cohort_v2_feature_parser_checkpoint_v1"
FEATURE_PARSER_MANIFEST_SCHEMA: Final = "cohort_v2_feature_parser_manifest_v1"
RELATION_PREDICATES: Final = ("contact", "supports")
MACRO_PREDICATES: Final = ("steady-state", "structure-unstable")
PREDICATES: Final = RELATION_PREDICATES + MACRO_PREDICATES
ENTITY_KINEMATIC_WIDTH: Final = 13
ENTITY_KINDS: Final = (
    "bird",
    "pig",
    "block",
    "platform",
    "slingshot",
    "world",
    "other",
)
ENTITY_FEATURE_WIDTH: Final = ENTITY_KINEMATIC_WIDTH + len(ENTITY_KINDS)
RELATION_FEATURE_WIDTH: Final = ENTITY_FEATURE_WIDTH * 3
MACRO_FEATURE_WIDTH: Final = ENTITY_FEATURE_WIDTH * 3 + 1


class CohortV2FeatureParserError(ValueError):
    """The issue-16 parser, source split, or checkpoint is invalid."""


@dataclass(frozen=True, slots=True)
class CohortV2FeatureParserConfig:
    seed: int = 20260828
    hidden_dim: int = 64
    epochs: int = 15
    relation_batch_size: int = 4096
    macro_batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        for field in (
            "seed",
            "hidden_dim",
            "epochs",
            "relation_batch_size",
            "macro_batch_size",
        ):
            value = getattr(self, field)
            minimum = 0 if field == "seed" else 1
            if type(value) is not int or value < minimum:
                raise CohortV2FeatureParserError(
                    f"{field} must be an integer >= {minimum}"
                )
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise CohortV2FeatureParserError("optimizer configuration is invalid")
        if not self.device:
            raise CohortV2FeatureParserError("device must be nonempty")

    @property
    def identity(self) -> str:
        return identity((
            "cohort-v2-feature-parser-config-v1",
            self.seed,
            self.hidden_dim,
            self.epochs,
            self.relation_batch_size,
            self.macro_batch_size,
            self.learning_rate,
            self.weight_decay,
            "body-kinematics+object-kind;oracle-relation-proxies-excluded",
        ))


@dataclass(frozen=True, slots=True)
class FeatureParserRoleData:
    exposure_role: str
    attempt_ids: tuple[str, ...]
    scenario_lineage_identities: tuple[str, ...]
    features: Mapping[str, torch.Tensor]
    labels: Mapping[str, torch.Tensor]
    available_frame_counts: Mapping[str, int]
    unavailable_frame_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class ParsedFrameSymbols:
    frame_record_identity: str
    contact: RelationTransitionValue
    supports: RelationTransitionValue
    steady_state: BooleanTransitionValue
    structure_unstable: BooleanTransitionValue


@dataclass(frozen=True, slots=True)
class CohortV2FeatureParserCheckpoint:
    path: Path
    identity: str
    model_state_identity: str
    config: CohortV2FeatureParserConfig
    temperatures: Mapping[str, float]
    thresholds: Mapping[str, float]
    source_bindings: Mapping[str, Any]


class CohortV2FeatureParser(nn.Module):
    """Small supervised parser over observable kinematics and object kinds."""

    def __init__(
        self,
        config: CohortV2FeatureParserConfig,
        *,
        relation_mean: torch.Tensor,
        relation_scale: torch.Tensor,
        macro_mean: torch.Tensor,
        macro_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("relation_mean", relation_mean.clone().float())
        self.register_buffer("relation_scale", relation_scale.clone().float())
        self.register_buffer("macro_mean", macro_mean.clone().float())
        self.register_buffer("macro_scale", macro_scale.clone().float())
        self.relation_network = nn.Sequential(
            nn.Linear(RELATION_FEATURE_WIDTH, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, len(RELATION_PREDICATES)),
        )
        self.macro_network = nn.Sequential(
            nn.Linear(MACRO_FEATURE_WIDTH, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, len(MACRO_PREDICATES)),
        )

    def relation_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.relation_network(
            (features - self.relation_mean) / self.relation_scale
        )

    def macro_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.macro_network((features - self.macro_mean) / self.macro_scale)


def _number(value: Any, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise CohortV2FeatureParserError(f"{field} is not finite")
    return float(value)


def _entity_kind(entity: Mapping[str, Any]) -> str:
    scenario_id = entity.get("scenario_object_id")
    if type(scenario_id) is not str or not scenario_id:
        return "other"
    prefix = scenario_id.split(":", 1)[0]
    return prefix if prefix in ENTITY_KINDS[:-1] else "other"


def entity_observable_features(entity: Mapping[str, Any]) -> torch.Tensor:
    """Encode only body kinematics and object kind, excluding relation proxies."""
    body_present = entity.get("body_present") is True
    body = entity.get("body")
    body_types = {"static": -1.0, "kinematic": 0.0, "dynamic": 1.0}
    values = [
        float(entity.get("lifecycle") == "active"),
        float(body_present),
    ]
    if body_present:
        if not isinstance(body, Mapping):
            raise CohortV2FeatureParserError("present entity body is malformed")
        position = body.get("position")
        velocity = body.get("velocity")
        body_type = body.get("body_type")
        if (
            not isinstance(position, tuple)
            or len(position) != 2
            or not isinstance(velocity, tuple)
            or len(velocity) != 2
            or body_type not in body_types
        ):
            raise CohortV2FeatureParserError("entity kinematics are malformed")
        rotation = math.radians(_number(body.get("rotation_degrees"), "rotation"))
        values.extend((
            body_types[str(body_type)],
            float(body.get("simulated") is True),
            float(body.get("gravity_applicable") is True),
            _number(body.get("gravity_scale"), "gravity scale"),
            _number(position[0], "position x") / 20.0,
            _number(position[1], "position y") / 20.0,
            _number(velocity[0], "velocity x") / 20.0,
            _number(velocity[1], "velocity y") / 20.0,
            math.sin(rotation),
            math.cos(rotation),
            _number(
                body.get("angular_velocity_degrees_per_second"),
                "angular velocity",
            ) / 360.0,
        ))
    else:
        values.extend((0.0,) * (ENTITY_KINEMATIC_WIDTH - 2))
    kind = _entity_kind(entity)
    values.extend(float(kind == candidate) for candidate in ENTITY_KINDS)
    if len(values) != ENTITY_FEATURE_WIDTH:
        raise AssertionError("entity feature declaration drifted")
    return torch.tensor(values, dtype=torch.float32)


def _active_entities(
    frame: CohortV2CentralFrameRecord,
) -> tuple[tuple[str, torch.Tensor], ...]:
    entities = frame.engine_state.get("entities")
    if not isinstance(entities, tuple):
        raise CohortV2FeatureParserError("engine state entities are malformed")
    result = []
    for entity in entities:
        if not isinstance(entity, Mapping) or entity.get("lifecycle") != "active":
            continue
        entity_id = entity.get("entity_id")
        if type(entity_id) is not str or not entity_id:
            raise CohortV2FeatureParserError("active entity identity is malformed")
        result.append((entity_id, entity_observable_features(entity)))
    return tuple(sorted(result, key=lambda item: item[0]))


def _relation_queries(
    frame: CohortV2CentralFrameRecord, predicate: str
) -> tuple[tuple[tuple[str, str], ...], torch.Tensor]:
    entities = _active_entities(frame)
    rows: list[torch.Tensor] = []
    queries: list[tuple[str, str]] = []
    if predicate == "contact":
        pairs = (
            (first, second)
            for index, first in enumerate(entities)
            for second in entities[index + 1 :]
        )
        for (first_id, first), (second_id, second) in pairs:
            queries.append((first_id, second_id))
            rows.append(torch.cat((first + second, (first - second).abs(), first * second)))
    elif predicate == "supports":
        for first_id, first in entities:
            for second_id, second in entities:
                if first_id == second_id:
                    continue
                queries.append((first_id, second_id))
                rows.append(torch.cat((first, second, first - second)))
    else:
        raise CohortV2FeatureParserError(f"unknown relation predicate {predicate}")
    features = (
        torch.stack(rows)
        if rows
        else torch.empty((0, RELATION_FEATURE_WIDTH), dtype=torch.float32)
    )
    return tuple(queries), features


def _macro_features(frame: CohortV2CentralFrameRecord) -> torch.Tensor:
    entities = _active_entities(frame)
    if not entities:
        raise CohortV2FeatureParserError("macro features require an active entity")
    stacked = torch.stack(tuple(features for _, features in entities))
    return torch.cat((
        stacked.mean(dim=0),
        stacked.amin(dim=0),
        stacked.amax(dim=0),
        torch.tensor((len(entities) / 15.0,), dtype=torch.float32),
    ))


def _available_label(frame: CohortV2CentralFrameRecord, predicate: str) -> Mapping[str, Any] | None:
    label = frame.labels[predicate]
    availability = label.get("availability")
    if availability == "available":
        return label
    if isinstance(availability, str) and availability.startswith("unavailable_"):
        return None
    raise CohortV2FeatureParserError(f"{predicate} availability is malformed")


def build_feature_parser_role_data(
    reader: CohortV2ReleaseReader,
    *,
    expected_role: str,
    frame_limit: int | None = None,
) -> FeatureParserRoleData:
    roles = {rollout.exposure_role for rollout in reader.rollouts}
    if roles != {expected_role}:
        raise CohortV2FeatureParserError(
            f"parser {expected_role} source crosses exposure roles: {sorted(roles)}"
        )
    feature_rows: dict[str, list[torch.Tensor]] = {predicate: [] for predicate in PREDICATES}
    label_rows: dict[str, list[float]] = {predicate: [] for predicate in PREDICATES}
    available = {predicate: 0 for predicate in PREDICATES}
    unavailable = {predicate: 0 for predicate in PREDICATES}
    seen = 0
    for rollout in reader.rollouts:
        for frame in rollout.frame_records:
            if frame_limit is not None and seen >= frame_limit:
                break
            seen += 1
            for predicate in RELATION_PREDICATES:
                label = _available_label(frame, predicate)
                if label is None:
                    unavailable[predicate] += 1
                    continue
                available[predicate] += 1
                queries, features = _relation_queries(frame, predicate)
                positives = set(label["relations"])
                if predicate == "contact":
                    positives = {tuple(sorted(item)) for item in positives}
                feature_rows[predicate].extend(features)
                label_rows[predicate].extend(
                    float((tuple(sorted(query)) if predicate == "contact" else query) in positives)
                    for query in queries
                )
            macro = _macro_features(frame)
            for predicate in MACRO_PREDICATES:
                label = _available_label(frame, predicate)
                if label is None:
                    unavailable[predicate] += 1
                    continue
                available[predicate] += 1
                feature_rows[predicate].append(macro)
                label_rows[predicate].append(float(label["value"]))
        if frame_limit is not None and seen >= frame_limit:
            break
    features = {
        predicate: torch.stack(rows)
        for predicate, rows in feature_rows.items()
    }
    labels = {
        predicate: torch.tensor(label_rows[predicate], dtype=torch.float32)
        for predicate in PREDICATES
    }
    return FeatureParserRoleData(
        expected_role,
        tuple(sorted(rollout.attempt_id for rollout in reader.rollouts)),
        tuple(sorted({rollout.scenario_lineage_identity for rollout in reader.rollouts})),
        features,
        labels,
        available,
        unavailable,
    )


def _normalization(data: FeatureParserRoleData) -> tuple[torch.Tensor, ...]:
    relation = torch.cat(tuple(data.features[p] for p in RELATION_PREDICATES))
    macro = torch.cat(tuple(data.features[p] for p in MACRO_PREDICATES))
    relation_scale = relation.std(dim=0, unbiased=False).clamp_min(1e-6)
    macro_scale = macro.std(dim=0, unbiased=False).clamp_min(1e-6)
    return relation.mean(dim=0), relation_scale, macro.mean(dim=0), macro_scale


def build_feature_parser_model(
    config: CohortV2FeatureParserConfig,
    training_data: FeatureParserRoleData,
) -> CohortV2FeatureParser:
    if training_data.exposure_role != "training":
        raise CohortV2FeatureParserError("learned parser parameters require training role")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    relation_mean, relation_scale, macro_mean, macro_scale = _normalization(
        training_data
    )
    model = CohortV2FeatureParser(
        config,
        relation_mean=relation_mean,
        relation_scale=relation_scale,
        macro_mean=macro_mean,
        macro_scale=macro_scale,
    )
    return model.to(torch.device(config.device))


def _batches(count: int, size: int, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
    order = torch.randperm(count, generator=generator)
    return tuple(order[start : start + size] for start in range(0, count, size))


def train_feature_parser(
    model: CohortV2FeatureParser,
    data: FeatureParserRoleData,
    *,
    progress: Callable[[str], None] | None = None,
) -> None:
    if data.exposure_role != "training":
        raise CohortV2FeatureParserError("learned parser parameters require training role")
    config = model.config
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generators = {
        predicate: torch.Generator().manual_seed(config.seed + index)
        for index, predicate in enumerate(PREDICATES)
    }
    model.train()
    for epoch in range(1, config.epochs + 1):
        losses = []
        for predicate_index, predicate in enumerate(PREDICATES):
            features = data.features[predicate]
            labels = data.labels[predicate]
            batch_size = (
                config.relation_batch_size
                if predicate in RELATION_PREDICATES
                else config.macro_batch_size
            )
            positives = float(labels.sum())
            negatives = float(labels.numel() - positives)
            positive_weight = negatives / positives if positives else 1.0
            for indices in _batches(labels.numel(), batch_size, generators[predicate]):
                batch_features = features[indices].to(device)
                batch_labels = labels[indices].to(device)
                logits = (
                    model.relation_logits(batch_features)[:, predicate_index]
                    if predicate in RELATION_PREDICATES
                    else model.macro_logits(batch_features)[:, predicate_index - 2]
                )
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    batch_labels,
                    pos_weight=torch.tensor(positive_weight, device=device),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
        if progress is not None and (
            epoch == 1 or epoch == config.epochs or epoch % 5 == 0
        ):
            progress(
                f"[parser train hidden={config.hidden_dim} epoch {epoch}/{config.epochs}] "
                f"mean_loss={sum(losses) / len(losses):.6f}"
            )
    model.eval()


def _probabilities(
    model: CohortV2FeatureParser,
    data: FeatureParserRoleData,
    temperatures: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    result = {}
    with torch.no_grad():
        for index, predicate in enumerate(PREDICATES):
            features = data.features[predicate].to(device)
            logits = (
                model.relation_logits(features)[:, index]
                if predicate in RELATION_PREDICATES
                else model.macro_logits(features)[:, index - 2]
            )
            if temperatures is not None:
                logits = logits / float(temperatures[predicate])
            result[predicate] = torch.sigmoid(logits).cpu().numpy()
    return result


def _binary_counts(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = probabilities >= threshold
    truth = labels.astype(bool)
    true_positive = int(np.logical_and(predicted, truth).sum())
    false_positive = int(np.logical_and(predicted, ~truth).sum())
    false_negative = int(np.logical_and(~predicted, truth).sum())
    true_negative = int(np.logical_and(~predicted, ~truth).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "agreement": (true_positive + true_negative) / len(labels),
    }


def calibrate_feature_parser_probabilities(
    model: CohortV2FeatureParser,
    calibration_data: FeatureParserRoleData,
) -> dict[str, float]:
    """Choose one fixed temperature per predicate on calibration NLL."""
    if calibration_data.exposure_role != "calibration":
        raise CohortV2FeatureParserError("probability calibration requires calibration role")
    raw = _probabilities(model, calibration_data)
    candidates = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    result = {}
    for predicate in PREDICATES:
        labels = calibration_data.labels[predicate].numpy()
        raw_values = np.clip(raw[predicate], 1e-7, 1.0 - 1e-7)
        logits = np.log(raw_values / (1.0 - raw_values))

        def nll(temperature: float) -> float:
            values = 1.0 / (1.0 + np.exp(-logits / temperature))
            values = np.clip(values, 1e-7, 1.0 - 1e-7)
            return float(
                -(labels * np.log(values) + (1.0 - labels) * np.log(1.0 - values)).mean()
            )

        result[predicate] = min(
            candidates, key=lambda temperature: (nll(temperature), abs(temperature - 1.0))
        )
    return result


def calibrate_feature_parser_thresholds(
    model: CohortV2FeatureParser,
    calibration_data: FeatureParserRoleData,
    temperatures: Mapping[str, float],
) -> dict[str, float]:
    if calibration_data.exposure_role != "calibration":
        raise CohortV2FeatureParserError("thresholds require the calibration role")
    probabilities = _probabilities(model, calibration_data, temperatures)
    thresholds = tuple(index / 20.0 for index in range(1, 20))
    return {
        predicate: max(
            thresholds,
            key=lambda threshold: (
                _binary_counts(
                    probabilities[predicate],
                    calibration_data.labels[predicate].numpy(),
                    threshold,
                )["f1"],
                -abs(threshold - 0.5),
            ),
        )
        for predicate in PREDICATES
    }


def feature_parser_metrics(
    model: CohortV2FeatureParser,
    data: FeatureParserRoleData,
    thresholds: Mapping[str, float] | None = None,
    temperatures: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    probabilities = _probabilities(model, data, temperatures)
    result = {}
    for predicate in PREDICATES:
        labels = data.labels[predicate].numpy()
        values = np.clip(probabilities[predicate], 1e-7, 1.0 - 1e-7)
        bins = np.minimum((values * 10).astype(int), 9)
        ece = 0.0
        for bin_index in range(10):
            selected = bins == bin_index
            if selected.any():
                ece += float(selected.mean()) * abs(
                    float(values[selected].mean()) - float(labels[selected].mean())
                )
        threshold = 0.5 if thresholds is None else float(thresholds[predicate])
        result[predicate] = {
            "sample_count": int(len(labels)),
            "positive_count": int(labels.sum()),
            "available_frame_count": data.available_frame_counts[predicate],
            "unavailable_frame_count": data.unavailable_frame_counts[predicate],
            "brier_score": float(np.square(values - labels).mean()),
            "negative_log_likelihood": float(
                -(labels * np.log(values) + (1.0 - labels) * np.log(1.0 - values)).mean()
            ),
            "expected_calibration_error_10_bin": ece,
            "threshold": threshold,
            "temperature": 1.0 if temperatures is None else float(temperatures[predicate]),
            **_binary_counts(values, labels, threshold),
        }
    result["mean_predicate_negative_log_likelihood"] = float(
        np.mean([result[p]["negative_log_likelihood"] for p in PREDICATES])
    )
    result["directed_predicate_agreement"] = result["supports"]["agreement"]
    result["macro_event_agreement"] = float(np.mean([
        result[p]["agreement"] for p in MACRO_PREDICATES
    ]))
    return result


def select_feature_parser(
    candidates: Sequence[CohortV2FeatureParser],
    model_selection_data: FeatureParserRoleData,
) -> tuple[CohortV2FeatureParser, list[dict[str, Any]]]:
    if model_selection_data.exposure_role != "model_selection":
        raise CohortV2FeatureParserError("configuration selection requires model-selection role")
    rows = []
    for model in candidates:
        metrics = feature_parser_metrics(model, model_selection_data)
        rows.append({
            "configuration_identity": model.config.identity,
            "hidden_dim": model.config.hidden_dim,
            "mean_predicate_negative_log_likelihood": metrics[
                "mean_predicate_negative_log_likelihood"
            ],
            "metrics": metrics,
        })
    selected_index = min(
        range(len(candidates)),
        key=lambda index: (
            rows[index]["mean_predicate_negative_log_likelihood"],
            candidates[index].config.hidden_dim,
        ),
    )
    for index, row in enumerate(rows):
        row["selected"] = index == selected_index
    return candidates[selected_index], rows


def parse_frame_symbols(
    model: CohortV2FeatureParser,
    frame: CohortV2CentralFrameRecord,
    temperatures: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> ParsedFrameSymbols:
    device = next(model.parameters()).device
    relation_values = {}
    with torch.no_grad():
        for predicate_index, predicate in enumerate(RELATION_PREDICATES):
            source = frame.labels[predicate]
            availability = source.get("availability")
            if availability != "available":
                relation_values[predicate] = RelationTransitionValue(str(availability), None)
                continue
            queries, features = _relation_queries(frame, predicate)
            probabilities = torch.sigmoid(
                model.relation_logits(features.to(device))[:, predicate_index]
                / float(temperatures[predicate])
            ).cpu()
            relation_values[predicate] = RelationTransitionValue(
                "available",
                tuple(
                    query
                    for query, probability in zip(queries, probabilities, strict=True)
                    if float(probability) >= thresholds[predicate]
                ),
            )
        macro_features = _macro_features(frame).unsqueeze(0).to(device)
        macro_logits = model.macro_logits(macro_features)[0]
        macro_probabilities = torch.sigmoid(torch.stack(tuple(
            macro_logits[index] / float(temperatures[predicate])
            for index, predicate in enumerate(MACRO_PREDICATES)
        ))).cpu()
    macro_values = {}
    for index, predicate in enumerate(MACRO_PREDICATES):
        availability = frame.labels[predicate].get("availability")
        macro_values[predicate] = BooleanTransitionValue(
            str(availability),
            bool(float(macro_probabilities[index]) >= thresholds[predicate])
            if availability == "available"
            else None,
        )
    return ParsedFrameSymbols(
        frame.identity,
        relation_values["contact"],
        relation_values["supports"],
        macro_values["steady-state"],
        macro_values["structure-unstable"],
    )


class LearnedFeatureTransitionRequestBuilder:
    def __init__(
        self,
        parsed_frames: Mapping[str, ParsedFrameSymbols],
        checkpoint_identity: str,
        source_identity: str,
    ) -> None:
        if not source_identity:
            raise CohortV2FeatureParserError("learned symbol source identity is required")
        self.parsed_frames = dict(parsed_frames)
        self.identity = identity((
            "cohort-v2-learned-feature-symbol-input-v1",
            checkpoint_identity,
            source_identity,
        ))

    def __call__(
        self,
        pair: PredictionPair,
        windows: tuple[CohortV2OracleWindow, ...],
    ) -> TransitionRequest:
        if pair.abstraction is Abstraction.CONTINUOUS:
            return TransitionRequest(pair, None)
        symbols = tuple(self.parsed_frames[window.context.identity] for window in windows)
        if pair.abstraction is Abstraction.MICRO:
            return TransitionRequest(pair, MicroTransitionBatch(tuple(
                MicroTransitionInput(
                    symbol.frame_record_identity,
                    symbol.contact,
                    symbol.supports,
                )
                for symbol in symbols
            )))
        return TransitionRequest(pair, MacroTransitionBatch(tuple(
            MacroTransitionInput(
                symbol.frame_record_identity,
                symbol.steady_state,
                symbol.structure_unstable,
            )
            for symbol in symbols
        )))


def parse_reader_frames(
    model: CohortV2FeatureParser,
    reader: CohortV2ReleaseReader,
    temperatures: Mapping[str, float],
    thresholds: Mapping[str, float],
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, ParsedFrameSymbols]:
    frames = tuple(frame for rollout in reader.rollouts for frame in rollout.frame_records)
    parsed = {}
    for index, frame in enumerate(frames, start=1):
        parsed[frame.identity] = parse_frame_symbols(
            model, frame, temperatures, thresholds
        )
        if progress is not None and (index == len(frames) or index % 250 == 0):
            progress(f"[parser infer {index}/{len(frames)}] frame={frame.identity}")
    return parsed


def parser_coherence(
    reader: CohortV2ReleaseReader,
    parsed: Mapping[str, ParsedFrameSymbols],
) -> dict[str, float | int]:
    support_count = support_without_contact = self_relation_count = 0
    for rollout in reader.rollouts:
        for frame in rollout.frame_records:
            value = parsed[frame.identity]
            if value.contact.available and value.supports.available:
                contacts = {tuple(sorted(item)) for item in value.contact.relations or ()}
                for relation in value.supports.relations or ():
                    support_count += 1
                    self_relation_count += int(relation[0] == relation[1])
                    support_without_contact += int(tuple(sorted(relation)) not in contacts)
    return {
        "predicted_support_relation_count": support_count,
        "self_relation_rate": self_relation_count / support_count if support_count else 0.0,
        "supports_without_contact_rate": (
            support_without_contact / support_count if support_count else 0.0
        ),
    }


def _source_bindings(data: Sequence[FeatureParserRoleData], readers: Sequence[CohortV2ReleaseReader]) -> dict[str, Any]:
    return {
        "release_identity": readers[0].release_identity,
        "capability_declaration_identity": readers[0].capability_declaration_identity,
        "partition_identity": readers[0].partition_identity,
        "roles": {
            role_data.exposure_role: {
                "attempt_ids": list(role_data.attempt_ids),
                "scenario_lineage_identities": list(role_data.scenario_lineage_identities),
                "derivation_identity": reader.derivation_identity,
            }
            for role_data, reader in zip(data, readers, strict=True)
        },
        "learned_parameter_role": "training",
        "configuration_selection_role": "model_selection",
        "threshold_role": "calibration",
    }


def save_feature_parser_checkpoint(
    root: Path,
    model: CohortV2FeatureParser,
    temperatures: Mapping[str, float],
    thresholds: Mapping[str, float],
    *,
    role_data: Sequence[FeatureParserRoleData],
    readers: Sequence[CohortV2ReleaseReader],
    model_selection: Sequence[Mapping[str, Any]],
    calibration_metrics: Mapping[str, Any],
    implementation_revision: str,
) -> CohortV2FeatureParserCheckpoint:
    root = Path(root)
    if root.exists():
        raise CohortV2FeatureParserError("immutable parser checkpoint already exists")
    source_bindings = _source_bindings(role_data, readers)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    state_identity = cohort_v2_model_state_identity(state)
    checkpoint_identity = identity((
        FEATURE_PARSER_SCHEMA,
        model.config.identity,
        state_identity,
        tuple((predicate, float(temperatures[predicate])) for predicate in PREDICATES),
        tuple((predicate, float(thresholds[predicate])) for predicate in PREDICATES),
        source_bindings,
        implementation_revision,
    ))
    payload = {
        "schema": FEATURE_PARSER_SCHEMA,
        "checkpoint_identity": checkpoint_identity,
        "model_state_identity": state_identity,
        "config": asdict(model.config),
        "temperatures": {predicate: float(temperatures[predicate]) for predicate in PREDICATES},
        "thresholds": {predicate: float(thresholds[predicate]) for predicate in PREDICATES},
        "model_state": state,
        "source_bindings": source_bindings,
        "implementation_revision": implementation_revision,
    }
    root.mkdir(parents=True)
    temporary = root / "checkpoint.pt.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, root / "checkpoint.pt")
    checkpoint_sha = hashlib.sha256((root / "checkpoint.pt").read_bytes()).hexdigest()
    manifest = {
        "schema": FEATURE_PARSER_MANIFEST_SCHEMA,
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_sha256": f"sha256:{checkpoint_sha}",
        "model_state_identity": state_identity,
        "config": asdict(model.config),
        "temperatures": payload["temperatures"],
        "thresholds": payload["thresholds"],
        "source_bindings": source_bindings,
        "model_selection": list(model_selection),
        "calibration_metrics": dict(calibration_metrics),
        "implementation_revision": implementation_revision,
    }
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return CohortV2FeatureParserCheckpoint(
        root / "checkpoint.pt",
        checkpoint_identity,
        state_identity,
        model.config,
        payload["temperatures"],
        payload["thresholds"],
        source_bindings,
    )


def load_feature_parser_checkpoint(
    root: Path,
    *,
    readers: Sequence[CohortV2ReleaseReader],
    device: str,
) -> tuple[CohortV2FeatureParser, CohortV2FeatureParserCheckpoint, dict[str, Any]]:
    root = Path(root)
    try:
        manifest_bytes = (root / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        checkpoint_bytes = (root / "checkpoint.pt").read_bytes()
        payload = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=True)
        config = CohortV2FeatureParserConfig(**payload["config"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, RuntimeError, ValueError) as error:
        raise CohortV2FeatureParserError(f"cannot load parser checkpoint: {error}") from error
    if (
        canonical_json_bytes(manifest) != manifest_bytes
        or manifest.get("schema") != FEATURE_PARSER_MANIFEST_SCHEMA
        or manifest.get("checkpoint_sha256")
        != f"sha256:{hashlib.sha256(checkpoint_bytes).hexdigest()}"
        or payload.get("schema") != FEATURE_PARSER_SCHEMA
        or payload.get("checkpoint_identity") != manifest.get("checkpoint_identity")
        or payload.get("source_bindings") != manifest.get("source_bindings")
        or payload.get("temperatures") != manifest.get("temperatures")
        or payload.get("thresholds") != manifest.get("thresholds")
        or payload.get("model_state_identity")
        != cohort_v2_model_state_identity(payload.get("model_state", {}))
    ):
        raise CohortV2FeatureParserError("parser checkpoint provenance is stale")
    bindings = payload["source_bindings"]
    if (
        len(readers) != 3
        or bindings.get("release_identity") != readers[0].release_identity
        or bindings.get("capability_declaration_identity")
        != readers[0].capability_declaration_identity
        or bindings.get("partition_identity") != readers[0].partition_identity
    ):
        raise CohortV2FeatureParserError("parser checkpoint crosses its source release")
    for role, reader in zip(("training", "calibration", "model_selection"), readers, strict=True):
        expected = bindings["roles"][role]
        if (
            expected["attempt_ids"] != sorted(item.attempt_id for item in reader.rollouts)
            or expected["scenario_lineage_identities"]
            != sorted({item.scenario_lineage_identity for item in reader.rollouts})
            or expected["derivation_identity"] != reader.derivation_identity
        ):
            raise CohortV2FeatureParserError(f"parser {role} source binding differs")
    model = CohortV2FeatureParser(
        config,
        relation_mean=payload["model_state"]["relation_mean"],
        relation_scale=payload["model_state"]["relation_scale"],
        macro_mean=payload["model_state"]["macro_mean"],
        macro_scale=payload["model_state"]["macro_scale"],
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(torch.device(device)).eval()
    checkpoint = CohortV2FeatureParserCheckpoint(
        root / "checkpoint.pt",
        payload["checkpoint_identity"],
        payload["model_state_identity"],
        config,
        payload["temperatures"],
        payload["thresholds"],
        bindings,
    )
    return model, checkpoint, manifest


__all__ = [
    "CohortV2FeatureParser",
    "CohortV2FeatureParserCheckpoint",
    "CohortV2FeatureParserConfig",
    "CohortV2FeatureParserError",
    "FeatureParserRoleData",
    "LearnedFeatureTransitionRequestBuilder",
    "ParsedFrameSymbols",
    "PREDICATES",
    "build_feature_parser_model",
    "build_feature_parser_role_data",
    "calibrate_feature_parser_probabilities",
    "calibrate_feature_parser_thresholds",
    "feature_parser_metrics",
    "load_feature_parser_checkpoint",
    "parse_frame_symbols",
    "parse_reader_frames",
    "parser_coherence",
    "save_feature_parser_checkpoint",
    "select_feature_parser",
    "train_feature_parser",
]
