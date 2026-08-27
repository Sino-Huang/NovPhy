"""Frozen-encoder visual predicate parser for issue #17."""
from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn
from torch.nn import functional as F

from world_model.data import (
    CohortV2AlignedObservationReader,
    CohortV2CentralFrameRecord,
)
from world_model.model import BooleanTransitionValue, RelationTransitionValue, identity
from world_model.training.cohort_v2_feature_parser import (
    LearnedFeatureTransitionRequestBuilder,
    ParsedFrameSymbols,
)
from world_model.training.grid_artifacts import canonical_json_bytes
from world_model.training.cohort_v2_micro import cohort_v2_model_state_identity


SCHEMA: Final = "cohort_v2_visual_parser_checkpoint_v1"
MANIFEST_SCHEMA: Final = "cohort_v2_visual_parser_manifest_v1"
RELATION_PREDICATES: Final = ("contact", "supports")
MACRO_PREDICATES: Final = ("steady-state", "structure-unstable")
PREDICATES: Final = RELATION_PREDICATES + MACRO_PREDICATES
CALIBRATED_TARGETS: Final = ("object_presence",) + PREDICATES
ENTITY_KINDS: Final = (
    "bird", "pig", "block", "platform", "slingshot", "world", "other",
)


class CohortV2VisualParserError(ValueError):
    """The issue-17 visual parser, aligned source, or checkpoint is invalid."""


@dataclass(frozen=True, slots=True)
class CohortV2VisualParserConfig:
    seed: int = 20260829
    image_height: int = 32
    image_width: int = 48
    hidden_dim: int = 128
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        for field in (
            "seed", "image_height", "image_width", "hidden_dim", "epochs", "batch_size"
        ):
            value = getattr(self, field)
            minimum = 0 if field == "seed" else 1
            if type(value) is not int or value < minimum:
                raise CohortV2VisualParserError(
                    f"{field} must be an integer >= {minimum}"
                )
        if self.learning_rate <= 0 or self.weight_decay < 0 or not self.device:
            raise CohortV2VisualParserError("visual parser optimizer or device is invalid")

    @property
    def identity(self) -> str:
        return identity((
            "cohort-v2-visual-parser-config-v1",
            *asdict(self).values(),
            "frozen-rgb+sobel-encoder",
            "fixed-scenario-object-query-vocabulary",
        ))


@dataclass(frozen=True, slots=True)
class VisualParserRoleData:
    exposure_role: str
    attempt_ids: tuple[str, ...]
    scenario_lineage_identities: tuple[str, ...]
    frame_identities: tuple[str, ...]
    images: torch.Tensor
    presence: torch.Tensor
    centers: torch.Tensor
    relation_labels: torch.Tensor
    relation_mask: torch.Tensor
    macro_labels: torch.Tensor
    macro_mask: torch.Tensor
    object_vocabulary: tuple[str, ...]
    available_frame_counts: Mapping[str, int]
    unavailable_frame_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class CohortV2VisualParserCheckpoint:
    path: Path
    identity: str
    model_state_identity: str
    config: CohortV2VisualParserConfig
    object_vocabulary: tuple[str, ...]
    temperatures: Mapping[str, float]
    thresholds: Mapping[str, float]
    object_kind_temperature: float
    source_bindings: Mapping[str, Any]


class FrozenVisualEncoder(nn.Module):
    """Parameter-free spatial RGB and Sobel features; never updated downstream."""

    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.register_buffer(
            "sobel_x",
            torch.tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1)), dtype=torch.float32)
            .view(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(((-1, -2, -1), (0, 0, 0), (1, 2, 1)), dtype=torch.float32)
            .view(1, 1, 3, 3),
        )

    @property
    def output_dim(self) -> int:
        return 5 * self.height * self.width

    @property
    def identity(self) -> str:
        return identity((
            "cohort-v2-frozen-visual-encoder-v1",
            self.height,
            self.width,
            "rgb8-srgb-bilinear-resize+grayscale-sobel",
        ))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        value = images.float() / 255.0
        gray = (
            0.299 * value[:, 0:1]
            + 0.587 * value[:, 1:2]
            + 0.114 * value[:, 2:3]
        )
        edge_x = F.conv2d(gray, self.sobel_x, padding=1).abs().clamp_max(4.0) / 4.0
        edge_y = F.conv2d(gray, self.sobel_y, padding=1).abs().clamp_max(4.0) / 4.0
        return torch.cat((value, edge_x, edge_y), dim=1).flatten(1)


class CohortV2VisualPredicateParser(nn.Module):
    def __init__(
        self,
        config: CohortV2VisualParserConfig,
        object_vocabulary: tuple[str, ...],
    ) -> None:
        super().__init__()
        if not object_vocabulary or len(object_vocabulary) > 15:
            raise CohortV2VisualParserError("visual object vocabulary must contain 1-15 slots")
        self.config = config
        self.object_vocabulary = object_vocabulary
        self.encoder = FrozenVisualEncoder(config.image_height, config.image_width)
        self.backbone = nn.Sequential(
            nn.Linear(self.encoder.output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.queries = nn.Parameter(
            torch.empty(len(object_vocabulary), config.hidden_dim)
        )
        nn.init.normal_(self.queries, std=0.02)
        self.presence_head = nn.Linear(config.hidden_dim, 1)
        self.center_head = nn.Linear(config.hidden_dim, 2)
        self.kind_head = nn.Linear(config.hidden_dim, len(ENTITY_KINDS))
        self.relation_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, len(RELATION_PREDICATES)),
        )
        self.macro_head = nn.Linear(config.hidden_dim, len(MACRO_PREDICATES))
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(images)
        global_features = self.backbone(encoded)
        slots = global_features.unsqueeze(1) + self.queries.unsqueeze(0)
        first = slots.unsqueeze(2).expand(-1, -1, len(self.object_vocabulary), -1)
        second = slots.unsqueeze(1).expand(-1, len(self.object_vocabulary), -1, -1)
        return {
            "presence_logits": self.presence_head(slots).squeeze(-1),
            "centers": torch.sigmoid(self.center_head(slots)),
            "kind_logits": self.kind_head(slots),
            "relation_logits": self.relation_head(
                torch.cat((first, second, first - second), dim=-1)
            ),
            "macro_logits": self.macro_head(global_features),
        }


class LearnedVisualTransitionRequestBuilder(
    LearnedFeatureTransitionRequestBuilder
):
    def __init__(
        self,
        parsed_frames: Mapping[str, ParsedFrameSymbols],
        checkpoint_identity: str,
        source_identity: str,
    ) -> None:
        super().__init__(parsed_frames, checkpoint_identity, source_identity)
        self.identity = identity((
            "cohort-v2-frozen-encoder-visual-symbol-input-v1",
            checkpoint_identity,
            source_identity,
        ))


def _image_tensor(png: bytes, config: CohortV2VisualParserConfig) -> torch.Tensor:
    try:
        with Image.open(io.BytesIO(png)) as opened:
            image = opened.convert("RGB").resize(
                (config.image_width, config.image_height), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError) as error:
        raise CohortV2VisualParserError("aligned agent observation is not RGB") from error
    return torch.from_numpy(array).permute(2, 0, 1)


def _kind(scenario_object_id: str) -> str:
    prefix = scenario_object_id.split(":", 1)[0]
    return prefix if prefix in ENTITY_KINDS[:-1] else "other"


def visual_object_vocabulary(reader: CohortV2AlignedObservationReader) -> tuple[str, ...]:
    if {item.exposure_role for item in reader.rollouts} != {"training"}:
        raise CohortV2VisualParserError("object vocabulary requires the training role")
    values = {
        entity["scenario_object_id"]
        for rollout in reader.rollouts
        for frame in rollout.frame_records
        for entity in frame.engine_state["entities"]
    }
    if any(not isinstance(value, str) or not value for value in values):
        raise CohortV2VisualParserError("scenario object vocabulary is malformed")
    return tuple(sorted(values))


def _project_center(entity: Mapping[str, Any], metadata: Mapping[str, Any]) -> tuple[float, float]:
    body = entity.get("body")
    if not isinstance(body, Mapping) or not isinstance(body.get("position"), tuple):
        raise CohortV2VisualParserError("active visual target lacks body position")
    x, y = (float(value) for value in body["position"])
    transform = metadata["world_to_observation_transform"]
    world = np.asarray(transform["world_to_camera_matrix"], dtype=np.float64).reshape(4, 4)
    projection = np.asarray(transform["camera_to_clip_matrix"], dtype=np.float64).reshape(4, 4)
    clip = projection @ world @ np.asarray((x, y, 0.0, 1.0))
    if abs(float(clip[3])) < 1e-9:
        raise CohortV2VisualParserError("visual target projection is singular")
    ndc = clip[:3] / clip[3]
    pixel = np.asarray(transform["ndc_to_observation_matrix"], dtype=np.float64).reshape(3, 3) @ np.asarray((ndc[0], ndc[1], 1.0))
    viewport = metadata["viewport"]
    return (
        float(np.clip(pixel[0] / viewport["width_pixels"], 0.0, 1.0)),
        float(np.clip(pixel[1] / viewport["height_pixels"], 0.0, 1.0)),
    )


def _available(frame: CohortV2CentralFrameRecord, predicate: str) -> Mapping[str, Any] | None:
    label = frame.labels[predicate]
    availability = label.get("availability")
    if availability == "available":
        return label
    if isinstance(availability, str) and availability.startswith("unavailable_"):
        return None
    raise CohortV2VisualParserError(f"{predicate} availability is malformed")


def build_visual_parser_role_data(
    reader: CohortV2AlignedObservationReader,
    config: CohortV2VisualParserConfig,
    *,
    expected_role: str,
    object_vocabulary: tuple[str, ...],
    frame_limit: int | None = None,
) -> VisualParserRoleData:
    if {item.exposure_role for item in reader.rollouts} != {expected_role}:
        raise CohortV2VisualParserError("visual role data crosses exposure roles")
    slot = {value: index for index, value in enumerate(object_vocabulary)}
    count = len(slot)
    images = []
    presence_rows = []
    center_rows = []
    relation_rows = []
    relation_masks = []
    macro_rows = []
    macro_masks = []
    frame_ids = []
    available = {predicate: 0 for predicate in PREDICATES}
    unavailable = {predicate: 0 for predicate in PREDICATES}
    for rollout in reader.rollouts:
        for frame in rollout.frame_records:
            if frame_limit is not None and len(images) >= frame_limit:
                break
            metadata = reader.frame_observation_metadata(rollout, frame)
            images.append(_image_tensor(
                reader.load_frame_observation(rollout, frame, observation_role="agent"),
                config,
            ))
            frame_ids.append(frame.identity)
            presence = torch.zeros(count)
            centers = torch.zeros((count, 2))
            entity_ids = {}
            for entity in frame.engine_state["entities"]:
                scenario_id = entity["scenario_object_id"]
                if scenario_id not in slot:
                    raise CohortV2VisualParserError(
                        "non-training role introduced an unseen object slot"
                    )
                if entity.get("lifecycle") != "active" or entity.get("body_present") is not True:
                    continue
                index = slot[scenario_id]
                presence[index] = 1.0
                centers[index] = torch.tensor(_project_center(entity, metadata))
                entity_ids[entity["entity_id"]] = index
            presence_rows.append(presence)
            center_rows.append(centers)
            relations = torch.zeros((count, count, 2))
            relation_mask = torch.zeros((count, count, 2), dtype=torch.bool)
            for predicate_index, predicate in enumerate(RELATION_PREDICATES):
                label = _available(frame, predicate)
                if label is None:
                    unavailable[predicate] += 1
                    continue
                available[predicate] += 1
                active = presence.bool()
                mask = active[:, None] & active[None, :]
                mask.fill_diagonal_(False)
                relation_mask[:, :, predicate_index] = mask
                for first, second in label["relations"]:
                    if first not in entity_ids or second not in entity_ids:
                        continue
                    first_index, second_index = entity_ids[first], entity_ids[second]
                    relations[first_index, second_index, predicate_index] = 1.0
                    if predicate == "contact":
                        relations[second_index, first_index, predicate_index] = 1.0
            relation_rows.append(relations)
            relation_masks.append(relation_mask)
            macro = torch.zeros(2)
            macro_mask = torch.zeros(2, dtype=torch.bool)
            for predicate_index, predicate in enumerate(MACRO_PREDICATES):
                label = _available(frame, predicate)
                if label is None:
                    unavailable[predicate] += 1
                    continue
                available[predicate] += 1
                macro[predicate_index] = float(label["value"])
                macro_mask[predicate_index] = True
            macro_rows.append(macro)
            macro_masks.append(macro_mask)
        if frame_limit is not None and len(images) >= frame_limit:
            break
    if not images:
        raise CohortV2VisualParserError("visual parser role contains no frames")
    return VisualParserRoleData(
        expected_role,
        tuple(sorted(item.attempt_id for item in reader.rollouts)),
        tuple(sorted({item.scenario_lineage_identity for item in reader.rollouts})),
        tuple(frame_ids),
        torch.stack(images),
        torch.stack(presence_rows),
        torch.stack(center_rows),
        torch.stack(relation_rows),
        torch.stack(relation_masks),
        torch.stack(macro_rows),
        torch.stack(macro_masks),
        object_vocabulary,
        available,
        unavailable,
    )


def build_visual_parser_model(
    config: CohortV2VisualParserConfig,
    training_data: VisualParserRoleData,
) -> CohortV2VisualPredicateParser:
    if training_data.exposure_role != "training":
        raise CohortV2VisualParserError("visual parameters require training role")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    return CohortV2VisualPredicateParser(
        config, training_data.object_vocabulary
    ).to(torch.device(config.device))


def _positive_weight(labels: torch.Tensor, mask: torch.Tensor) -> float:
    selected = labels[mask]
    positives = float(selected.sum())
    negatives = float(selected.numel() - positives)
    return negatives / positives if positives else 1.0


def train_visual_parser(
    model: CohortV2VisualPredicateParser,
    data: VisualParserRoleData,
    *,
    progress: Callable[[str], None] | None = None,
) -> None:
    if data.exposure_role != "training":
        raise CohortV2VisualParserError("visual parameters require training role")
    config = model.config
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(config.seed)
    relation_weights = torch.tensor([
        _positive_weight(data.relation_labels[:, :, :, index], data.relation_mask[:, :, :, index])
        for index in range(2)
    ], device=device)
    macro_weights = torch.tensor([
        _positive_weight(data.macro_labels[:, index], data.macro_mask[:, index])
        for index in range(2)
    ], device=device)
    kind_targets = torch.tensor(
        [ENTITY_KINDS.index(_kind(value)) for value in data.object_vocabulary],
        dtype=torch.long,
        device=device,
    )
    for epoch in range(1, config.epochs + 1):
        order = torch.randperm(len(data.images), generator=generator)
        losses = []
        model.train()
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            presence = data.presence[indices].to(device)
            outputs = model(data.images[indices].to(device))
            loss = F.binary_cross_entropy_with_logits(
                outputs["presence_logits"], presence
            )
            present = presence.bool()
            if present.any():
                loss = loss + F.smooth_l1_loss(
                    outputs["centers"][present],
                    data.centers[indices].to(device)[present],
                )
                expanded_kinds = kind_targets.unsqueeze(0).expand(len(indices), -1)
                loss = loss + F.cross_entropy(
                    outputs["kind_logits"][present], expanded_kinds[present]
                )
            relation_mask = data.relation_mask[indices].to(device)
            relation_labels = data.relation_labels[indices].to(device)
            for predicate_index in range(2):
                mask = relation_mask[:, :, :, predicate_index]
                if mask.any():
                    loss = loss + F.binary_cross_entropy_with_logits(
                        outputs["relation_logits"][:, :, :, predicate_index][mask],
                        relation_labels[:, :, :, predicate_index][mask],
                        pos_weight=relation_weights[predicate_index],
                    )
            macro_mask = data.macro_mask[indices].to(device)
            macro_labels = data.macro_labels[indices].to(device)
            for predicate_index in range(2):
                mask = macro_mask[:, predicate_index]
                if mask.any():
                    loss = loss + F.binary_cross_entropy_with_logits(
                        outputs["macro_logits"][:, predicate_index][mask],
                        macro_labels[:, predicate_index][mask],
                        pos_weight=macro_weights[predicate_index],
                    )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if progress is not None and (epoch == 1 or epoch == config.epochs or epoch % 5 == 0):
            progress(
                f"[visual train hidden={config.hidden_dim} epoch {epoch}/{config.epochs}] "
                f"mean_loss={sum(losses) / len(losses):.6f}"
            )
    model.eval()
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise CohortV2VisualParserError("visual encoder was not frozen")


def _raw_outputs(
    model: CohortV2VisualPredicateParser,
    data: VisualParserRoleData,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    device = next(model.parameters()).device
    chunks: dict[str, list[np.ndarray]] = {target: [] for target in CALIBRATED_TARGETS}
    labels: dict[str, list[np.ndarray]] = {target: [] for target in CALIBRATED_TARGETS}
    with torch.no_grad():
        for start in range(0, len(data.images), model.config.batch_size):
            end = start + model.config.batch_size
            output = model(data.images[start:end].to(device))
            chunks["object_presence"].append(output["presence_logits"].cpu().numpy().reshape(-1))
            labels["object_presence"].append(data.presence[start:end].numpy().reshape(-1))
            for index, predicate in enumerate(RELATION_PREDICATES):
                mask = data.relation_mask[start:end, :, :, index]
                chunks[predicate].append(
                    output["relation_logits"][:, :, :, index].cpu()[mask].numpy()
                )
                labels[predicate].append(
                    data.relation_labels[start:end, :, :, index][mask].numpy()
                )
            for index, predicate in enumerate(MACRO_PREDICATES):
                mask = data.macro_mask[start:end, index]
                chunks[predicate].append(output["macro_logits"][:, index].cpu()[mask].numpy())
                labels[predicate].append(data.macro_labels[start:end, index][mask].numpy())
    return {
        target: (np.concatenate(chunks[target]), np.concatenate(labels[target]))
        for target in CALIBRATED_TARGETS
    }


def _binary_counts(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = probabilities >= threshold
    truth = labels.astype(bool)
    tp = int(np.logical_and(predicted, truth).sum())
    fp = int(np.logical_and(predicted, ~truth).sum())
    fn = int(np.logical_and(~predicted, truth).sum())
    tn = int(np.logical_and(~predicted, ~truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "true_negative": tn, "precision": precision, "recall": recall,
        "f1": f1, "agreement": (tp + tn) / len(labels),
    }


def calibrate_visual_parser(
    model: CohortV2VisualPredicateParser,
    data: VisualParserRoleData,
) -> tuple[dict[str, float], dict[str, float], float]:
    if data.exposure_role != "calibration":
        raise CohortV2VisualParserError("visual calibration requires calibration role")
    raw = _raw_outputs(model, data)
    candidates = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    temperatures = {}
    thresholds = {}
    for target, (logits, labels) in raw.items():
        def nll(temperature: float) -> float:
            values = np.clip(1 / (1 + np.exp(-logits / temperature)), 1e-7, 1 - 1e-7)
            return float(-(labels * np.log(values) + (1 - labels) * np.log(1 - values)).mean())
        temperature = min(candidates, key=lambda value: (nll(value), abs(value - 1)))
        probabilities = 1 / (1 + np.exp(-logits / temperature))
        threshold = max(
            (index / 20 for index in range(1, 20)),
            key=lambda value: (
                _binary_counts(probabilities, labels, value)["f1"],
                -abs(value - 0.5),
            ),
        )
        temperatures[target] = temperature
        thresholds[target] = threshold
    kind_temperature = _calibrate_kind_temperature(model, data, candidates)
    return temperatures, thresholds, kind_temperature


def _calibrate_kind_temperature(
    model: CohortV2VisualPredicateParser,
    data: VisualParserRoleData,
    candidates: Sequence[float],
) -> float:
    device = next(model.parameters()).device
    logits = []
    labels = []
    targets = torch.tensor([ENTITY_KINDS.index(_kind(value)) for value in data.object_vocabulary])
    with torch.no_grad():
        for start in range(0, len(data.images), model.config.batch_size):
            end = start + model.config.batch_size
            output = model(data.images[start:end].to(device))["kind_logits"].cpu()
            present = data.presence[start:end].bool()
            logits.append(output[present])
            labels.append(targets.unsqueeze(0).expand(len(output), -1)[present])
    values = torch.cat(logits)
    truth = torch.cat(labels)
    return min(candidates, key=lambda value: (float(F.cross_entropy(values / value, truth)), abs(value - 1)))


def visual_parser_metrics(
    model: CohortV2VisualPredicateParser,
    data: VisualParserRoleData,
    temperatures: Mapping[str, float] | None = None,
    thresholds: Mapping[str, float] | None = None,
    object_kind_temperature: float = 1.0,
) -> dict[str, Any]:
    raw = _raw_outputs(model, data)
    result = {}
    for target, (logits, labels) in raw.items():
        temperature = 1.0 if temperatures is None else float(temperatures[target])
        threshold = 0.5 if thresholds is None else float(thresholds[target])
        probabilities = np.clip(1 / (1 + np.exp(-logits / temperature)), 1e-7, 1 - 1e-7)
        bins = np.minimum((probabilities * 10).astype(int), 9)
        ece = sum(
            float((bins == index).mean())
            * abs(float(probabilities[bins == index].mean()) - float(labels[bins == index].mean()))
            for index in range(10) if (bins == index).any()
        )
        result[target] = {
            "sample_count": int(len(labels)),
            "positive_count": int(labels.sum()),
            "brier_score": float(np.square(probabilities - labels).mean()),
            "negative_log_likelihood": float(
                -(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities)).mean()
            ),
            "expected_calibration_error_10_bin": ece,
            "temperature": temperature,
            "threshold": threshold,
            **_binary_counts(probabilities, labels, threshold),
        }
        if target in PREDICATES:
            result[target]["available_frame_count"] = data.available_frame_counts[target]
            result[target]["unavailable_frame_count"] = data.unavailable_frame_counts[target]
    result.update(_object_metrics(model, data, object_kind_temperature))
    result["mean_selection_negative_log_likelihood"] = float(np.mean([
        result[target]["negative_log_likelihood"] for target in CALIBRATED_TARGETS
    ] + [result["object_kind"]["negative_log_likelihood"]]))
    result["directed_predicate_agreement"] = result["supports"]["agreement"]
    result["macro_event_agreement"] = float(np.mean([
        result[target]["agreement"] for target in MACRO_PREDICATES
    ]))
    return result


def _object_metrics(
    model: CohortV2VisualPredicateParser,
    data: VisualParserRoleData,
    temperature: float,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    center_errors = []
    correct = count = 0
    nll_sum = 0.0
    targets = torch.tensor([ENTITY_KINDS.index(_kind(value)) for value in data.object_vocabulary])
    with torch.no_grad():
        for start in range(0, len(data.images), model.config.batch_size):
            end = start + model.config.batch_size
            output = model(data.images[start:end].to(device))
            present = data.presence[start:end].bool()
            center_errors.extend(
                torch.linalg.vector_norm(
                    output["centers"].cpu()[present] - data.centers[start:end][present], dim=1
                ).tolist()
            )
            truth = targets.unsqueeze(0).expand(len(output["kind_logits"]), -1)[present]
            kind_logits = output["kind_logits"].cpu()[present] / temperature
            correct += int((kind_logits.argmax(dim=1) == truth).sum())
            count += len(truth)
            nll_sum += float(F.cross_entropy(kind_logits, truth, reduction="sum"))
    return {
        "object_alignment": {
            "matched_object_count": len(center_errors),
            "mean_normalized_center_error": float(np.mean(center_errors)),
            "within_0_05_normalized_distance_rate": float(np.mean(np.asarray(center_errors) <= 0.05)),
        },
        "object_kind": {
            "sample_count": count,
            "accuracy": correct / count,
            "negative_log_likelihood": nll_sum / count,
            "temperature": temperature,
        },
    }


def select_visual_parser(
    candidates: Sequence[CohortV2VisualPredicateParser],
    data: VisualParserRoleData,
) -> tuple[CohortV2VisualPredicateParser, list[dict[str, Any]]]:
    if data.exposure_role != "model_selection":
        raise CohortV2VisualParserError("visual selection requires model-selection role")
    rows = []
    for model in candidates:
        metrics = visual_parser_metrics(model, data)
        rows.append({
            "configuration_identity": model.config.identity,
            "hidden_dim": model.config.hidden_dim,
            "mean_selection_negative_log_likelihood": metrics[
                "mean_selection_negative_log_likelihood"
            ],
            "metrics": metrics,
        })
    selected = min(
        range(len(rows)),
        key=lambda index: (rows[index]["mean_selection_negative_log_likelihood"], rows[index]["hidden_dim"]),
    )
    for index, row in enumerate(rows):
        row["selected"] = index == selected
    return candidates[selected], rows


def parse_visual_frame_symbols(
    model: CohortV2VisualPredicateParser,
    image: torch.Tensor,
    frame: CohortV2CentralFrameRecord,
    temperatures: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> ParsedFrameSymbols:
    device = next(model.parameters()).device
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
    presence = torch.sigmoid(
        output["presence_logits"][0] / temperatures["object_presence"]
    ).cpu() >= thresholds["object_presence"]
    relations = {}
    for predicate_index, predicate in enumerate(RELATION_PREDICATES):
        availability = str(frame.labels[predicate].get("availability"))
        if availability != "available":
            relations[predicate] = RelationTransitionValue(availability, None)
            continue
        probabilities = torch.sigmoid(
            output["relation_logits"][0, :, :, predicate_index]
            / temperatures[predicate]
        ).cpu()
        values = []
        for first in range(len(model.object_vocabulary)):
            for second in range(len(model.object_vocabulary)):
                if first == second or not presence[first] or not presence[second]:
                    continue
                if float(probabilities[first, second]) < thresholds[predicate]:
                    continue
                pair = (
                    "runtime:" + model.object_vocabulary[first],
                    "runtime:" + model.object_vocabulary[second],
                )
                if predicate == "contact" and first > second:
                    continue
                values.append(pair)
        relations[predicate] = RelationTransitionValue("available", tuple(values))
    macros = {}
    for predicate_index, predicate in enumerate(MACRO_PREDICATES):
        availability = str(frame.labels[predicate].get("availability"))
        value = bool(float(torch.sigmoid(
            output["macro_logits"][0, predicate_index] / temperatures[predicate]
        ).cpu()) >= thresholds[predicate]) if availability == "available" else None
        macros[predicate] = BooleanTransitionValue(availability, value)
    return ParsedFrameSymbols(
        frame.identity,
        relations["contact"],
        relations["supports"],
        macros["steady-state"],
        macros["structure-unstable"],
    )


def parse_visual_reader_frames(
    model: CohortV2VisualPredicateParser,
    reader: CohortV2AlignedObservationReader,
    temperatures: Mapping[str, float],
    thresholds: Mapping[str, float],
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, ParsedFrameSymbols]:
    total = sum(len(item.frame_records) for item in reader.rollouts)
    parsed = {}
    current = 0
    for rollout in reader.rollouts:
        for frame in rollout.frame_records:
            current += 1
            image = _image_tensor(
                reader.load_frame_observation(rollout, frame, observation_role="agent"),
                model.config,
            )
            parsed[frame.identity] = parse_visual_frame_symbols(
                model, image, frame, temperatures, thresholds
            )
            if progress is not None and (current == total or current % 250 == 0):
                progress(f"[visual infer {current}/{total}] frame={frame.identity}")
    return parsed


def save_visual_parser_checkpoint(
    root: Path,
    model: CohortV2VisualPredicateParser,
    temperatures: Mapping[str, float],
    thresholds: Mapping[str, float],
    object_kind_temperature: float,
    *,
    role_data: Sequence[VisualParserRoleData],
    readers: Sequence[CohortV2AlignedObservationReader],
    model_selection: Sequence[Mapping[str, Any]],
    calibration_metrics: Mapping[str, Any],
    implementation_revision: str,
) -> CohortV2VisualParserCheckpoint:
    target = Path(root)
    if target.exists():
        raise CohortV2VisualParserError("visual checkpoint destination exists")
    target.mkdir(parents=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    state_identity = cohort_v2_model_state_identity(state)
    payload = {
        "schema": SCHEMA,
        "config": asdict(model.config),
        "object_vocabulary": list(model.object_vocabulary),
        "temperatures": dict(temperatures),
        "thresholds": dict(thresholds),
        "object_kind_temperature": object_kind_temperature,
        "model_state": state,
        "model_state_identity": state_identity,
    }
    torch.save(payload, target / "checkpoint.pt")
    source_bindings = {
        "release_identity": readers[0].release_identity,
        "partition_identity": readers[0].partition_identity,
        "encoder_identity": model.encoder.identity,
        "encoder_frozen": True,
        "learned_parameter_role": "training",
        "configuration_selection_role": "model_selection",
        "probability_calibration_role": "calibration",
        "roles": {
            data.exposure_role: {
                "attempt_ids": list(data.attempt_ids),
                "scenario_lineage_identities": list(data.scenario_lineage_identities),
            }
            for data in role_data
        },
    }
    checkpoint_identity = identity((
        "cohort-v2-visual-parser-checkpoint-v1",
        model.config.identity,
        state_identity,
        tuple(model.object_vocabulary),
        tuple(sorted(temperatures.items())),
        tuple(sorted(thresholds.items())),
        object_kind_temperature,
        implementation_revision,
    ))
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "identity": checkpoint_identity,
        "implementation_revision": implementation_revision,
        "checkpoint_path": "checkpoint.pt",
        "model_state_identity": state_identity,
        "config_identity": model.config.identity,
        "object_vocabulary": list(model.object_vocabulary),
        "temperatures": dict(temperatures),
        "thresholds": dict(thresholds),
        "object_kind_temperature": object_kind_temperature,
        "source_bindings": source_bindings,
        "model_selection": list(model_selection),
        "calibration_metrics": dict(calibration_metrics),
    }
    (target / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return CohortV2VisualParserCheckpoint(
        target / "checkpoint.pt", checkpoint_identity, state_identity,
        model.config, model.object_vocabulary, dict(temperatures), dict(thresholds),
        object_kind_temperature, source_bindings,
    )


def load_visual_parser_checkpoint(
    root: Path,
    *,
    readers: Sequence[CohortV2AlignedObservationReader],
    device: str = "cpu",
) -> tuple[CohortV2VisualPredicateParser, CohortV2VisualParserCheckpoint, dict[str, Any]]:
    target = Path(root)
    raw = (target / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if canonical_json_bytes(manifest) != raw or manifest.get("schema") != MANIFEST_SCHEMA:
        raise CohortV2VisualParserError("visual parser manifest is malformed")
    try:
        payload = torch.load(target / manifest["checkpoint_path"], map_location="cpu", weights_only=True)
        config = CohortV2VisualParserConfig(**payload["config"])
        vocabulary = tuple(payload["object_vocabulary"])
        model = CohortV2VisualPredicateParser(config, vocabulary)
        model.load_state_dict(payload["model_state"], strict=True)
    except (OSError, KeyError, TypeError, RuntimeError, ValueError) as error:
        raise CohortV2VisualParserError(f"visual checkpoint is invalid: {error}") from error
    state_identity = cohort_v2_model_state_identity({
        key: value.detach().cpu() for key, value in model.state_dict().items()
    })
    roles = tuple(data.rollouts[0].exposure_role for data in readers)
    bindings = manifest["source_bindings"]
    expected_identity = identity((
        "cohort-v2-visual-parser-checkpoint-v1",
        config.identity,
        state_identity,
        vocabulary,
        tuple(sorted(payload["temperatures"].items())),
        tuple(sorted(payload["thresholds"].items())),
        float(payload["object_kind_temperature"]),
        manifest["implementation_revision"],
    ))
    if (
        roles != ("training", "calibration", "model_selection")
        or state_identity != payload["model_state_identity"]
        or state_identity != manifest["model_state_identity"]
        or manifest["identity"] != expected_identity
        or bindings["release_identity"] != readers[0].release_identity
        or bindings["encoder_identity"] != model.encoder.identity
        or bindings["encoder_frozen"] is not True
        or set(bindings["roles"]) != set(roles)
        or any(
            bindings["roles"][role]["attempt_ids"]
            != list(sorted(item.attempt_id for item in reader.rollouts))
            for role, reader in zip(roles, readers, strict=True)
        )
    ):
        raise CohortV2VisualParserError("visual checkpoint source binding differs")
    checkpoint = CohortV2VisualParserCheckpoint(
        target / "checkpoint.pt", manifest["identity"], state_identity, config,
        vocabulary, dict(payload["temperatures"]), dict(payload["thresholds"]),
        float(payload["object_kind_temperature"]), bindings,
    )
    return model.to(torch.device(device)).eval(), checkpoint, manifest


__all__ = [
    "CALIBRATED_TARGETS", "CohortV2VisualParserCheckpoint",
    "CohortV2VisualParserConfig", "CohortV2VisualParserError",
    "CohortV2VisualPredicateParser", "FrozenVisualEncoder",
    "VisualParserRoleData", "build_visual_parser_model",
    "build_visual_parser_role_data", "calibrate_visual_parser",
    "load_visual_parser_checkpoint", "parse_visual_frame_symbols",
    "parse_visual_reader_frames", "save_visual_parser_checkpoint",
    "select_visual_parser", "train_visual_parser", "visual_object_vocabulary",
    "visual_parser_metrics",
    "LearnedVisualTransitionRequestBuilder",
]
