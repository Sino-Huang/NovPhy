"""Model-relative micro-constraint reliability derivation and ablations."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

from world_model.data import (
    CohortV2OracleWindow,
    CohortV2OracleWindowDataset,
    CohortV2ReleaseReader,
)
from world_model.model import (
    Abstraction,
    DualOutputPredictor,
    MicroTransitionBatch,
    MicroTransitionInput,
    PredictionPair,
    RelationTransitionValue,
    TransitionRequest,
    identity,
)
from world_model.training.cohort_v2 import build_cohort_v2_transition_request
from world_model.training.cohort_v2_controller import (
    CohortV2ControllerConfig,
    CohortV2ControllerFeatureCodec,
)
from world_model.training.cohort_v2_micro import (
    MICRO_PREDICATES,
    CohortV2MicroConfig,
    CohortV2MicroTrainer,
    CohortV2StateCodec,
    cohort_v2_action,
    cohort_v2_model_state_identity,
)
from world_model.training.grid_artifacts import canonical_json_bytes


RELIABILITY_SCHEMA = "cohort_v2_micro_reliability_v1"
RELIABILITY_CHECKPOINT_SCHEMA = "cohort_v2_micro_reliability_checkpoints_v1"


class CohortV2ReliabilityError(ValueError):
    """The reliability derivation, estimator, or ablation is invalid."""


@dataclass(frozen=True, slots=True)
class CohortV2ReliabilityConfig:
    preliminary_steps: int = 1200
    final_steps: int = 1200
    estimator_epochs: int = 80
    controller_epochs: int = 80
    batch_size: int = 32
    evaluation_batch_size: int = 128
    hidden_dim: int = 32
    learning_rate: float = 3e-3
    seed: int = 20260826
    device: str = "cuda"
    requested_horizon: int = 1

    def __post_init__(self) -> None:
        for field in (
            "preliminary_steps",
            "final_steps",
            "estimator_epochs",
            "controller_epochs",
            "batch_size",
            "evaluation_batch_size",
            "hidden_dim",
            "requested_horizon",
        ):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise CohortV2ReliabilityError(f"{field} must be a positive integer")
        if (
            type(self.seed) is not int
            or self.seed < 0
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
            or type(self.device) is not str
            or not self.device
        ):
            raise CohortV2ReliabilityError("reliability optimizer configuration is invalid")

    @property
    def micro_config(self) -> CohortV2MicroConfig:
        return CohortV2MicroConfig(
            seed=self.seed,
            steps=self.preliminary_steps,
            batch_size=self.batch_size,
            device=self.device,
        )

    @property
    def final_micro_config(self) -> CohortV2MicroConfig:
        return CohortV2MicroConfig(
            seed=self.seed + 1,
            steps=self.final_steps,
            batch_size=self.batch_size,
            device=self.device,
        )

    @property
    def feature_config(self) -> CohortV2ControllerConfig:
        return CohortV2ControllerConfig(hidden_dim=self.hidden_dim, seed=self.seed)

    @property
    def identity(self) -> str:
        return identity(("cohort-v2-micro-reliability-config-v1", *asdict(self).values()))


@dataclass(frozen=True, slots=True)
class CohortV2ReliabilityLabel:
    state_id: str
    exposure_role: str
    attempt_id: str
    scenario_lineage_identity: str
    context_position: int
    context_fixed_step: int
    without_micro_objective: float
    with_micro_objective: float
    usefulness_margin: float
    useful: bool
    features: torch.Tensor


@dataclass(frozen=True, slots=True)
class CohortV2ReliabilityDerivation:
    labels: tuple[CohortV2ReliabilityLabel, ...]
    excluded_unavailable_count: int
    preliminary_checkpoint_identity: str
    target_identity: str


class CohortV2ReliabilityEstimator(nn.Module):
    """Predict micro usefulness from deployment-observable inputs only."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


class CohortV2ReliabilityModeController(nn.Module):
    """Select continuous versus micro at the declared short horizon."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


def split_reliability_training_attempts(
    reader: CohortV2ReleaseReader,
) -> tuple[frozenset[str], frozenset[str]]:
    """Return a deterministic train/label split over complete training rollouts."""
    attempts = tuple(sorted(rollout.attempt_id for rollout in reader.rollouts))
    if len(attempts) < 2 or {rollout.exposure_role for rollout in reader.rollouts} != {
        "training"
    }:
        raise CohortV2ReliabilityError(
            "reliability splitting requires at least two training rollouts"
        )
    midpoint = len(attempts) // 2
    preliminary = frozenset(attempts[:midpoint])
    label = frozenset(attempts[midpoint:])
    return preliminary, label


def preliminary_checkpoint_identity(
    trainer: CohortV2MicroTrainer,
    included_attempt_ids: frozenset[str],
) -> str:
    return identity((
        "cohort-v2-micro-reliability-preliminary-checkpoint-v1",
        trainer.data.reader.release_identity,
        trainer.data.reader.partition_identity,
        trainer.config.identity,
        tuple(sorted(included_attempt_ids)),
        cohort_v2_model_state_identity(trainer.predictor.state_dict()),
        trainer.step_count,
    ))


def _micro_available(window: CohortV2OracleWindow) -> bool:
    return all(
        window.context.labels[predicate].get("availability") == "available"
        and window.target.labels[predicate].get("availability") == "available"
        for predicate in MICRO_PREDICATES
    )


def _counterfactual_batch(
    predictor: DualOutputPredictor,
    codec: CohortV2StateCodec,
    windows: tuple[CohortV2OracleWindow, ...],
    frame_counts: Mapping[str, int],
) -> tuple[tuple[float, float], ...]:
    device = next(predictor.parameters()).device
    with torch.no_grad():
        context = codec.batch(tuple(window.context for window in windows)).to(device)
        target = codec.batch(tuple(window.target for window in windows)).to(device)
        action = torch.stack(tuple(cohort_v2_action(window) for window in windows)).to(
            device
        )
        pair = PredictionPair(windows[0].requested_horizon, Abstraction.MICRO)
        without_micro = TransitionRequest(
            pair,
            MicroTransitionBatch(tuple(
                MicroTransitionInput(
                    frame_record_identity=window.context.identity,
                    contact=RelationTransitionValue(
                        "unavailable_counterfactual_ablation", None
                    ),
                    supports=RelationTransitionValue(
                        "unavailable_counterfactual_ablation", None
                    ),
                )
                for window in windows
            )),
        )
        objectives = []
        for request in (
            without_micro,
            build_cohort_v2_transition_request(pair, windows),
        ):
            carrier = predictor.carrier(context, action, request)
            duration = torch.tensor(
                [
                    window.effective_horizon
                    / (frame_counts[window.attempt_id] - 1)
                    for window in windows
                ],
                dtype=carrier.dtype,
                device=device,
            )
            objectives.append(duration * (carrier - target).pow(2).mean(dim=1))
    without_micro, with_micro = (
        value.detach().cpu().tolist() for value in objectives
    )
    return tuple(
        (float(left), float(right))
        for left, right in zip(without_micro, with_micro, strict=True)
    )


def derive_cohort_v2_reliability_labels(
    predictor: DualOutputPredictor,
    codec: CohortV2StateCodec,
    readers: tuple[CohortV2ReleaseReader, ...],
    held_out_training_attempt_ids: frozenset[str],
    config: CohortV2ReliabilityConfig,
    checkpoint_identity: str,
    *,
    max_examples_per_role: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> CohortV2ReliabilityDerivation:
    if tuple(reader.rollouts[0].exposure_role for reader in readers) != (
        "training",
        "calibration",
        "model_selection",
    ):
        raise CohortV2ReliabilityError("reliability readers must preserve role order")
    if not held_out_training_attempt_ids:
        raise CohortV2ReliabilityError("held-out training trajectories are empty")
    feature_codec = CohortV2ControllerFeatureCodec(config.feature_config)
    frame_counts = {
        rollout.attempt_id: len(rollout.frame_records)
        for reader in readers
        for rollout in reader.rollouts
    }
    selected: list[CohortV2OracleWindow] = []
    excluded = 0
    for reader in (readers[0], readers[2]):
        role_count = 0
        for window in CohortV2OracleWindowDataset(
            reader, requested_horizons=(config.requested_horizon,)
        ):
            if (
                window.exposure_role == "training"
                and window.attempt_id not in held_out_training_attempt_ids
            ):
                continue
            if not _micro_available(window):
                excluded += 1
                continue
            if max_examples_per_role is not None and role_count >= max_examples_per_role:
                continue
            selected.append(window)
            role_count += 1
    if not selected or not any(window.exposure_role == "training" for window in selected):
        raise CohortV2ReliabilityError("no available out-of-sample training labels")

    labels = []
    predictor.eval()
    for start in range(0, len(selected), config.evaluation_batch_size):
        windows = tuple(selected[start : start + config.evaluation_batch_size])
        values = _counterfactual_batch(predictor, codec, windows, frame_counts)
        for window, (without_micro, with_micro) in zip(windows, values, strict=True):
            margin = without_micro - with_micro
            labels.append(CohortV2ReliabilityLabel(
                state_id=identity((
                    "cohort-v2-micro-usefulness-state-v1",
                    window.context.identity,
                    window.requested_horizon,
                )),
                exposure_role=window.exposure_role,
                attempt_id=window.attempt_id,
                scenario_lineage_identity=window.scenario_lineage_identity,
                context_position=window.context_position,
                context_fixed_step=window.context.fixed_step,
                without_micro_objective=without_micro,
                with_micro_objective=with_micro,
                usefulness_margin=margin,
                useful=margin > 0.0,
                features=feature_codec.encode(
                    window.agent_observation,
                    elapsed_fixed_steps=(
                        window.context.fixed_step
                        - next(
                            rollout.frame_records[0].fixed_step
                            for reader in readers
                            for rollout in reader.rollouts
                            if rollout.attempt_id == window.attempt_id
                        )
                    ),
                    intervention=window.intervention,
                ),
            ))
        if progress is not None:
            progress(f"[labels] scored={min(start + len(windows), len(selected))}/{len(selected)}")
    target_identity = identity((
        "cohort-v2-micro-relation-usefulness-derivation-v1",
        readers[0].release_identity,
        readers[0].partition_identity,
        checkpoint_identity,
        config.requested_horizon,
        "duration-weighted-carrier-mse:micro-mode-without-minus-with-oracle-relations",
        tuple(
            (
                label.state_id,
                label.exposure_role,
                label.without_micro_objective,
                label.with_micro_objective,
                label.useful,
            )
            for label in labels
        ),
    ))
    return CohortV2ReliabilityDerivation(
        tuple(labels), excluded, checkpoint_identity, target_identity
    )


def _train_binary(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    progress_prefix: str,
    progress: Callable[[str], None] | None,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(features), generator=generator)
        total = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = functional.binary_cross_entropy_with_logits(
                model(features[indices]), targets[indices]
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        if progress is not None and (
            epoch == 0 or epoch + 1 == epochs or (epoch + 1) % 10 == 0
        ):
            progress(
                f"[train:{progress_prefix}] epoch={epoch + 1}/{epochs} "
                f"loss={total / len(features):.6f}"
            )
    model.eval()


def train_cohort_v2_reliability_models(
    derivation: CohortV2ReliabilityDerivation,
    config: CohortV2ReliabilityConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[
    CohortV2ReliabilityEstimator,
    CohortV2ReliabilityModeController,
    CohortV2ReliabilityModeController,
]:
    training = tuple(
        label for label in derivation.labels if label.exposure_role == "training"
    )
    if not training:
        raise CohortV2ReliabilityError("estimator training labels are empty")
    features = torch.stack(tuple(label.features for label in training))
    targets = torch.tensor(tuple(float(label.useful) for label in training))
    torch.manual_seed(config.seed)
    estimator = CohortV2ReliabilityEstimator(features.shape[1], config.hidden_dim)
    _train_binary(
        estimator,
        features,
        targets,
        epochs=config.estimator_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=config.seed,
        progress_prefix="reliability",
        progress=progress,
    )
    with torch.no_grad():
        probabilities = torch.sigmoid(estimator(features)).unsqueeze(1)
    control_feature = torch.zeros((len(features), 1), dtype=features.dtype)
    torch.manual_seed(config.seed + 1)
    raw_controller = CohortV2ReliabilityModeController(features.shape[1] + 1)
    _train_binary(
        raw_controller,
        torch.cat((features, control_feature), dim=1),
        targets,
        epochs=config.controller_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=config.seed + 1,
        progress_prefix="controller_raw",
        progress=progress,
    )
    torch.manual_seed(config.seed + 1)
    feature_controller = CohortV2ReliabilityModeController(features.shape[1] + 1)
    _train_binary(
        feature_controller,
        torch.cat((features, probabilities), dim=1),
        targets,
        epochs=config.controller_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=config.seed + 1,
        progress_prefix="controller_with_reliability",
        progress=progress,
    )
    return estimator, raw_controller, feature_controller


def evaluate_cohort_v2_reliability_models(
    derivation: CohortV2ReliabilityDerivation,
    estimator: CohortV2ReliabilityEstimator,
    raw_controller: CohortV2ReliabilityModeController,
    feature_controller: CohortV2ReliabilityModeController,
) -> dict[str, object]:
    training = tuple(
        label for label in derivation.labels if label.exposure_role == "training"
    )
    held_out = tuple(
        label for label in derivation.labels if label.exposure_role == "model_selection"
    )
    if not held_out:
        raise CohortV2ReliabilityError("model-selection reliability labels are empty")
    features = torch.stack(tuple(label.features for label in held_out))
    targets = torch.tensor(tuple(float(label.useful) for label in held_out))
    with torch.no_grad():
        probabilities = torch.sigmoid(estimator(features))
        raw_micro = torch.sigmoid(
            raw_controller(torch.cat((
                features,
                torch.zeros_like(probabilities.unsqueeze(1)),
            ), dim=1))
        ) >= 0.5
        feature_micro = torch.sigmoid(
            feature_controller(torch.cat((features, probabilities.unsqueeze(1)), dim=1))
        ) >= 0.5
    without_micro = torch.tensor(
        tuple(label.without_micro_objective for label in held_out)
    )
    with_micro = torch.tensor(tuple(label.with_micro_objective for label in held_out))

    training_without = sum(
        label.without_micro_objective for label in training
    ) / len(training)
    training_with = sum(label.with_micro_objective for label in training) / len(training)
    fixed_micro = training_with < training_without
    fixed = torch.full_like(raw_micro, fixed_micro)

    def controller_score(selection: torch.Tensor) -> dict[str, float | int]:
        selected = torch.where(selection, with_micro, without_micro)
        return {
            "micro_selection_count": int(selection.sum()),
            "mean_selected_objective": float(selected.mean()),
            "oracle_decision_accuracy": float((selection == targets.bool()).float().mean()),
        }

    estimator_accuracy = float(((probabilities >= 0.5) == targets.bool()).float().mean())
    fixed_score = controller_score(fixed)
    raw_score = controller_score(raw_micro)
    feature_score = controller_score(feature_micro)
    controller_parameter_counts = tuple(
        sum(parameter.numel() for parameter in model.parameters())
        for model in (raw_controller, feature_controller)
    )
    if controller_parameter_counts[0] != controller_parameter_counts[1]:
        raise CohortV2ReliabilityError("controller feature ablation capacity differs")
    return {
        "controller_feature_ablation": {
            "controller_parameter_count_each": controller_parameter_counts[0],
            "fixed_rule": fixed_score,
            "raw_deployment_features": raw_score,
            "raw_plus_reliability_feature": feature_score,
            "incremental_held_out_value": (
                float(raw_score["mean_selected_objective"])
                - float(feature_score["mean_selected_objective"])
            ),
            "loss_gate_fixed": "off",
        },
        "estimator": {
            "accuracy": estimator_accuracy,
            "available_label_count": len(held_out),
            "positive_label_count": int(targets.sum()),
            "input_contract": "agent_observation+declared_intervention+elapsed_fixed_steps",
        },
    }


def reliability_symbolic_gate(
    estimator: CohortV2ReliabilityEstimator,
    config: CohortV2ReliabilityConfig,
    readers: tuple[CohortV2ReleaseReader, ...],
) -> Callable[[tuple[CohortV2OracleWindow, ...]], torch.Tensor]:
    codec = CohortV2ControllerFeatureCodec(config.feature_config)
    first_fixed_steps = {
        rollout.attempt_id: rollout.frame_records[0].fixed_step
        for reader in readers
        for rollout in reader.rollouts
    }
    estimator.eval()

    def gate(windows: tuple[CohortV2OracleWindow, ...]) -> torch.Tensor:
        features = torch.stack(tuple(
            codec.encode(
                window.agent_observation,
                elapsed_fixed_steps=(
                    window.context.fixed_step - first_fixed_steps[window.attempt_id]
                ),
                intervention=window.intervention,
            )
            for window in windows
        ))
        with torch.no_grad():
            return torch.sigmoid(estimator(features)).clamp_min(1e-6)

    return gate


def score_micro_carrier_objective(
    trainer: CohortV2MicroTrainer,
    reader: CohortV2ReleaseReader,
    config: CohortV2ReliabilityConfig,
    *,
    max_examples: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> float:
    windows = tuple(
        window
        for window in CohortV2OracleWindowDataset(
            reader, requested_horizons=(config.requested_horizon,)
        )
        if _micro_available(window)
    )
    if max_examples is not None:
        windows = windows[:max_examples]
    frame_counts = {
        rollout.attempt_id: len(rollout.frame_records) for rollout in reader.rollouts
    }
    values = []
    trainer.predictor.eval()
    for start in range(0, len(windows), config.evaluation_batch_size):
        batch = windows[start : start + config.evaluation_batch_size]
        values.extend(
            with_micro
            for _, with_micro in _counterfactual_batch(
                trainer.predictor, trainer.codec, batch, frame_counts
            )
        )
        if progress is not None:
            progress(f"[score:loss_gate] {min(start + len(batch), len(windows))}/{len(windows)}")
    if not values:
        raise CohortV2ReliabilityError("loss-gate evaluation has no available labels")
    return sum(values) / len(values)


def _label_payload(label: CohortV2ReliabilityLabel) -> dict[str, object]:
    return {
        "attempt_id": label.attempt_id,
        "context_fixed_step": label.context_fixed_step,
        "context_position": label.context_position,
        "exposure_role": label.exposure_role,
        "with_micro_objective": label.with_micro_objective,
        "without_micro_objective": label.without_micro_objective,
        "scenario_lineage_identity": label.scenario_lineage_identity,
        "state_id": label.state_id,
        "useful": label.useful,
        "usefulness_margin": label.usefulness_margin,
    }


def _digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_cohort_v2_reliability_artifact(
    root: Path,
    *,
    readers: tuple[CohortV2ReleaseReader, ...],
    config: CohortV2ReliabilityConfig,
    preliminary_attempt_ids: frozenset[str],
    label_attempt_ids: frozenset[str],
    derivation: CohortV2ReliabilityDerivation,
    estimator: CohortV2ReliabilityEstimator,
    raw_controller: CohortV2ReliabilityModeController,
    feature_controller: CohortV2ReliabilityModeController,
    preliminary_trainer: CohortV2MicroTrainer,
    ungated_trainer: CohortV2MicroTrainer,
    gated_trainer: CohortV2MicroTrainer,
    scores: Mapping[str, object],
    implementation_revision: str,
) -> dict[str, object]:
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    labels_bytes = b"".join(
        canonical_json_bytes(_label_payload(label)) for label in derivation.labels
    )
    scores_bytes = canonical_json_bytes(dict(scores))
    states = {
        "estimator": estimator.state_dict(),
        "feature_controller": feature_controller.state_dict(),
        "gated_final": gated_trainer.predictor.state_dict(),
        "preliminary": preliminary_trainer.predictor.state_dict(),
        "raw_controller": raw_controller.state_dict(),
        "ungated_final": ungated_trainer.predictor.state_dict(),
    }
    state_identities = {
        name: cohort_v2_model_state_identity(state) for name, state in states.items()
    }
    checkpoint_payload = {
        "config_identity": config.identity,
        "schema": RELIABILITY_CHECKPOINT_SCHEMA,
        "state_identities": state_identities,
        "states": states,
    }
    checkpoint_path = target / "checkpoint.pt"
    temporary = checkpoint_path.with_name("checkpoint.pt.tmp")
    torch.save(checkpoint_payload, temporary)
    os.replace(temporary, checkpoint_path)
    _atomic_write(target / "labels.jsonl", labels_bytes)
    _atomic_write(target / "scores.json", scores_bytes)
    artifact_identity = identity((
        RELIABILITY_SCHEMA,
        readers[0].release_identity,
        readers[0].partition_identity,
        config.identity,
        derivation.preliminary_checkpoint_identity,
        derivation.target_identity,
        tuple(sorted(state_identities.items())),
        _digest(labels_bytes),
        _digest(scores_bytes),
        implementation_revision,
    ))
    manifest = {
        "artifact_identity": artifact_identity,
        "checkpoint": "checkpoint.pt",
        "checkpoint_identity": derivation.preliminary_checkpoint_identity,
        "config": asdict(config),
        "config_identity": config.identity,
        "controller_feature_ablation_holds_loss_gate": "off",
        "derivation_identity": derivation.target_identity,
        "excluded_unavailable_count": derivation.excluded_unavailable_count,
        "final_evaluation_consumed": False,
        "implementation_revision": implementation_revision,
        "label_attempt_ids": sorted(label_attempt_ids),
        "labels": "labels.jsonl",
        "labels_identity": _digest(labels_bytes),
        "loss_gate_ablation_holds_controller_feature": "absent",
        "model_selection_label_count": sum(
            label.exposure_role == "model_selection" for label in derivation.labels
        ),
        "partition_identity": readers[0].partition_identity,
        "preliminary_attempt_ids": sorted(preliminary_attempt_ids),
        "release_identity": readers[0].release_identity,
        "schema": RELIABILITY_SCHEMA,
        "scores": "scores.json",
        "scores_identity": _digest(scores_bytes),
        "state_identities": state_identities,
        "training_label_count": sum(
            label.exposure_role == "training" for label in derivation.labels
        ),
    }
    _atomic_write(target / "manifest.json", canonical_json_bytes(manifest))
    return manifest


def validate_cohort_v2_reliability_artifact(
    root: Path,
    *,
    readers: tuple[CohortV2ReleaseReader, ...],
    config: CohortV2ReliabilityConfig,
) -> dict[str, object]:
    if tuple(reader.rollouts[0].exposure_role for reader in readers) != (
        "training",
        "calibration",
        "model_selection",
    ):
        raise CohortV2ReliabilityError("reliability readers must preserve role order")
    reader = readers[0]
    target = Path(root)
    try:
        manifest_bytes = (target / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        labels_bytes = (target / manifest["labels"]).read_bytes()
        scores_bytes = (target / manifest["scores"]).read_bytes()
        checkpoint = torch.load(
            target / manifest["checkpoint"], map_location="cpu", weights_only=True
        )
    except (OSError, KeyError, json.JSONDecodeError, RuntimeError, ValueError) as error:
        raise CohortV2ReliabilityError(f"cannot load reliability artifact: {error}") from error
    preliminary, label = split_reliability_training_attempts(reader)
    if (
        manifest_bytes != canonical_json_bytes(manifest)
        or manifest.get("schema") != RELIABILITY_SCHEMA
        or manifest.get("release_identity") != reader.release_identity
        or manifest.get("partition_identity") != reader.partition_identity
        or manifest.get("config_identity") != config.identity
        or manifest.get("config") != asdict(config)
        or manifest.get("preliminary_attempt_ids") != sorted(preliminary)
        or manifest.get("label_attempt_ids") != sorted(label)
        or manifest.get("labels_identity") != _digest(labels_bytes)
        or manifest.get("scores_identity") != _digest(scores_bytes)
        or manifest.get("final_evaluation_consumed") is not False
        or type(checkpoint) is not dict
        or checkpoint.get("schema") != RELIABILITY_CHECKPOINT_SCHEMA
        or checkpoint.get("config_identity") != config.identity
    ):
        raise CohortV2ReliabilityError("reliability artifact provenance is invalid")
    states = checkpoint.get("states")
    expected_state_names = {
        "estimator",
        "feature_controller",
        "gated_final",
        "preliminary",
        "raw_controller",
        "ungated_final",
    }
    if not isinstance(states, Mapping) or set(states) != expected_state_names:
        raise CohortV2ReliabilityError("reliability checkpoint states are malformed")
    identities = {
        name: cohort_v2_model_state_identity(state) for name, state in states.items()
    }
    if identities != checkpoint.get("state_identities") or identities != manifest.get(
        "state_identities"
    ):
        raise CohortV2ReliabilityError("reliability checkpoint identity differs")
    try:
        label_records = tuple(json.loads(line) for line in labels_bytes.splitlines())
        score_record = json.loads(scores_bytes)
    except json.JSONDecodeError as error:
        raise CohortV2ReliabilityError("reliability result JSON is malformed") from error
    if (
        labels_bytes
        != b"".join(canonical_json_bytes(record) for record in label_records)
        or scores_bytes != canonical_json_bytes(score_record)
    ):
        raise CohortV2ReliabilityError("reliability result JSON is not canonical")
    expected_checkpoint_identity = identity((
        "cohort-v2-micro-reliability-preliminary-checkpoint-v1",
        reader.release_identity,
        reader.partition_identity,
        config.micro_config.identity,
        tuple(sorted(preliminary)),
        identities["preliminary"],
        config.preliminary_steps,
    ))
    label_fields = {
        "attempt_id",
        "context_fixed_step",
        "context_position",
        "exposure_role",
        "scenario_lineage_identity",
        "state_id",
        "useful",
        "usefulness_margin",
        "with_micro_objective",
        "without_micro_objective",
    }
    model_selection_attempts = {
        rollout.attempt_id for rollout in readers[2].rollouts
    }
    if (
        not label_records
        or any(type(record) is not dict or set(record) != label_fields for record in label_records)
        or any(
            (
                record["exposure_role"] == "training"
                and record["attempt_id"] not in label
            )
            or (
                record["exposure_role"] == "model_selection"
                and record["attempt_id"] not in model_selection_attempts
            )
            or record["exposure_role"] not in {"training", "model_selection"}
            for record in label_records
        )
    ):
        raise CohortV2ReliabilityError("reliability label scope is invalid")
    expected_derivation_identity = identity((
        "cohort-v2-micro-relation-usefulness-derivation-v1",
        reader.release_identity,
        reader.partition_identity,
        expected_checkpoint_identity,
        config.requested_horizon,
        "duration-weighted-carrier-mse:micro-mode-without-minus-with-oracle-relations",
        tuple(
            (
                record["state_id"],
                record["exposure_role"],
                record["without_micro_objective"],
                record["with_micro_objective"],
                record["useful"],
            )
            for record in label_records
        ),
    ))
    if (
        manifest.get("checkpoint_identity") != expected_checkpoint_identity
        or manifest.get("derivation_identity") != expected_derivation_identity
        or manifest.get("training_label_count")
        != sum(record["exposure_role"] == "training" for record in label_records)
        or manifest.get("model_selection_label_count")
        != sum(
            record["exposure_role"] == "model_selection" for record in label_records
        )
    ):
        raise CohortV2ReliabilityError("reliability derivation identity differs")
    expected_artifact_identity = identity((
        RELIABILITY_SCHEMA,
        reader.release_identity,
        reader.partition_identity,
        config.identity,
        expected_checkpoint_identity,
        expected_derivation_identity,
        tuple(sorted(identities.items())),
        _digest(labels_bytes),
        _digest(scores_bytes),
        manifest.get("implementation_revision"),
    ))
    if manifest.get("artifact_identity") != expected_artifact_identity:
        raise CohortV2ReliabilityError("reliability artifact identity differs")
    return manifest


def load_cohort_v2_reliability_estimator(
    root: Path,
    *,
    readers: tuple[CohortV2ReleaseReader, ...],
    config: CohortV2ReliabilityConfig,
    device: str,
) -> tuple[CohortV2ReliabilityEstimator, dict[str, object]]:
    """Load the validated issue-12 estimator used by downstream loss gating."""
    manifest = validate_cohort_v2_reliability_artifact(
        root, readers=readers, config=config
    )
    checkpoint = torch.load(
        Path(root) / str(manifest["checkpoint"]),
        map_location="cpu",
        weights_only=True,
    )
    estimator = CohortV2ReliabilityEstimator(
        config.feature_config.feature_dim, config.hidden_dim
    )
    try:
        estimator.load_state_dict(checkpoint["states"]["estimator"], strict=True)
    except (KeyError, RuntimeError, TypeError) as error:
        raise CohortV2ReliabilityError(
            f"reliability estimator state is invalid: {error}"
        ) from error
    estimator.to(torch.device(device)).eval()
    return estimator, manifest
