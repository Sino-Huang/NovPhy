"""Observation-bound distillation and evaluation for cohort-v2 pair controllers."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn
from torch.nn import functional

from world_model.data import CohortV2ReleaseReader, CohortV2Rollout
from world_model.model import ABSTRACTION_ORDER, PredictionPair, identity
from world_model.training.cohort_v2_evaluation import (
    CohortV2EvaluationResult,
    CohortV2PairOutcome,
)
from world_model.training.cohort_v2_measurement import (
    CohortV2MeasurementResult,
    CohortV2PairMeasurement,
)
from world_model.training.cohort_v2_trajectory_labels import (
    CohortV2ControllerLabelResult,
    CohortV2TrajectoryCostSpec,
)
from world_model.training.grid_artifacts import canonical_json_bytes


CONTROLLER_SCHEMA = "cohort_v2_distilled_pair_controllers_v1"
CONTROLLER_IDS = ("joint_pair", "matched_capacity_two_head")
CONTROLLER_ROLES = ("training", "model_selection")
EVALUATION_ROLES = ("model_selection",)


class CohortV2ControllerError(ValueError):
    """Controller inputs, checkpoints, or result artifacts are invalid."""


@dataclass(frozen=True, slots=True)
class CohortV2ControllerConfig:
    image_height: int = 6
    image_width: int = 10
    hidden_dim: int = 32
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    seed: int = 10

    def __post_init__(self) -> None:
        for field in ("image_height", "image_width", "hidden_dim", "epochs", "batch_size"):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise CohortV2ControllerError(f"{field} must be a positive integer")
        for field in ("learning_rate", "weight_decay"):
            value = getattr(self, field)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise CohortV2ControllerError(f"{field} must be finite")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise CohortV2ControllerError("optimizer values are invalid")
        if type(self.seed) is not int or self.seed < 0:
            raise CohortV2ControllerError("seed must be a nonnegative integer")

    @property
    def feature_dim(self) -> int:
        return 3 * self.image_height * self.image_width + 5

    @property
    def identity(self) -> str:
        return identity(("cohort-v2-controller-config-v1", *asdict(self).values()))


@dataclass(frozen=True, slots=True)
class CohortV2ControllerExample:
    state_id: str
    exposure_role: str
    scenario_lineage_identity: str
    features: torch.Tensor
    oracle_pair: PredictionPair
    oracle_segment_cost: float


@dataclass(frozen=True, slots=True)
class CohortV2ControllerDecision:
    controller_id: str
    state_id: str
    exposure_role: str
    scenario_lineage_identity: str
    selected_pair: PredictionPair
    oracle_pair: PredictionPair
    utility_available: bool
    prediction_objective: float | None
    endpoint_violation_rate: float | None
    policy_compute_per_simulated_frame: float | None
    full_compute_per_simulated_frame: float | None
    segment_cost: float | None
    oracle_segment_cost: float


@dataclass(frozen=True, slots=True)
class CohortV2ControllerScore:
    controller_id: str
    exposure_role: str
    state_count: int
    pair_accuracy: float
    horizon_accuracy: float
    description_mode_accuracy: float
    utility_available_count: int
    utility_unavailable_count: int
    mean_prediction_objective: float | None
    mean_endpoint_violation_rate: float | None
    mean_policy_compute_per_simulated_frame: float | None
    mean_full_compute_per_simulated_frame: float | None
    mean_selected_segment_cost: float | None
    mean_oracle_segment_cost: float
    mean_pair_regret: float | None


@dataclass(frozen=True, slots=True)
class CohortV2ControllerResult:
    decisions: tuple[CohortV2ControllerDecision, ...]
    scores: tuple[CohortV2ControllerScore, ...]


@dataclass(frozen=True, slots=True)
class CohortV2ControllerReceipt:
    controller_artifact_identity: str
    implementation_revision: str
    checkpoint_identity: str
    joint_model_state_identity: str
    two_head_model_state_identity: str
    parameter_count: int
    training_state_count: int
    evaluation_state_count: int


class CohortV2ControllerFeatureCodec:
    """Encode only the agent observation and deployment-observable rollout context."""

    def __init__(self, config: CohortV2ControllerConfig) -> None:
        self.config = config

    @property
    def identity(self) -> str:
        return identity((
            "cohort-v2-agent-observation-controller-features-v1",
            self.config.image_height,
            self.config.image_width,
            "rgb8_srgb_area_resize",
            "elapsed_fixed_steps",
            "declared_intervention",
        ))

    def encode(
        self,
        agent_observation: bytes,
        *,
        elapsed_fixed_steps: int,
        intervention: Mapping[str, object],
    ) -> torch.Tensor:
        if type(agent_observation) is not bytes:
            raise CohortV2ControllerError("controller input must be agent observation bytes")
        if type(elapsed_fixed_steps) is not int or elapsed_fixed_steps < 0:
            raise CohortV2ControllerError("elapsed fixed steps must be nonnegative")
        try:
            with Image.open(BytesIO(agent_observation)) as opened:
                pixels = np.array(opened.convert("RGB"), dtype=np.float32) / 255.0
        except (OSError, UnidentifiedImageError) as error:
            raise CohortV2ControllerError("agent observation is not a readable image") from error
        image = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
        image = functional.adaptive_avg_pool2d(
            image, (self.config.image_height, self.config.image_width)
        ).flatten()
        interface = intervention.get("interface_action")
        if not isinstance(interface, Mapping):
            raise CohortV2ControllerError("declared intervention lacks its interface action")
        drag = interface.get("drag_release")
        frame_height = interface.get("frame_height")
        release_time = interface.get("releaseTime")
        tap_time = interface.get("tapTime")
        if (
            not isinstance(drag, tuple)
            or len(drag) != 2
            or type(frame_height) not in (int, float)
            or frame_height <= 0
            or type(release_time) not in (int, float)
            or type(tap_time) not in (int, float)
        ):
            raise CohortV2ControllerError("declared intervention values are malformed")
        context = torch.tensor((
            elapsed_fixed_steps / 1000.0,
            float(drag[0]) / float(frame_height),
            float(drag[1]) / float(frame_height),
            float(release_time) / 1000.0,
            float(tap_time) / 1000.0,
        ), dtype=torch.float32)
        return torch.cat((image, context))


class _ControllerBackbone(nn.Module):
    def __init__(self, config: CohortV2ControllerConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class CohortV2JointPairController(nn.Module):
    def __init__(self, config: CohortV2ControllerConfig) -> None:
        super().__init__()
        self.backbone = _ControllerBackbone(config)
        self.pair_head = nn.Linear(config.hidden_dim, 9)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.pair_head(self.backbone(features))


class CohortV2TwoHeadController(nn.Module):
    """Independent axis heads with a shared adapter used to match joint capacity."""

    def __init__(self, config: CohortV2ControllerConfig) -> None:
        super().__init__()
        self.backbone = _ControllerBackbone(config)
        self.horizon_head = nn.Linear(config.hidden_dim, 3)
        self.mode_head = nn.Linear(config.hidden_dim, 3)
        self.shared_adapter = nn.Linear(config.hidden_dim, 3)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        shared = self.shared_adapter(hidden)
        return self.horizon_head(hidden) + shared, self.mode_head(hidden) + shared


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _model_state_identity(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return f"sha256:{digest.hexdigest()}"


def _pair_index(pair: PredictionPair, pairs: tuple[PredictionPair, ...]) -> int:
    return pairs.index(pair)


def build_cohort_v2_controller_examples(
    readers: tuple[CohortV2ReleaseReader, ...],
    labels: CohortV2ControllerLabelResult,
    config: CohortV2ControllerConfig,
    *,
    included_roles: tuple[str, ...] = CONTROLLER_ROLES,
) -> tuple[CohortV2ControllerExample, ...]:
    """Join trajectory labels to agent-only features without reading engine state."""
    if not isinstance(labels, CohortV2ControllerLabelResult):
        raise CohortV2ControllerError("controller examples require trajectory labels")
    if tuple(reader.rollouts[0].exposure_role for reader in readers) != (
        "training", "calibration", "model_selection"
    ):
        raise CohortV2ControllerError("controller readers must preserve public role order")
    if (
        not included_roles
        or len(set(included_roles)) != len(included_roles)
        or any(
            role not in ("training", "calibration", "model_selection")
            for role in included_roles
        )
    ):
        raise CohortV2ControllerError("controller example roles are invalid")
    codec = CohortV2ControllerFeatureCodec(config)
    rollouts: dict[str, tuple[CohortV2ReleaseReader, CohortV2Rollout]] = {}
    observations: dict[str, bytes] = {}
    for reader in readers:
        for rollout in reader.rollouts:
            if rollout.attempt_id in rollouts:
                raise CohortV2ControllerError("controller rollout identities are not unique")
            rollouts[rollout.attempt_id] = (reader, rollout)
    examples = []
    for label in labels.labels:
        if label.exposure_role not in included_roles:
            continue
        item = rollouts.get(label.attempt_id)
        if item is None or item[1].exposure_role != label.exposure_role:
            raise CohortV2ControllerError("trajectory label crossed its rollout role")
        reader, rollout = item
        if label.context_position >= len(rollout.frame_records) - 1:
            raise CohortV2ControllerError("trajectory label context is not a rollout state")
        observation = observations.get(label.attempt_id)
        if observation is None:
            observation = reader.load_observation(rollout, observation_role="agent")
            observations[label.attempt_id] = observation
        examples.append(CohortV2ControllerExample(
            state_id=label.state_id,
            exposure_role=label.exposure_role,
            scenario_lineage_identity=label.scenario_lineage_identity,
            features=codec.encode(
                observation,
                elapsed_fixed_steps=(
                    label.context_fixed_step - rollout.frame_records[0].fixed_step
                ),
                intervention=rollout.intervention,
            ),
            oracle_pair=label.selected_pair,
            oracle_segment_cost=label.segment_cost,
        ))
    return tuple(examples)


def _train_one(
    controller_id: str,
    model: nn.Module,
    examples: tuple[CohortV2ControllerExample, ...],
    pairs: tuple[PredictionPair, ...],
    config: CohortV2ControllerConfig,
    progress: Callable[[str], None] | None,
) -> None:
    training = tuple(item for item in examples if item.exposure_role == "training")
    if not training:
        raise CohortV2ControllerError("controller training role is empty")
    features = torch.stack(tuple(item.features for item in training))
    pair_targets = torch.tensor(
        tuple(_pair_index(item.oracle_pair, pairs) for item in training), dtype=torch.long
    )
    horizons = tuple(dict.fromkeys(pair.delta for pair in pairs))
    horizon_targets = torch.tensor(
        tuple(horizons.index(item.oracle_pair.delta) for item in training), dtype=torch.long
    )
    mode_targets = torch.tensor(
        tuple(ABSTRACTION_ORDER.index(item.oracle_pair.abstraction) for item in training),
        dtype=torch.long,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    model.train()
    for epoch in range(config.epochs):
        order = torch.randperm(len(training), generator=generator)
        total_loss = 0.0
        for start in range(0, len(training), config.batch_size):
            indices = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            if controller_id == "joint_pair":
                loss = functional.cross_entropy(model(features[indices]), pair_targets[indices])
            else:
                horizon_logits, mode_logits = model(features[indices])
                loss = 0.5 * (
                    functional.cross_entropy(horizon_logits, horizon_targets[indices])
                    + functional.cross_entropy(mode_logits, mode_targets[indices])
                )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        if progress is not None and (
            epoch == 0 or epoch + 1 == config.epochs or (epoch + 1) % 10 == 0
        ):
            progress(
                f"[train:{controller_id}] epoch={epoch + 1}/{config.epochs} "
                f"loss={total_loss / len(training):.6f}"
            )


def train_cohort_v2_controllers(
    examples: tuple[CohortV2ControllerExample, ...],
    pairs: tuple[PredictionPair, ...],
    config: CohortV2ControllerConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[CohortV2JointPairController, CohortV2TwoHeadController]:
    if len(pairs) != 9:
        raise CohortV2ControllerError("controller training requires the declared 3x3 pair grid")
    torch.manual_seed(config.seed)
    joint = CohortV2JointPairController(config)
    torch.manual_seed(config.seed)
    two_head = CohortV2TwoHeadController(config)
    if _parameter_count(joint) != _parameter_count(two_head):
        raise CohortV2ControllerError("joint and two-head controller capacity differs")
    _train_one("joint_pair", joint, examples, pairs, config, progress)
    _train_one("matched_capacity_two_head", two_head, examples, pairs, config, progress)
    joint.eval()
    two_head.eval()
    return joint, two_head


def select_cohort_v2_controller_pairs(
    controller_id: str,
    model: nn.Module,
    features: torch.Tensor,
    pairs: tuple[PredictionPair, ...],
) -> tuple[PredictionPair, ...]:
    with torch.no_grad():
        if controller_id == "joint_pair":
            indices = model(features).argmax(dim=1).tolist()
            return tuple(pairs[index] for index in indices)
        horizon_logits, mode_logits = model(features)
        horizons = tuple(dict.fromkeys(pair.delta for pair in pairs))
        horizon_indices = horizon_logits.argmax(dim=1).tolist()
        mode_indices = mode_logits.argmax(dim=1).tolist()
        return tuple(
            PredictionPair(horizons[horizon], ABSTRACTION_ORDER[mode])
            for horizon, mode in zip(horizon_indices, mode_indices, strict=True)
        )


def cohort_v2_pair_utility(
    state,
    outcome: CohortV2PairOutcome,
    measured: CohortV2PairMeasurement,
    spec: CohortV2TrajectoryCostSpec,
) -> tuple[float, float, float, float, float] | None:
    if not outcome.available or outcome.objective is None:
        return None
    plausibility = measured.endpoint_plausibility
    compute = measured.compute
    if plausibility is None or compute is None or plausibility.violation_rate is None:
        return None
    duration_weight = outcome.effective_horizon / (state.frame_record_count - 1)
    segment_cost = (
        float(outcome.objective)
        + duration_weight * spec.physical_violation_weight * plausibility.violation_rate
        + spec.compute_weight * compute.policy_dependent_total / spec.compute_reference
    )
    return (
        float(outcome.objective),
        plausibility.violation_rate,
        compute.policy_dependent_per_simulated_frame,
        compute.full_end_to_end_per_simulated_frame,
        segment_cost,
    )


def _mean(values: tuple[float, ...]) -> float | None:
    if not values:
        return None
    # Keep artifact aggregation stable across Python 3.11's left-to-right sum
    # and the compensated float sum introduced in Python 3.12.
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def evaluate_cohort_v2_controllers(
    models: tuple[CohortV2JointPairController, CohortV2TwoHeadController],
    examples: tuple[CohortV2ControllerExample, ...],
    evaluation: CohortV2EvaluationResult,
    measurement: CohortV2MeasurementResult,
    spec: CohortV2TrajectoryCostSpec,
    *,
    evaluation_roles: tuple[str, ...] = EVALUATION_ROLES,
) -> CohortV2ControllerResult:
    if measurement.evaluation_identity != evaluation.identity:
        raise CohortV2ControllerError("controller utilities do not belong to the evaluation")
    by_state = {
        state.state_id: (state, measured)
        for state, measured in zip(evaluation.states, measurement.states, strict=True)
    }
    if (
        not evaluation_roles
        or len(set(evaluation_roles)) != len(evaluation_roles)
        or any(role not in ("calibration", "model_selection") for role in evaluation_roles)
    ):
        raise CohortV2ControllerError("controller evaluation roles are invalid")
    held_out = tuple(item for item in examples if item.exposure_role in evaluation_roles)
    if not held_out:
        raise CohortV2ControllerError("controller evaluation scope is empty")
    features = torch.stack(tuple(item.features for item in held_out))
    decisions = []
    for controller_id, model in zip(CONTROLLER_IDS, models, strict=True):
        selected = select_cohort_v2_controller_pairs(
            controller_id, model, features, evaluation.grid.pairs
        )
        for example, pair in zip(held_out, selected, strict=True):
            state, measured = by_state[example.state_id]
            pair_index = evaluation.grid.pairs.index(pair)
            utility = cohort_v2_pair_utility(
                state, state.outcomes[pair_index], measured.outcomes[pair_index], spec
            )
            decisions.append(CohortV2ControllerDecision(
                controller_id=controller_id,
                state_id=example.state_id,
                exposure_role=example.exposure_role,
                scenario_lineage_identity=example.scenario_lineage_identity,
                selected_pair=pair,
                oracle_pair=example.oracle_pair,
                utility_available=utility is not None,
                prediction_objective=None if utility is None else utility[0],
                endpoint_violation_rate=None if utility is None else utility[1],
                policy_compute_per_simulated_frame=None if utility is None else utility[2],
                full_compute_per_simulated_frame=None if utility is None else utility[3],
                segment_cost=None if utility is None else utility[4],
                oracle_segment_cost=example.oracle_segment_cost,
            ))
    scores = []
    for role in evaluation_roles:
        for controller_id in CONTROLLER_IDS:
            rows = tuple(
                item for item in decisions
                if item.controller_id == controller_id and item.exposure_role == role
            )
            available = tuple(item for item in rows if item.utility_available)
            state_count = len(rows)
            scores.append(CohortV2ControllerScore(
                controller_id=controller_id,
                exposure_role=role,
                state_count=state_count,
                pair_accuracy=sum(item.selected_pair == item.oracle_pair for item in rows) / state_count,
                horizon_accuracy=sum(item.selected_pair.delta == item.oracle_pair.delta for item in rows) / state_count,
                description_mode_accuracy=sum(
                    item.selected_pair.abstraction is item.oracle_pair.abstraction for item in rows
                ) / state_count,
                utility_available_count=len(available),
                utility_unavailable_count=state_count - len(available),
                mean_prediction_objective=_mean(tuple(item.prediction_objective for item in available)),
                mean_endpoint_violation_rate=_mean(tuple(item.endpoint_violation_rate for item in available)),
                mean_policy_compute_per_simulated_frame=_mean(tuple(
                    item.policy_compute_per_simulated_frame for item in available
                )),
                mean_full_compute_per_simulated_frame=_mean(tuple(
                    item.full_compute_per_simulated_frame for item in available
                )),
                mean_selected_segment_cost=_mean(tuple(item.segment_cost for item in available)),
                mean_oracle_segment_cost=_mean(tuple(
                    item.oracle_segment_cost for item in rows
                )),
                mean_pair_regret=_mean(tuple(
                    item.segment_cost - item.oracle_segment_cost for item in available
                )),
            ))
    return CohortV2ControllerResult(tuple(decisions), tuple(scores))


def _pair_payload(pair: PredictionPair) -> dict[str, str | int]:
    return {"requested_horizon": pair.delta, "abstraction": str(pair.abstraction)}


def _decision_payload(item: CohortV2ControllerDecision) -> dict[str, object]:
    value = asdict(item)
    value["selected_pair"] = _pair_payload(item.selected_pair)
    value["oracle_pair"] = _pair_payload(item.oracle_pair)
    value["record_type"] = "controller_decision"
    value["schema"] = CONTROLLER_SCHEMA
    return value


def _result_bytes(result: CohortV2ControllerResult) -> tuple[bytes, bytes]:
    decisions = b"".join(canonical_json_bytes(_decision_payload(item)) for item in result.decisions)
    scores = canonical_json_bytes({
        "evaluation_roles": list(EVALUATION_ROLES),
        "schema": CONTROLLER_SCHEMA,
        "scores": [asdict(item) for item in result.scores],
    })
    return decisions, scores


def _bytes_identity(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _short_identity(value: object) -> str:
    text = str(value)
    if text.startswith("sha256:"):
        return text[:19]
    return text if len(text) <= 32 else text[:29] + "..."


def _display_value(value: object) -> str:
    text = repr(value)
    return text if len(text) <= 120 else text[:117] + "..."


def _first_value_difference(expected: object, actual: object, path: str = "$") -> str:
    if type(expected) is not type(actual):
        return (
            f"field={path} expected_type={type(expected).__name__} "
            f"actual_type={type(actual).__name__}"
        )
    if isinstance(expected, dict):
        for key in sorted(set(expected) | set(actual)):
            field = f"{path}.{key}"
            if key not in expected:
                return f"field={field} expected=<absent> actual={_display_value(actual[key])}"
            if key not in actual:
                return f"field={field} expected={_display_value(expected[key])} actual=<absent>"
            if expected[key] != actual[key]:
                return _first_value_difference(expected[key], actual[key], field)
    elif isinstance(expected, list):
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            if expected_item != actual_item:
                return _first_value_difference(
                    expected_item, actual_item, f"{path}[{index}]"
                )
        if len(expected) != len(actual):
            return f"field={path} expected_length={len(expected)} actual_length={len(actual)}"
    elif expected != actual:
        return (
            f"field={path} expected={_display_value(expected)} "
            f"actual={_display_value(actual)}"
        )
    return f"field={path} values differ"


def _first_jsonl_difference(expected: bytes, actual: bytes) -> str:
    expected_lines = expected.splitlines()
    actual_lines = actual.splitlines()
    for index, (expected_line, actual_line) in enumerate(
        zip(expected_lines, actual_lines)
    ):
        if expected_line == actual_line:
            continue
        try:
            expected_record = json.loads(expected_line)
        except json.JSONDecodeError as error:
            return f"record={index} expected_json_error={error.msg}"
        try:
            actual_record = json.loads(actual_line)
        except json.JSONDecodeError as error:
            return f"record={index} actual_json_error={error.msg}"
        return f"record={index} {_first_value_difference(expected_record, actual_record)}"
    return (
        f"record={min(len(expected_lines), len(actual_lines))} "
        f"expected_record_count={len(expected_lines)} actual_record_count={len(actual_lines)}"
    )


def _first_json_difference(expected: bytes, actual: bytes) -> str:
    try:
        expected_value = json.loads(expected)
    except json.JSONDecodeError as error:
        return f"field=$ expected_json_error={error.msg}"
    try:
        actual_value = json.loads(actual)
    except json.JSONDecodeError as error:
        return f"field=$ actual_json_error={error.msg}"
    return _first_value_difference(expected_value, actual_value)


def _validation_diagnostics(root: Path) -> str:
    files = []
    for name in (
        "checkpoint.pt",
        "controller_decisions.jsonl",
        "scores.json",
        "manifest.json",
    ):
        try:
            status = (root / name).stat()
            files.append(f"{name}(size={status.st_size},mtime_ns={status.st_mtime_ns})")
        except OSError as error:
            files.append(f"{name}({type(error).__name__})")
    return (
        "runtime=("
        f"python={sys.executable},python_version={sys.version_info.major}."
        f"{sys.version_info.minor}.{sys.version_info.micro},torch={torch.__version__},"
        f"threads={torch.get_num_threads()},interop_threads={torch.get_num_interop_threads()},"
        f"deterministic_algorithms={torch.are_deterministic_algorithms_enabled()},"
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')!r},"
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')!r},"
        f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')!r}) "
        f"files=({','.join(files)})"
    )


def _validation_mismatch(
    root: Path,
    component: str,
    expected_identity: object,
    actual_identity: object,
    difference: str,
) -> CohortV2ControllerError:
    return CohortV2ControllerError(
        f"controller validation failed [{component}]: "
        f"expected={_short_identity(expected_identity)} "
        f"actual={_short_identity(actual_identity)}; {difference}; "
        f"{_validation_diagnostics(root)}"
    )


def _checkpoint_payload(
    models: tuple[CohortV2JointPairController, CohortV2TwoHeadController],
    config: CohortV2ControllerConfig,
) -> dict[str, object]:
    joint, two_head = models
    joint_state = {name: value.detach().cpu() for name, value in joint.state_dict().items()}
    two_head_state = {name: value.detach().cpu() for name, value in two_head.state_dict().items()}
    return {
        "schema": CONTROLLER_SCHEMA,
        "config": asdict(config),
        "config_identity": config.identity,
        "joint_model_state": joint_state,
        "joint_model_state_identity": _model_state_identity(joint_state),
        "two_head_model_state": two_head_state,
        "two_head_model_state_identity": _model_state_identity(two_head_state),
        "parameter_count": _parameter_count(joint),
    }


def _load_checkpoint(path: Path) -> tuple[
    tuple[CohortV2JointPairController, CohortV2TwoHeadController], dict[str, object]
]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        config = CohortV2ControllerConfig(**payload["config"])
        joint = CohortV2JointPairController(config)
        two_head = CohortV2TwoHeadController(config)
        joint.load_state_dict(payload["joint_model_state"], strict=True)
        two_head.load_state_dict(payload["two_head_model_state"], strict=True)
    except (OSError, KeyError, TypeError, RuntimeError, CohortV2ControllerError) as error:
        raise CohortV2ControllerError(f"cannot load controller checkpoint: {error}") from error
    if (
        payload.get("schema") != CONTROLLER_SCHEMA
        or payload.get("config_identity") != config.identity
        or payload.get("joint_model_state_identity") != _model_state_identity(joint.state_dict())
        or payload.get("two_head_model_state_identity") != _model_state_identity(two_head.state_dict())
        or payload.get("parameter_count") != _parameter_count(joint)
        or _parameter_count(joint) != _parameter_count(two_head)
    ):
        raise CohortV2ControllerError("controller checkpoint identity or capacity is stale")
    joint.eval()
    two_head.eval()
    return (joint, two_head), payload


def load_cohort_v2_controller_checkpoint(
    path: Path,
) -> tuple[
    tuple[CohortV2JointPairController, CohortV2TwoHeadController],
    CohortV2ControllerConfig,
    str,
]:
    """Load the source-bound controller pair for downstream experiments."""
    models, payload = _load_checkpoint(path)
    config = CohortV2ControllerConfig(**payload["config"])
    checkpoint_identity = identity((
        "cohort-v2-controller-checkpoint-v1",
        config.identity,
        payload["joint_model_state_identity"],
        payload["two_head_model_state_identity"],
    ))
    return models, config, checkpoint_identity


def _manifest(
    result: CohortV2ControllerResult,
    checkpoint: Mapping[str, object],
    config: CohortV2ControllerConfig,
    evaluation: CohortV2EvaluationResult,
    measurement: CohortV2MeasurementResult,
    labels: CohortV2ControllerLabelResult,
    examples: tuple[CohortV2ControllerExample, ...],
    artifact_bytes: tuple[bytes, bytes],
    *,
    trajectory_label_artifact_identity: str,
    baseline_artifact_identity: str,
    derivation_index_identity: str,
    implementation_revision: str,
) -> dict[str, object]:
    for field, value in (
        ("trajectory label artifact", trajectory_label_artifact_identity),
        ("baseline artifact", baseline_artifact_identity),
        ("derivation index", derivation_index_identity),
        ("implementation revision", implementation_revision),
    ):
        if type(value) is not str or not value:
            raise CohortV2ControllerError(f"{field} must be a nonempty identity")
    decisions, scores = artifact_bytes
    checkpoint_identity = identity((
        "cohort-v2-controller-checkpoint-v1",
        config.identity,
        checkpoint["joint_model_state_identity"],
        checkpoint["two_head_model_state_identity"],
    ))
    artifact_identity = identity((
        "cohort-v2-controller-artifact-v1",
        checkpoint_identity,
        evaluation.identity,
        measurement.identity,
        trajectory_label_artifact_identity,
        baseline_artifact_identity,
        derivation_index_identity,
        implementation_revision,
        _bytes_identity(decisions),
        _bytes_identity(scores),
    ))
    training_count = sum(item.exposure_role == "training" for item in examples)
    return {
        "artifact_type": "cohort_v2_distilled_pair_controllers",
        "baseline_artifact_identity": baseline_artifact_identity,
        "capability_declaration_identity": evaluation.capability_declaration_identity,
        "checkpoint": "checkpoint.pt",
        "checkpoint_identity": checkpoint_identity,
        "config": asdict(config),
        "config_identity": config.identity,
        "controller_artifact_identity": artifact_identity,
        "controllers": list(CONTROLLER_IDS),
        "decisions": "controller_decisions.jsonl",
        "decisions_identity": _bytes_identity(decisions),
        "deployment_inputs": [
            "agent_observation",
            "declared_intervention",
            "elapsed_fixed_step_position",
        ],
        "derivation_index_identity": derivation_index_identity,
        "evaluation_identity": evaluation.identity,
        "evaluation_roles": list(EVALUATION_ROLES),
        "evaluation_state_count": len(result.decisions) // len(CONTROLLER_IDS),
        "final_evaluation_consumed": False,
        "grid_identity": evaluation.grid.identity,
        "implementation_revision": implementation_revision,
        "joint_model_state_identity": checkpoint["joint_model_state_identity"],
        "matched_parameter_count": checkpoint["parameter_count"],
        "measurement_identity": measurement.identity,
        "objective_identity": evaluation.objective_identity,
        "oracle_engine_state_is_controller_input": False,
        "parameter_matching": "same_backbone_and_equal_total_trainable_parameters",
        "partition_identity": evaluation.partition_identity,
        "release_identity": evaluation.release_identity,
        "role_permissions": {
            "training": ["learned_parameters"],
            "calibration": [],
            "model_selection": ["configuration_selection"],
        },
        "schema": CONTROLLER_SCHEMA,
        "scores": "scores.json",
        "scores_identity": _bytes_identity(scores),
        "teacher": str(labels.teacher),
        "training_role": "training",
        "training_state_count": training_count,
        "trajectory_label_artifact_identity": trajectory_label_artifact_identity,
        "two_head_model_state_identity": checkpoint["two_head_model_state_identity"],
    }


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _save_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _receipt(manifest: Mapping[str, object]) -> CohortV2ControllerReceipt:
    return CohortV2ControllerReceipt(
        controller_artifact_identity=manifest["controller_artifact_identity"],
        implementation_revision=manifest["implementation_revision"],
        checkpoint_identity=manifest["checkpoint_identity"],
        joint_model_state_identity=manifest["joint_model_state_identity"],
        two_head_model_state_identity=manifest["two_head_model_state_identity"],
        parameter_count=manifest["matched_parameter_count"],
        training_state_count=manifest["training_state_count"],
        evaluation_state_count=manifest["evaluation_state_count"],
    )


def write_cohort_v2_controllers(
    root: Path,
    readers: tuple[CohortV2ReleaseReader, ...],
    evaluation: CohortV2EvaluationResult,
    measurement: CohortV2MeasurementResult,
    labels: CohortV2ControllerLabelResult,
    spec: CohortV2TrajectoryCostSpec,
    config: CohortV2ControllerConfig,
    *,
    trajectory_label_artifact_identity: str,
    baseline_artifact_identity: str,
    derivation_index_identity: str,
    implementation_revision: str,
    progress: Callable[[str], None] | None = None,
) -> CohortV2ControllerReceipt:
    examples = build_cohort_v2_controller_examples(readers, labels, config)
    models = train_cohort_v2_controllers(
        examples, evaluation.grid.pairs, config, progress=progress
    )
    result = evaluate_cohort_v2_controllers(
        models, examples, evaluation, measurement, spec
    )
    checkpoint = _checkpoint_payload(models, config)
    artifacts = _result_bytes(result)
    manifest = _manifest(
        result, checkpoint, config, evaluation, measurement, labels, examples, artifacts,
        trajectory_label_artifact_identity=trajectory_label_artifact_identity,
        baseline_artifact_identity=baseline_artifact_identity,
        derivation_index_identity=derivation_index_identity,
        implementation_revision=implementation_revision,
    )
    root = Path(root)
    _save_checkpoint(root / "checkpoint.pt", checkpoint)
    _atomic_write(root / "controller_decisions.jsonl", artifacts[0])
    _atomic_write(root / "scores.json", artifacts[1])
    _atomic_write(root / "manifest.json", canonical_json_bytes(manifest))
    return validate_cohort_v2_controllers(
        root, readers, evaluation, measurement, labels, spec,
        trajectory_label_artifact_identity=trajectory_label_artifact_identity,
        baseline_artifact_identity=baseline_artifact_identity,
        derivation_index_identity=derivation_index_identity,
        implementation_revision=implementation_revision,
    )


def validate_cohort_v2_controllers(
    root: Path,
    readers: tuple[CohortV2ReleaseReader, ...],
    evaluation: CohortV2EvaluationResult,
    measurement: CohortV2MeasurementResult,
    labels: CohortV2ControllerLabelResult,
    spec: CohortV2TrajectoryCostSpec,
    *,
    trajectory_label_artifact_identity: str,
    baseline_artifact_identity: str,
    derivation_index_identity: str,
    implementation_revision: str,
) -> CohortV2ControllerReceipt:
    root = Path(root)
    try:
        manifest_raw = (root / "manifest.json").read_bytes()
        manifest = json.loads(manifest_raw)
        actual_decisions = (root / "controller_decisions.jsonl").read_bytes()
        actual_scores = (root / "scores.json").read_bytes()
    except (OSError, json.JSONDecodeError) as error:
        raise CohortV2ControllerError(f"cannot load controller artifacts: {error}") from error

    canonical_manifest = canonical_json_bytes(manifest)
    if canonical_manifest != manifest_raw:
        differing_byte = next(
            (
                index
                for index, (expected_byte, actual_byte) in enumerate(
                    zip(canonical_manifest, manifest_raw)
                )
                if expected_byte != actual_byte
            ),
            min(len(canonical_manifest), len(manifest_raw)),
        )
        raise _validation_mismatch(
            root,
            "canonical_manifest",
            _bytes_identity(canonical_manifest),
            _bytes_identity(manifest_raw),
            f"field=canonical_json_encoding first_differing_byte={differing_byte}",
        )
    if not isinstance(manifest, dict):
        raise _validation_mismatch(
            root,
            "recomputed_manifest_provenance",
            "json_object",
            type(manifest).__name__,
            "field=$ expected_type=dict "
            f"actual_type={type(manifest).__name__}",
        )

    try:
        models, checkpoint = _load_checkpoint(root / "checkpoint.pt")
    except CohortV2ControllerError as error:
        raise _validation_mismatch(
            root,
            "stored_artifact_identities",
            "valid_checkpoint",
            "invalid_checkpoint",
            f"field=checkpoint.pt error={error}",
        ) from error
    config = CohortV2ControllerConfig(**checkpoint["config"])
    checkpoint_identity = identity((
        "cohort-v2-controller-checkpoint-v1",
        config.identity,
        checkpoint["joint_model_state_identity"],
        checkpoint["two_head_model_state_identity"],
    ))
    stored_identities = (
        ("decisions_identity", _bytes_identity(actual_decisions)),
        ("scores_identity", _bytes_identity(actual_scores)),
        ("checkpoint_identity", checkpoint_identity),
    )
    for field, observed in stored_identities:
        declared = manifest.get(field, "<absent>")
        if declared != observed:
            raise _validation_mismatch(
                root,
                "stored_artifact_identities",
                declared,
                observed,
                f"field={field} expected={_display_value(declared)} "
                f"actual={_display_value(observed)}",
            )

    examples = build_cohort_v2_controller_examples(readers, labels, config)
    result = evaluate_cohort_v2_controllers(models, examples, evaluation, measurement, spec)
    expected_decisions, expected_scores = _result_bytes(result)
    if expected_decisions != actual_decisions:
        raise _validation_mismatch(
            root,
            "recomputed_decisions",
            _bytes_identity(expected_decisions),
            _bytes_identity(actual_decisions),
            _first_jsonl_difference(expected_decisions, actual_decisions),
        )
    if expected_scores != actual_scores:
        raise _validation_mismatch(
            root,
            "recomputed_scores",
            _bytes_identity(expected_scores),
            _bytes_identity(actual_scores),
            _first_json_difference(expected_scores, actual_scores),
        )

    expected = (expected_decisions, expected_scores)
    expected_manifest = _manifest(
        result, checkpoint, config, evaluation, measurement, labels, examples, expected,
        trajectory_label_artifact_identity=trajectory_label_artifact_identity,
        baseline_artifact_identity=baseline_artifact_identity,
        derivation_index_identity=derivation_index_identity,
        implementation_revision=implementation_revision,
    )
    if manifest != expected_manifest:
        raise _validation_mismatch(
            root,
            "recomputed_manifest_provenance",
            _bytes_identity(canonical_json_bytes(expected_manifest)),
            _bytes_identity(manifest_raw),
            _first_value_difference(expected_manifest, manifest),
        )
    return _receipt(manifest)


__all__ = [
    "CohortV2ControllerConfig",
    "CohortV2ControllerDecision",
    "CohortV2ControllerError",
    "CohortV2ControllerExample",
    "CohortV2ControllerFeatureCodec",
    "CohortV2ControllerReceipt",
    "CohortV2ControllerResult",
    "CohortV2ControllerScore",
    "CohortV2JointPairController",
    "CohortV2TwoHeadController",
    "build_cohort_v2_controller_examples",
    "cohort_v2_pair_utility",
    "evaluate_cohort_v2_controllers",
    "load_cohort_v2_controller_checkpoint",
    "select_cohort_v2_controller_pairs",
    "train_cohort_v2_controllers",
    "validate_cohort_v2_controllers",
    "write_cohort_v2_controllers",
]
