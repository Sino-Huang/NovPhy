"""Short self-conditioned h15 training for deployment-carrier predictors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch.nn import functional as F

from world_model.model import (
    Abstraction,
    DualOutputPredictor,
    PredictionPair,
    PredictorConfig,
)
from world_model.training.lineage_scaling import (
    CarrierKind,
    CarrierLineage,
    ContinuousTransitionExample,
    LineageScalingError,
)


H15 = 15
SUPPORTED_UNROLL_STEPS = (1, 2, 4)


@dataclass(frozen=True, slots=True)
class ShortUnrollTrainingSpec:
    """One prospectively declared h15 optimizer cell."""

    name: str
    unroll_steps: int
    local_loss_weight: float
    unrolled_loss_weight: float
    carrier_bound: float
    carrier_bound_loss_weight: float
    optimizer_example_budget: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    seed: int
    carrier_identity: str
    lineage_manifest_reference: str
    predictor_config: PredictorConfig

    def __post_init__(self) -> None:
        numeric_weights = (
            self.local_loss_weight,
            self.unrolled_loss_weight,
            self.carrier_bound_loss_weight,
        )
        if (
            not self.name
            or self.unroll_steps not in SUPPORTED_UNROLL_STEPS
            or any(not math.isfinite(value) or value < 0.0 for value in numeric_weights)
            or not any(value > 0.0 for value in numeric_weights)
            or not math.isfinite(self.carrier_bound)
            or self.carrier_bound <= 0.0
            or type(self.optimizer_example_budget) is not int
            or self.optimizer_example_budget <= 0
            or type(self.batch_size) is not int
            or self.batch_size <= 0
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0.0
            or not math.isfinite(self.grad_clip)
            or self.grad_clip < 0.0
            or type(self.seed) is not int
            or self.seed < 0
            or not self.carrier_identity
            or not self.lineage_manifest_reference
            or type(self.predictor_config) is not PredictorConfig
            or self.predictor_config.action_dim != 5
        ):
            raise LineageScalingError("short-unroll training specification is invalid")
        if self.unroll_steps == 1 and (
            self.unrolled_loss_weight != 0.0
            or self.carrier_bound_loss_weight != 0.0
        ):
            raise LineageScalingError(
                "the one-step matched baseline must remain teacher forced"
            )

    @property
    def checkpoint_identity(self) -> str:
        return f"issue-67-short-unroll-checkpoint-v1:{self.name}:seed-{self.seed}"


@dataclass(frozen=True, slots=True)
class ShortUnrollWindow:
    """An exact h15 sequence contained by one shot segment."""

    lineage_ordinal: int
    segment_ordinal: int
    start_decision_index: int
    transitions: tuple[ContinuousTransitionExample, ...]


def build_short_unroll_windows(
    lineages: tuple[CarrierLineage, ...],
    *,
    unroll_steps: int,
) -> tuple[ShortUnrollWindow, ...]:
    """Return every exact-length h15 window without crossing role, lineage, or shot."""

    if (
        type(lineages) is not tuple
        or not lineages
        or unroll_steps not in SUPPORTED_UNROLL_STEPS
        or any(type(lineage) is not CarrierLineage for lineage in lineages)
        or len({lineage.scenario_lineage_identity for lineage in lineages})
        != len(lineages)
        or {lineage.exposure_role for lineage in lineages} != {"training"}
        or {lineage.carrier for lineage in lineages} != {CarrierKind.DEPLOYMENT}
        or len({lineage.carrier_identity for lineage in lineages}) != 1
    ):
        raise LineageScalingError(
            "short-unroll windows require unique deployment training lineages"
        )

    result: list[ShortUnrollWindow] = []
    for lineage_ordinal, lineage in enumerate(lineages, start=1):
        h15 = {
            transition.decision_index: transition
            for transition in lineage.transitions
            if transition.horizon == H15
        }
        starts = (0, *lineage.segment_ends[:-1])
        for segment_ordinal, (segment_start, segment_end) in enumerate(
            zip(starts, lineage.segment_ends, strict=True), start=1
        ):
            position = segment_start
            segment: list[ContinuousTransitionExample] = []
            while position < segment_end:
                transition = h15.get(position)
                if (
                    transition is None
                    or transition.target_decision_index
                    != min(position + H15, segment_end)
                ):
                    raise LineageScalingError(
                        "complete training segment lacks contiguous h15 windows"
                    )
                segment.append(transition)
                position = transition.target_decision_index
            for offset in range(0, len(segment) - unroll_steps + 1):
                transitions = tuple(segment[offset : offset + unroll_steps])
                result.append(ShortUnrollWindow(
                    lineage_ordinal=lineage_ordinal,
                    segment_ordinal=segment_ordinal,
                    start_decision_index=transitions[0].decision_index,
                    transitions=transitions,
                ))
    if not result:
        raise LineageScalingError(
            f"training data has no complete {unroll_steps}-step h15 windows"
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ShortUnrollLosses:
    local: torch.Tensor
    unrolled: torch.Tensor
    carrier_bound_penalty: torch.Tensor


def short_unroll_losses(
    model: torch.nn.Module,
    windows: tuple[ShortUnrollWindow, ...],
    *,
    carrier_bound: float,
    device: torch.device,
) -> ShortUnrollLosses:
    """Compute teacher-forced and self-conditioned losses for one window batch."""

    if not windows or len({len(window.transitions) for window in windows}) != 1:
        raise LineageScalingError("short-unroll loss requires equal-length windows")
    contexts = torch.stack(tuple(
        torch.stack(tuple(item.context for item in window.transitions))
        for window in windows
    )).to(device)
    actions = torch.stack(tuple(
        torch.stack(tuple(item.action for item in window.transitions))
        for window in windows
    )).to(device)
    targets = torch.stack(tuple(
        torch.stack(tuple(item.target for item in window.transitions))
        for window in windows
    )).to(device)
    batch_size, step_count, latent_dim = contexts.shape
    pair = PredictionPair(H15, Abstraction.CONTINUOUS)
    local_predictions = model.carrier(
        contexts.reshape(batch_size * step_count, latent_dim),
        actions.reshape(batch_size * step_count, actions.shape[-1]),
        pair,
    ).reshape(batch_size, step_count, latent_dim)

    recursive = [local_predictions[:, 0]]
    current = recursive[0]
    for step in range(1, step_count):
        current = model.carrier(current, actions[:, step], pair)
        recursive.append(current)
    recursive_predictions = torch.stack(recursive, dim=1)
    return ShortUnrollLosses(
        local=F.mse_loss(local_predictions, targets),
        unrolled=F.mse_loss(recursive_predictions, targets),
        carrier_bound_penalty=torch.relu(
            recursive_predictions.abs() - carrier_bound
        ).pow(2).mean(),
    )


@dataclass(frozen=True, slots=True)
class ShortUnrollTrainingReport:
    checkpoint_identity: str
    spec: ShortUnrollTrainingSpec
    lineage_count: int
    available_window_count: int
    optimizer_examples: int
    optimizer_steps: int
    effective_epochs: float
    mean_local_loss: float
    mean_unrolled_loss: float
    mean_carrier_bound_penalty: float
    mean_objective: float
    model_evaluations: int
    wall_seconds: float
    device: str
    parameter_count: int
    failures: tuple[str, ...]


def _scheduled_window_batches(
    windows: tuple[ShortUnrollWindow, ...],
    *,
    example_budget: int,
    batch_size: int,
    seed: int,
):
    generator = torch.Generator().manual_seed(seed)
    pending: list[ShortUnrollWindow] = []
    emitted = 0
    while emitted < example_budget:
        take = min(batch_size, example_budget - emitted)
        while len(pending) < take:
            pending.extend(
                windows[index]
                for index in torch.randperm(len(windows), generator=generator).tolist()
            )
        batch = tuple(pending[:take])
        del pending[:take]
        emitted += take
        yield batch


def train_short_unroll_predictor(
    spec: ShortUnrollTrainingSpec,
    lineages: tuple[CarrierLineage, ...],
    *,
    device: str,
    progress: Callable[[str], None] | None = None,
) -> tuple[DualOutputPredictor, ShortUnrollTrainingReport]:
    """Train one fixed h15 cell under an exact sequence-example budget."""

    windows = build_short_unroll_windows(
        lineages, unroll_steps=spec.unroll_steps
    )
    if next(iter(lineages)).carrier_identity != spec.carrier_identity:
        raise LineageScalingError(
            "training carrier differs from the frozen specification"
        )
    torch.manual_seed(spec.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.seed)
    target_device = torch.device(device)
    model = DualOutputPredictor(spec.predictor_config).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    examples_seen = 0
    steps = 0
    model_evaluations = 0
    totals = {"local": 0.0, "unrolled": 0.0, "bound": 0.0, "objective": 0.0}
    progress_interval = max(spec.batch_size, spec.optimizer_example_budget // 100)
    next_progress = progress_interval
    started = time.monotonic()
    model.train()
    for batch in _scheduled_window_batches(
        windows,
        example_budget=spec.optimizer_example_budget,
        batch_size=spec.batch_size,
        seed=spec.seed + spec.unroll_steps,
    ):
        optimizer.zero_grad(set_to_none=True)
        losses = short_unroll_losses(
            model, batch, carrier_bound=spec.carrier_bound, device=target_device
        )
        objective = (
            spec.local_loss_weight * losses.local
            + spec.unrolled_loss_weight * losses.unrolled
            + spec.carrier_bound_loss_weight * losses.carrier_bound_penalty
        )
        if not bool(torch.isfinite(objective)):
            raise LineageScalingError("short-unroll training produced a nonfinite loss")
        objective.backward()
        if spec.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip)
        optimizer.step()

        take = len(batch)
        examples_seen += take
        steps += 1
        model_evaluations += take * (2 * spec.unroll_steps - 1)
        values = {
            "local": float(losses.local.detach().cpu()),
            "unrolled": float(losses.unrolled.detach().cpu()),
            "bound": float(losses.carrier_bound_penalty.detach().cpu()),
            "objective": float(objective.detach().cpu()),
        }
        for name, value in values.items():
            totals[name] += value * take
        if progress is not None and (
            examples_seen >= next_progress
            or examples_seen == spec.optimizer_example_budget
        ):
            progress(
                f"[train {spec.name}/seed-{spec.seed}] examples "
                f"{examples_seen}/{spec.optimizer_example_budget} "
                f"local={values['local']:.8f} unrolled={values['unrolled']:.8f} "
                f"bound={values['bound']:.8f} objective={values['objective']:.8f}"
            )
            while next_progress <= examples_seen:
                next_progress += progress_interval

    report = ShortUnrollTrainingReport(
        checkpoint_identity=spec.checkpoint_identity,
        spec=spec,
        lineage_count=len(lineages),
        available_window_count=len(windows),
        optimizer_examples=examples_seen,
        optimizer_steps=steps,
        effective_epochs=examples_seen / len(windows),
        mean_local_loss=totals["local"] / examples_seen,
        mean_unrolled_loss=totals["unrolled"] / examples_seen,
        mean_carrier_bound_penalty=totals["bound"] / examples_seen,
        mean_objective=totals["objective"] / examples_seen,
        model_evaluations=model_evaluations,
        wall_seconds=time.monotonic() - started,
        device=str(target_device),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        failures=(),
    )
    return model, report


def _spec_payload(spec: ShortUnrollTrainingSpec) -> dict[str, Any]:
    return asdict(spec)


def _spec_from_payload(raw: Mapping[str, Any]) -> ShortUnrollTrainingSpec:
    payload = dict(raw)
    payload["predictor_config"] = PredictorConfig(**payload["predictor_config"])
    return ShortUnrollTrainingSpec(**payload)


def save_short_unroll_checkpoint(
    path: Path,
    model: DualOutputPredictor,
    report: ShortUnrollTrainingReport,
) -> None:
    """Save compact weights and exact semantic bindings without recursive IDs."""

    target = Path(path)
    if target.exists():
        raise LineageScalingError(f"short-unroll checkpoint already exists: {target}")
    if model.config != report.spec.predictor_config:
        raise LineageScalingError("checkpoint model differs from its training report")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "issue_67_short_unroll_checkpoint_v1",
            "metadata": {
                "checkpoint_identity": report.checkpoint_identity,
                "spec": _spec_payload(report.spec),
                "lineage_count": report.lineage_count,
                "available_window_count": report.available_window_count,
                "optimizer_examples": report.optimizer_examples,
                "optimizer_steps": report.optimizer_steps,
                "effective_epochs": report.effective_epochs,
                "mean_local_loss": report.mean_local_loss,
                "mean_unrolled_loss": report.mean_unrolled_loss,
                "mean_carrier_bound_penalty": report.mean_carrier_bound_penalty,
                "mean_objective": report.mean_objective,
                "model_evaluations": report.model_evaluations,
                "wall_seconds": report.wall_seconds,
                "device": report.device,
                "parameter_count": report.parameter_count,
                "failures": report.failures,
            },
            "model_state": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
        },
        target,
    )


def load_short_unroll_checkpoint(
    path: Path,
    expected_spec: ShortUnrollTrainingSpec,
    *,
    device: str,
) -> tuple[DualOutputPredictor, ShortUnrollTrainingReport]:
    try:
        payload = torch.load(
            Path(path), map_location="cpu", weights_only=True, mmap=True
        )
        if payload["schema"] != "issue_67_short_unroll_checkpoint_v1":
            raise KeyError("schema")
        raw = payload["metadata"]
        spec = _spec_from_payload(raw["spec"])
        report = ShortUnrollTrainingReport(
            checkpoint_identity=raw["checkpoint_identity"],
            spec=spec,
            lineage_count=raw["lineage_count"],
            available_window_count=raw["available_window_count"],
            optimizer_examples=raw["optimizer_examples"],
            optimizer_steps=raw["optimizer_steps"],
            effective_epochs=raw["effective_epochs"],
            mean_local_loss=raw["mean_local_loss"],
            mean_unrolled_loss=raw["mean_unrolled_loss"],
            mean_carrier_bound_penalty=raw["mean_carrier_bound_penalty"],
            mean_objective=raw["mean_objective"],
            model_evaluations=raw["model_evaluations"],
            wall_seconds=raw["wall_seconds"],
            device=raw["device"],
            parameter_count=raw["parameter_count"],
            failures=tuple(raw["failures"]),
        )
        state = payload["model_state"]
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        raise LineageScalingError(
            f"short-unroll checkpoint is invalid: {error}"
        ) from error
    model = DualOutputPredictor(spec.predictor_config)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise LineageScalingError(
            f"short-unroll checkpoint weights are invalid: {error}"
        ) from error
    expected_steps = math.ceil(
        expected_spec.optimizer_example_budget / expected_spec.batch_size
    )
    if (
        spec != expected_spec
        or report.checkpoint_identity != expected_spec.checkpoint_identity
        or report.optimizer_examples != expected_spec.optimizer_example_budget
        or report.optimizer_steps != expected_steps
        or report.lineage_count <= 0
        or report.available_window_count <= 0
        or report.effective_epochs
        != report.optimizer_examples / report.available_window_count
        or report.model_evaluations
        != report.optimizer_examples * (2 * spec.unroll_steps - 1)
        or any(
            not math.isfinite(value)
            for value in (
                report.mean_local_loss,
                report.mean_unrolled_loss,
                report.mean_carrier_bound_penalty,
                report.mean_objective,
                report.wall_seconds,
            )
        )
        or report.wall_seconds < 0.0
        or report.parameter_count
        != sum(parameter.numel() for parameter in model.parameters())
        or report.failures
    ):
        raise LineageScalingError(
            "short-unroll checkpoint differs from its frozen training cell"
        )
    return model.to(torch.device(device)), report


@dataclass(frozen=True, slots=True)
class RecursiveStepMetric:
    step: int
    mean_mse: float | None
    mean_absolute_carrier_bound_excess: float | None
    evaluated_predictions: int


@dataclass(frozen=True, slots=True)
class RecursiveCarrierEvaluation:
    horizon: int
    exposure_role: str
    lineage_count: int
    error_auc: float | None
    mean_absolute_carrier_bound_excess: float | None
    step_curve: tuple[RecursiveStepMetric, ...]
    nonfinite_failures: int
    execution_failures: tuple[str, ...]
    model_evaluations: int
    wall_seconds: float


def evaluate_recursive_carrier(
    model: torch.nn.Module,
    lineages: tuple[CarrierLineage, ...],
    *,
    horizon: int,
    carrier_bound: float,
    progress: Callable[[str], None] | None = None,
) -> RecursiveCarrierEvaluation:
    """Evaluate unclamped recursive predictions and retain every step's error."""

    roles = {lineage.exposure_role for lineage in lineages}
    if (
        not lineages
        or horizon not in (1, H15)
        or len(roles) != 1
        or not roles <= {"training", "calibration", "model_selection"}
        or {lineage.carrier for lineage in lineages} != {CarrierKind.DEPLOYMENT}
        or not math.isfinite(carrier_bound)
        or carrier_bound <= 0.0
    ):
        raise LineageScalingError("recursive carrier evaluation inputs are invalid")
    role = next(iter(roles))
    device = next(model.parameters()).device
    errors_by_step: dict[int, list[float]] = {}
    bounds_by_step: dict[int, list[float]] = {}
    lineage_aucs: list[float] = []
    failures: list[str] = []
    nonfinite = 0
    model_evaluations = 0
    started = time.monotonic()
    model.eval()
    with torch.no_grad():
        for lineage_ordinal, lineage in enumerate(lineages, start=1):
            windows = {
                transition.decision_index: transition
                for transition in lineage.transitions
                if transition.horizon == horizon
            }
            starts = (0, *lineage.segment_ends[:-1])
            lineage_area = 0.0
            lineage_evaluated = False
            for segment_ordinal, (segment_start, segment_end) in enumerate(
                zip(starts, lineage.segment_ends, strict=True), start=1
            ):
                position = segment_start
                current = windows.get(position)
                if current is None:
                    raise LineageScalingError(
                        f"complete lineage lacks a starting h{horizon} window"
                    )
                carrier = current.context.to(device)
                previous_position = segment_start
                previous_error = 0.0
                recursive_step = 0
                while position < segment_end:
                    transition = windows.get(position)
                    if (
                        transition is None
                        or transition.target_decision_index
                        != min(position + horizon, segment_end)
                    ):
                        raise LineageScalingError(
                            f"complete lineage lacks contiguous h{horizon} windows"
                        )
                    recursive_step += 1
                    try:
                        model_evaluations += 1
                        predicted = model.carrier(
                            carrier.unsqueeze(0),
                            transition.action.to(device).unsqueeze(0),
                            PredictionPair(horizon, Abstraction.CONTINUOUS),
                        )[0]
                        if not bool(torch.isfinite(predicted).all()):
                            nonfinite += 1
                            break
                        error = float(F.mse_loss(
                            predicted, transition.target.to(device)
                        ).cpu())
                        bound_excess = float(torch.relu(
                            predicted.abs() - carrier_bound
                        ).mean().cpu())
                        errors_by_step.setdefault(recursive_step, []).append(error)
                        bounds_by_step.setdefault(recursive_step, []).append(
                            bound_excess
                        )
                        target_position = transition.target_decision_index
                        lineage_area += (
                            (previous_error + error)
                            * (target_position - previous_position)
                            / 2.0
                        )
                        previous_position = target_position
                        previous_error = error
                        position = target_position
                        carrier = predicted
                        lineage_evaluated = True
                    except Exception as error:
                        failures.append(
                            f"lineage-{lineage_ordinal}/segment-{segment_ordinal}/"
                            f"step-{recursive_step}:{type(error).__name__}: {error}"
                        )
                        break
            if lineage_evaluated:
                lineage_aucs.append(lineage_area)
            if progress is not None:
                progress(
                    f"[score recursive-h{horizon}] lineage "
                    f"{lineage_ordinal}/{len(lineages)}"
                )
    steps = tuple(
        RecursiveStepMetric(
            step=step,
            mean_mse=sum(errors_by_step[step]) / len(errors_by_step[step]),
            mean_absolute_carrier_bound_excess=(
                sum(bounds_by_step[step]) / len(bounds_by_step[step])
            ),
            evaluated_predictions=len(errors_by_step[step]),
        )
        for step in sorted(errors_by_step)
    )
    all_bounds = [value for values in bounds_by_step.values() for value in values]
    return RecursiveCarrierEvaluation(
        horizon=horizon,
        exposure_role=role,
        lineage_count=len(lineages),
        error_auc=(None if not lineage_aucs else sum(lineage_aucs) / len(lineage_aucs)),
        mean_absolute_carrier_bound_excess=(
            None if not all_bounds else sum(all_bounds) / len(all_bounds)
        ),
        step_curve=steps,
        nonfinite_failures=nonfinite,
        execution_failures=tuple(failures),
        model_evaluations=model_evaluations,
        wall_seconds=time.monotonic() - started,
    )
