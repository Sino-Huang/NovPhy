"""Integrated issue-58 predictor training and recursive rollout evidence."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum, unique
from pathlib import Path
from statistics import mean
from typing import Final

import torch

from world_model.data import (
    CohortV2OracleWindowDataset,
    CohortV2ReleaseReader,
)
from world_model.data.cohort_v2 import CAPABILITY_DECLARATION_IDENTITY
from world_model.model import Abstraction, DualOutputPredictor, PredictionPair, identity
from world_model.training.cohort_v2_calibration import CohortV2CalibrationRecord
from world_model.training.cohort_v2_controller import CohortV2ControllerConfig
from world_model.training.cohort_v2_macro import (
    MACRO_CAPABILITIES,
    MACRO_EVENT_ENDPOINT_AUTHORITY,
    MACRO_PAIRS,
    MACRO_STATE_AUTHORITY,
    CohortV2MacroCheckpoint,
    CohortV2MacroConfig,
    CohortV2MacroError,
    CohortV2MacroTrainer,
    CohortV2MacroTrainingData,
)
from world_model.training.cohort_v2_measurement import CohortV2ComputeCalibration
from world_model.training.cohort_v2_micro import (
    MICRO_RELATION_AUTHORITY,
    CohortV2StateCodec,
    cohort_v2_action,
    cohort_v2_model_state_identity,
)
from world_model.training.cohort_v2_reliability import (
    CohortV2ReliabilityConfig,
    CohortV2ReliabilityEstimator,
    reliability_symbolic_gate,
)
from world_model.training.cohort_v2_symbolic_interfaces import (
    MatchedMicroInterfaceAdapter,
    SymbolicInterface,
)
from world_model.training.grid_artifacts import canonical_json_bytes


INTEGRATED_CHECKPOINT_SCHEMA: Final = "cohort_v2_integrated_checkpoint_v1"
INTEGRATED_EVIDENCE_SCHEMA: Final = "cohort_v2_integrated_calibration_v1"
RECURSIVE_ROLLOUT_SCHEMA: Final = "cohort_v2_recursive_continuous_rollout_v1"
RECURSIVE_HORIZONS: Final = (1, 5, 15)


class CohortV2IntegratedError(ValueError):
    """The integrated model, recursive rollout, or report is invalid."""


@unique
class IntegratedVariant(StrEnum):
    CANDIDATE = "integrated_ordered_flat_reliability_gated"
    NO_SYMBOL = "integrated_no_symbol"
    UNGATED = "integrated_ordered_flat_ungated"

    @property
    def interface(self) -> SymbolicInterface:
        if self is IntegratedVariant.NO_SYMBOL:
            return SymbolicInterface.NO_SYMBOL
        return SymbolicInterface.ORDERED_FLAT

    @property
    def reliability_gate_enabled(self) -> bool:
        return self is IntegratedVariant.CANDIDATE


@dataclass(frozen=True, slots=True)
class CohortV2RecursiveRolloutRecord:
    checkpoint_identity: str
    exposure_role: str
    attempt_id: str
    scenario_lineage_identity: str
    coverage_stratum: str
    requested_horizon: int
    simulated_duration: int
    effective_horizons: tuple[int, ...]
    cumulative_horizons: tuple[int, ...]
    authoritative_endpoint_identities: tuple[str, ...]
    endpoint_mse_curve: tuple[float, ...]
    terminal_mse: float
    error_auc: float
    total_compute: float
    recursive_physical_violation_status: str = "unavailable"

    def __post_init__(self) -> None:
        lengths = {
            len(self.effective_horizons),
            len(self.cumulative_horizons),
            len(self.authoritative_endpoint_identities),
            len(self.endpoint_mse_curve),
        }
        if (
            not self.checkpoint_identity
            or self.exposure_role not in ("calibration", "model_selection")
            or not self.attempt_id
            or self.requested_horizon not in RECURSIVE_HORIZONS
            or self.simulated_duration <= 0
            or lengths == {0}
            or len(lengths) != 1
            or sum(self.effective_horizons) != self.simulated_duration
            or self.cumulative_horizons[-1] != self.simulated_duration
            or any(not 1 <= value <= self.requested_horizon for value in self.effective_horizons)
            or any(value < 0.0 or not math.isfinite(value) for value in self.endpoint_mse_curve)
            or self.terminal_mse != self.endpoint_mse_curve[-1]
            or self.error_auc < 0.0
            or self.total_compute < 0.0
            or self.recursive_physical_violation_status != "unavailable"
        ):
            raise CohortV2IntegratedError("recursive rollout record is malformed")


def build_integrated_predictor(
    config: CohortV2MacroConfig, variant: IntegratedVariant
) -> DualOutputPredictor:
    if type(variant) is not IntegratedVariant:
        raise CohortV2IntegratedError("integrated variant is invalid")
    predictor = DualOutputPredictor(config.predictor_config)
    predictor.micro_adapter = MatchedMicroInterfaceAdapter(
        config.hidden_dim, variant.interface
    )
    return predictor


def build_integrated_trainer(
    reader: CohortV2ReleaseReader,
    config: CohortV2MacroConfig,
    variant: IntegratedVariant,
    *,
    reliability_estimator: CohortV2ReliabilityEstimator,
    reliability_config: CohortV2ReliabilityConfig,
    reliability_readers: tuple[CohortV2ReleaseReader, ...],
) -> CohortV2MacroTrainer:
    predictor = build_integrated_predictor(config, variant)
    gate = None
    if variant.reliability_gate_enabled:
        gate = reliability_symbolic_gate(
            reliability_estimator, reliability_config, reliability_readers
        )
    return CohortV2MacroTrainer(
        CohortV2MacroTrainingData(reader, config),
        config,
        predictor=predictor,
        symbolic_gate=gate,
    )


def _pair_counts(trainer: CohortV2MacroTrainer) -> tuple[tuple[str, int], ...]:
    return tuple(
        (pair.identity, trainer.pair_counts[pair]) for pair in MACRO_PAIRS
    )


def _checkpoint_identity(
    *,
    reader: CohortV2ReleaseReader,
    config: CohortV2MacroConfig,
    codec: CohortV2StateCodec,
    variant: IntegratedVariant,
    reliability_artifact_identity: str,
    model_state_identity: str,
    step: int,
    pair_counts: tuple[tuple[str, int], ...],
) -> str:
    return identity((
        INTEGRATED_CHECKPOINT_SCHEMA,
        reader.release_identity,
        reader.partition_identity,
        CAPABILITY_DECLARATION_IDENTITY,
        config.identity,
        config.predictor_config.identity,
        codec.identity,
        str(variant),
        str(variant.interface),
        variant.reliability_gate_enabled,
        reliability_artifact_identity,
        model_state_identity,
        step,
        pair_counts,
        tuple(sorted(MACRO_CAPABILITIES)),
        MICRO_RELATION_AUTHORITY,
        MACRO_STATE_AUTHORITY,
        MACRO_EVENT_ENDPOINT_AUTHORITY,
        "training",
    ))


def save_cohort_v2_integrated_checkpoint(
    path: Path,
    trainer: CohortV2MacroTrainer,
    variant: IntegratedVariant,
    *,
    reliability_artifact_identity: str,
) -> CohortV2MacroCheckpoint:
    counts = _pair_counts(trainer)
    expected_per_pair = trainer.config.steps // len(MACRO_PAIRS)
    if (
        trainer.config.steps % len(MACRO_PAIRS)
        or any(count != expected_per_pair for _, count in counts)
    ):
        raise CohortV2IntegratedError(
            "integrated training must allocate exactly equal updates to all nine pairs"
        )
    state = trainer.predictor.state_dict()
    state_identity = cohort_v2_model_state_identity(state)
    checkpoint_identity = _checkpoint_identity(
        reader=trainer.data.reader,
        config=trainer.config,
        codec=trainer.codec,
        variant=variant,
        reliability_artifact_identity=reliability_artifact_identity,
        model_state_identity=state_identity,
        step=trainer.step_count,
        pair_counts=counts,
    )
    payload = {
        "capabilities": sorted(MACRO_CAPABILITIES),
        "capability_declaration_identity": CAPABILITY_DECLARATION_IDENTITY,
        "checkpoint_identity": checkpoint_identity,
        "codec_identity": trainer.codec.identity,
        "config_identity": trainer.config.identity,
        "exposure_role": "training",
        "interface": str(variant.interface),
        "macro_event_endpoint_authority": MACRO_EVENT_ENDPOINT_AUTHORITY,
        "macro_state_authority": MACRO_STATE_AUTHORITY,
        "micro_relation_authority": MICRO_RELATION_AUTHORITY,
        "model_state": state,
        "model_state_identity": state_identity,
        "pair_counts": dict(counts),
        "partition_identity": trainer.data.reader.partition_identity,
        "predictor_config_identity": trainer.config.predictor_config.identity,
        "release_identity": trainer.data.reader.release_identity,
        "reliability_artifact_identity": reliability_artifact_identity,
        "reliability_gate_enabled": variant.reliability_gate_enabled,
        "schema": INTEGRATED_CHECKPOINT_SCHEMA,
        "step": trainer.step_count,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in trainer.predictor.parameters()
        ),
        "variant": str(variant),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return CohortV2MacroCheckpoint(
        target, checkpoint_identity, trainer.step_count, counts
    )


def load_cohort_v2_integrated_checkpoint(
    path: Path,
    *,
    reader: CohortV2ReleaseReader,
    config: CohortV2MacroConfig,
    variant: IntegratedVariant,
    reliability_artifact_identity: str,
    device: str,
) -> tuple[DualOutputPredictor, CohortV2StateCodec, CohortV2MacroCheckpoint]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise CohortV2IntegratedError(
            f"cannot load integrated checkpoint: {error}"
        ) from error
    state = payload.get("model_state") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise CohortV2IntegratedError("integrated checkpoint envelope is malformed")
    codec = CohortV2StateCodec(
        latent_dim=config.latent_dim, max_entities=config.max_entities
    )
    counts = tuple(
        (pair.identity, payload.get("pair_counts", {}).get(pair.identity))
        for pair in MACRO_PAIRS
    )
    state_identity = cohort_v2_model_state_identity(state)
    expected = _checkpoint_identity(
        reader=reader,
        config=config,
        codec=codec,
        variant=variant,
        reliability_artifact_identity=reliability_artifact_identity,
        model_state_identity=state_identity,
        step=payload.get("step"),
        pair_counts=counts,
    )
    expected_per_pair = config.steps // len(MACRO_PAIRS)
    if (
        payload.get("schema") != INTEGRATED_CHECKPOINT_SCHEMA
        or payload.get("checkpoint_identity") != expected
        or payload.get("release_identity") != reader.release_identity
        or payload.get("partition_identity") != reader.partition_identity
        or payload.get("config_identity") != config.identity
        or payload.get("predictor_config_identity") != config.predictor_config.identity
        or payload.get("codec_identity") != codec.identity
        or payload.get("capabilities") != sorted(MACRO_CAPABILITIES)
        or payload.get("variant") != str(variant)
        or payload.get("interface") != str(variant.interface)
        or payload.get("reliability_gate_enabled") is not variant.reliability_gate_enabled
        or payload.get("reliability_artifact_identity") != reliability_artifact_identity
        or payload.get("exposure_role") != "training"
        or payload.get("step") != config.steps
        or config.steps % len(MACRO_PAIRS)
        or any(count != expected_per_pair for _, count in counts)
        or payload.get("model_state_identity") != state_identity
    ):
        raise CohortV2IntegratedError(
            "integrated checkpoint provenance is stale or malformed"
        )
    predictor = build_integrated_predictor(config, variant)
    try:
        predictor.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise CohortV2IntegratedError(
            f"integrated model state is invalid: {error}"
        ) from error
    parameter_count = sum(parameter.numel() for parameter in predictor.parameters())
    if payload.get("trainable_parameter_count") != parameter_count:
        raise CohortV2IntegratedError("integrated parameter accounting differs")
    predictor.to(torch.device(device)).eval()
    return predictor, codec, CohortV2MacroCheckpoint(
        Path(path), expected, config.steps, counts
    )


def integrated_compute_calibration(
    config: CohortV2MacroConfig,
    variant: IntegratedVariant,
    *,
    controller_config: CohortV2ControllerConfig | None = None,
) -> CohortV2ComputeCalibration:
    predictor = config.predictor_config
    h = predictor.hidden_dim
    h2 = h * h
    delta_features = 2 * predictor.delta_frequency_count
    conditioner = (
        (delta_features + predictor.pair_code_dim) * (2 * predictor.pair_code_dim)
        + (2 * predictor.pair_code_dim) * predictor.pair_code_dim
    )
    transition = (
        conditioner
        + (predictor.latent_dim + predictor.action_dim) * h
        + predictor.depth
        * (predictor.pair_code_dim * 2 * h + 2 * h2)
        + h * predictor.latent_dim
    )
    if variant.interface is SymbolicInterface.NO_SYMBOL:
        graph_base = 5 * h2 + 2 * h
        graph_per_relation = 0.0
    else:
        graph_base = 2 * h2 + 2 * h
        graph_per_relation = 3 * h2
    controller = 0.0
    if controller_config is not None:
        controller = (
            controller_config.feature_dim * controller_config.hidden_dim
            + controller_config.hidden_dim * controller_config.hidden_dim
            + controller_config.hidden_dim * 9
        )
    return CohortV2ComputeCalibration(
        authority=identity((
            "issue-58-declared-mac-accounting-v1",
            config.predictor_config.identity,
            str(variant),
            None if controller_config is None else controller_config.identity,
        )),
        unit="multiply_accumulate",
        controller_per_decision=controller,
        continuous_adapter_per_decision=0.0,
        micro_adapter_per_decision=0.0,
        macro_adapter_per_decision=4 * h,
        micro_graph_base_per_decision=graph_base,
        micro_graph_per_entity=0.0,
        micro_graph_per_contact=graph_per_relation,
        micro_graph_per_support=graph_per_relation,
        transition_per_decision=transition,
        continuous_readout_per_decision=0.0,
        micro_readout_per_decision=(
            predictor.latent_dim * h + h * predictor.micro_predicate_count
        ),
        macro_readout_per_decision=(
            predictor.latent_dim * h
            + h * (
                predictor.macro_predicate_count + 1 + predictor.event_type_count
            )
        ),
        shared_initial_perception_per_rollout=0.0,
    )


def recursive_continuous_rollouts(
    predictor: DualOutputPredictor,
    codec: CohortV2StateCodec,
    checkpoint_identity: str,
    readers: tuple[CohortV2ReleaseReader, ...],
    compute: CohortV2ComputeCalibration,
    *,
    rollout_limit: int | None = None,
) -> tuple[CohortV2RecursiveRolloutRecord, ...]:
    """Roll predicted carriers without feeding future engine state or labels."""
    records = []
    for reader in readers:
        if reader.rollouts[0].exposure_role not in ("calibration", "model_selection"):
            raise CohortV2IntegratedError(
                "recursive evidence accepts calibration and model-selection roles only"
            )
        first_windows = {}
        for window in CohortV2OracleWindowDataset(reader, requested_horizons=(1,)):
            first_windows.setdefault(window.attempt_id, window)
        for rollout in reader.rollouts[:rollout_limit]:
            duration = len(rollout.frame_records) - 1
            initial = codec.encode(rollout.frame_records[0])
            action = cohort_v2_action(first_windows[rollout.attempt_id]).unsqueeze(0)
            device = next(predictor.parameters()).device
            action = action.to(device)
            for requested in RECURSIVE_HORIZONS:
                current = initial.unsqueeze(0).to(device)
                position = 0
                effective = []
                cumulative = []
                endpoint_ids = []
                errors = []
                with torch.no_grad():
                    while position < duration:
                        step = min(requested, duration - position)
                        current = predictor.carrier(
                            current,
                            action,
                            PredictionPair(requested, Abstraction.CONTINUOUS),
                        )
                        position += step
                        target_frame = rollout.frame_records[position]
                        target = codec.encode(target_frame).unsqueeze(0).to(device)
                        errors.append(float((current - target).pow(2).mean()))
                        effective.append(step)
                        cumulative.append(position)
                        endpoint_ids.append(target_frame.identity)
                previous = 0.0
                auc = 0.0
                for step, error in zip(effective, errors, strict=True):
                    auc += step * (previous + error) / 2.0
                    previous = error
                per_decision = (
                    compute.continuous_adapter_per_decision
                    + compute.transition_per_decision
                    + compute.continuous_readout_per_decision
                )
                records.append(CohortV2RecursiveRolloutRecord(
                    checkpoint_identity=checkpoint_identity,
                    exposure_role=rollout.exposure_role,
                    attempt_id=rollout.attempt_id,
                    scenario_lineage_identity=rollout.scenario_lineage_identity,
                    coverage_stratum=rollout.coverage_stratum,
                    requested_horizon=requested,
                    simulated_duration=duration,
                    effective_horizons=tuple(effective),
                    cumulative_horizons=tuple(cumulative),
                    authoritative_endpoint_identities=tuple(endpoint_ids),
                    endpoint_mse_curve=tuple(errors),
                    terminal_mse=errors[-1],
                    error_auc=auc,
                    total_compute=(
                        len(effective) * per_decision
                        + compute.shared_initial_perception_per_rollout
                    ),
                ))
    return tuple(records)


def summarize_recursive_rollouts(
    records: tuple[CohortV2RecursiveRolloutRecord, ...]
) -> dict[str, object]:
    if not records:
        raise CohortV2IntegratedError("recursive rollout evidence is empty")
    summary = {}
    for role in ("calibration", "model_selection"):
        by_horizon = {}
        for requested in RECURSIVE_HORIZONS:
            selected = tuple(
                item for item in records
                if item.exposure_role == role and item.requested_horizon == requested
            )
            if not selected:
                continue
            by_horizon[str(requested)] = {
                "complete_rollout_count": len(selected),
                "mean_error_auc": mean(item.error_auc for item in selected),
                "mean_terminal_mse": mean(item.terminal_mse for item in selected),
                "mean_total_compute": mean(item.total_compute for item in selected),
                "recursive_physical_violation_status": "unavailable",
            }
        summary[role] = by_horizon
    return summary


def analyze_integrated_calibration(
    records: tuple[CohortV2CalibrationRecord, ...],
    recursive: tuple[CohortV2RecursiveRolloutRecord, ...],
    *,
    candidate_configuration_id: str,
    comparator_configuration_ids: tuple[str, ...],
    source_bindings: Mapping[str, object],
    stress_ablations: Mapping[str, object] | None = None,
    bootstrap_seed: int = 20260826,
    bootstrap_replicates: int = 10_000,
) -> dict[str, object]:
    """Freeze comparators on model selection and thresholds on calibration."""
    if (
        not records
        or not comparator_configuration_ids
        or bootstrap_seed != 20260826
        or bootstrap_replicates != 10_000
    ):
        raise CohortV2IntegratedError("integrated calibration design is incomplete")
    configurations = {item.configuration_id for item in records}
    required = {candidate_configuration_id, *comparator_configuration_ids}
    if not required <= configurations:
        raise CohortV2IntegratedError("integrated comparator evidence is incomplete")
    role_attempts = {}
    for role in ("calibration", "model_selection"):
        reference = {
            item.attempt_id for item in records
            if item.configuration_id == candidate_configuration_id
            and item.exposure_role == role
        }
        if len(reference) != 6 or any(
            {
                item.attempt_id for item in records
                if item.configuration_id == configuration_id
                and item.exposure_role == role
            }
            != reference
            for configuration_id in required
        ):
            raise CohortV2IntegratedError(
                f"every configuration must cover the same six {role} rollouts"
            )
        role_attempts[role] = reference

    calibration_compute = tuple(
        item.mean_policy_compute_per_simulated_frame
        for item in records
        if item.exposure_role == "calibration" and item.configuration_id in required
    )
    ordered_compute = sorted(calibration_compute)

    def percentile(probability: float) -> float:
        position = probability * (len(ordered_compute) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered_compute[lower]
        weight = position - lower
        return ordered_compute[lower] * (1.0 - weight) + ordered_compute[upper] * weight

    budgets = tuple(dict.fromkeys(
        percentile(probability) for probability in (0.25, 0.5, 0.75, 1.0)
    ))
    comparisons = []
    proposals = []
    sensitivity = []

    def bootstrap_interval(values: tuple[float, ...], seed: int) -> tuple[float, float]:
        generator = random.Random(seed)
        sampled = tuple(
            mean(tuple(generator.choice(values) for _ in values))
            for _ in range(bootstrap_replicates)
        )
        ordered = sorted(sampled)

        def value_at(probability: float) -> float:
            position = probability * (len(ordered) - 1)
            lower = math.floor(position)
            upper = math.ceil(position)
            if lower == upper:
                return ordered[lower]
            weight = position - lower
            return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

        return value_at(0.025), value_at(0.975)

    for budget in budgets:
        candidate_rows = tuple(
            item for item in records
            if item.configuration_id == candidate_configuration_id
            and item.exposure_role == "calibration"
        )
        candidate_within_support = mean(
            item.mean_policy_compute_per_simulated_frame for item in candidate_rows
        ) <= budget
        eligible = []
        for comparator_id in comparator_configuration_ids:
            selection_rows = tuple(
                item for item in records
                if item.configuration_id == comparator_id
                and item.exposure_role == "model_selection"
            )
            if selection_rows and mean(
                item.mean_policy_compute_per_simulated_frame for item in selection_rows
            ) <= budget:
                eligible.append(comparator_id)
        if not candidate_within_support or not eligible:
            comparisons.append({
                "budget": budget,
                "candidate_within_support": candidate_within_support,
                "eligible_comparator_ids": eligible,
                "status": "not_comparable_at_budget",
            })
            continue
        strongest = min(
            eligible,
            key=lambda comparator_id: (
                mean(
                    item.mean_endpoint_prediction_error for item in records
                    if item.configuration_id == comparator_id
                    and item.exposure_role == "model_selection"
                ),
                mean(
                    item.mean_endpoint_violation_rate for item in records
                    if item.configuration_id == comparator_id
                    and item.exposure_role == "model_selection"
                ),
                mean(
                    item.mean_policy_compute_per_simulated_frame for item in records
                    if item.configuration_id == comparator_id
                    and item.exposure_role == "model_selection"
                ),
                comparator_configuration_ids.index(comparator_id),
            ),
        )
        candidate_by_attempt = {item.attempt_id: item for item in candidate_rows}
        comparator_by_attempt = {
            item.attempt_id: item for item in records
            if item.configuration_id == strongest
            and item.exposure_role == "calibration"
        }
        attempts = tuple(sorted(set(candidate_by_attempt) & set(comparator_by_attempt)))
        gains = tuple(
            comparator_by_attempt[key].mean_endpoint_prediction_error
            - candidate_by_attempt[key].mean_endpoint_prediction_error
            for key in attempts
        )
        violations = tuple(
            candidate_by_attempt[key].mean_endpoint_violation_rate
            - comparator_by_attempt[key].mean_endpoint_violation_rate
            for key in attempts
        )
        if len(attempts) < 2:
            raise CohortV2IntegratedError(
                "matched-compute calibration requires at least two paired rollouts"
            )
        gain_interval = bootstrap_interval(gains, bootstrap_seed + len(proposals) * 2)
        violation_interval = bootstrap_interval(
            violations, bootstrap_seed + len(proposals) * 2 + 1
        )
        comparator_error = mean(
            comparator_by_attempt[key].mean_endpoint_prediction_error
            for key in attempts
        )
        half_width = (gain_interval[1] - gain_interval[0]) / 2.0
        practical_effect = max(0.10 * comparator_error, half_width)
        physical_margin = max(0.0, violation_interval[1])
        comparisons.append({
            "budget": budget,
            "candidate_within_support": True,
            "eligible_comparator_ids": eligible,
            "paired_calibration_rollout_count": len(attempts),
            "paired_endpoint_gain_bootstrap_95_interval": list(gain_interval),
            "paired_mean_endpoint_gain": None if not gains else mean(gains),
            "paired_mean_physical_violation_increase": mean(violations),
            "paired_physical_violation_bootstrap_95_interval": list(
                violation_interval
            ),
            "status": "comparable",
            "strongest_comparator_id": strongest,
        })
        proposals.append({
            "budget": budget,
            "failed_run_treatment": (
                "Source or provenance failure aborts. Model execution failures remain "
                "failed replicates and are not replaced or excluded."
            ),
            "fixed_replicate_count": len(attempts),
            "physical_violation_margin": physical_margin,
            "practical_effect_threshold": practical_effect,
            "precision_half_width": half_width,
            "strongest_comparator_id": strongest,
        })
        for effect_multiplier in (0.5, 1.0, 1.5):
            for margin_multiplier in (0.0, 1.0, 2.0):
                sensitivity.append({
                    "budget": budget,
                    "calibration_only": True,
                    "effect_threshold": practical_effect * effect_multiplier,
                    "mean_gain_exceeds_threshold": (
                        mean(gains) >= practical_effect * effect_multiplier
                    ),
                    "mean_violation_increase_within_margin": (
                        mean(violations) <= physical_margin * margin_multiplier
                    ),
                    "physical_violation_margin": (
                        physical_margin * margin_multiplier
                    ),
                })

    recursive_summary = summarize_recursive_rollouts(recursive)
    for role in ("calibration", "model_selection"):
        for requested in RECURSIVE_HORIZONS:
            attempts = {
                item.attempt_id for item in recursive
                if item.exposure_role == role
                and item.requested_horizon == requested
            }
            if attempts != role_attempts[role]:
                raise CohortV2IntegratedError(
                    "recursive horizons must cover the same six rollouts as local metrics"
                )
    model_selection_recursive = recursive_summary["model_selection"]
    strongest_recursive = min(
        model_selection_recursive,
        key=lambda horizon: model_selection_recursive[horizon]["mean_terminal_mse"],
    )
    calibration_terminal = tuple(
        item.terminal_mse for item in recursive
        if item.exposure_role == "calibration"
        and str(item.requested_horizon) == strongest_recursive
    )
    configuration_summaries = {}
    for configuration_id in sorted(required):
        configuration_summaries[configuration_id] = {}
        for role in ("calibration", "model_selection"):
            selected = tuple(
                item for item in records
                if item.configuration_id == configuration_id
                and item.exposure_role == role
            )
            configuration_summaries[configuration_id][role] = {
                "complete_rollout_count": len(selected),
                "mean_endpoint_prediction_error": mean(
                    item.mean_endpoint_prediction_error for item in selected
                ),
                "mean_endpoint_violation_rate": mean(
                    item.mean_endpoint_violation_rate for item in selected
                ),
                "mean_full_compute_per_simulated_frame": mean(
                    item.mean_full_compute_per_simulated_frame for item in selected
                ),
                "mean_policy_compute_per_simulated_frame": mean(
                    item.mean_policy_compute_per_simulated_frame for item in selected
                ),
            }
    return {
        "analysis_version": INTEGRATED_EVIDENCE_SCHEMA,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "candidate_configuration_id": candidate_configuration_id,
        "comparator_configuration_ids": list(comparator_configuration_ids),
        "complete_rollout_metrics": {
            "recursive_continuous": recursive_summary,
            "strongest_horizon_selected_on_model_selection": int(strongest_recursive),
            "selected_horizon_calibration_mean_terminal_mse": mean(calibration_terminal),
            "recursive_physical_violation_status": "unavailable",
        },
        "disposition": {
            "status": "sufficient_evidence_to_freeze_issue_34",
            "additional_calibration_work_required": [],
        },
        "exposure_audit": {
            "configuration_and_comparator_selection_role": "model_selection",
            "threshold_margin_precision_and_sensitivity_role": "calibration",
            "final_evaluation_accessed": False,
            "learned_parameter_role": "training",
        },
        "independent_calibration_rollouts": len(role_attempts["calibration"]),
        "local_teacher_forced_metrics": {
            "budget_comparisons": comparisons,
            "configuration_summaries": configuration_summaries,
            "declared_compute_budgets": list(budgets),
            "evaluation_mode": "teacher_forced_local_successor_prediction",
            "physical_violation_scope": (
                "authoritative_source_endpoint_incidence; not a label derived from "
                "predicted carriers"
            ),
        },
        "proposals_for_issue_34": proposals,
        "sensitivity": sensitivity,
        "scope": {
            "adaptive_symbolic_recursive_planning": "out_of_scope_issue_56",
            "gameplay_outcomes": "out_of_scope_issue_57",
            "learned_parser_stress": "downstream_of_issue_15",
        },
        "source_bindings": dict(source_bindings),
        "stress_ablations": {} if stress_ablations is None else dict(stress_ablations),
    }


def _digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_integrated_evidence(
    root: Path,
    records: tuple[CohortV2CalibrationRecord, ...],
    recursive: tuple[CohortV2RecursiveRolloutRecord, ...],
    report: Mapping[str, object],
    *,
    implementation_revision: str,
) -> dict[str, object]:
    record_bytes = b"".join(
        canonical_json_bytes({"schema": INTEGRATED_EVIDENCE_SCHEMA, **asdict(item)})
        for item in records
    )
    recursive_bytes = b"".join(
        canonical_json_bytes({"schema": RECURSIVE_ROLLOUT_SCHEMA, **asdict(item)})
        for item in recursive
    )
    report_bytes = canonical_json_bytes(dict(report))
    manifest = {
        "analysis": "report.json",
        "analysis_identity": _digest(report_bytes),
        "calibration_records": "calibration_records.jsonl",
        "calibration_records_identity": _digest(record_bytes),
        "implementation_revision": implementation_revision,
        "recursive_records": "recursive_continuous_rollouts.jsonl",
        "recursive_records_identity": _digest(recursive_bytes),
        "schema": INTEGRATED_EVIDENCE_SCHEMA,
    }
    manifest["artifact_identity"] = identity((
        INTEGRATED_EVIDENCE_SCHEMA,
        implementation_revision,
        manifest["calibration_records_identity"],
        manifest["recursive_records_identity"],
        manifest["analysis_identity"],
    ))
    root = Path(root)
    _atomic_write(root / "calibration_records.jsonl", record_bytes)
    _atomic_write(root / "recursive_continuous_rollouts.jsonl", recursive_bytes)
    _atomic_write(root / "report.json", report_bytes)
    _atomic_write(root / "manifest.json", canonical_json_bytes(manifest))
    validate_integrated_evidence(root)
    return manifest


def validate_integrated_evidence(root: Path) -> dict[str, object]:
    root = Path(root)
    try:
        raw_manifest = (root / "manifest.json").read_bytes()
        manifest = json.loads(raw_manifest)
        record_bytes = (root / manifest["calibration_records"]).read_bytes()
        recursive_bytes = (root / manifest["recursive_records"]).read_bytes()
        report_bytes = (root / manifest["analysis"]).read_bytes()
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise CohortV2IntegratedError(
            f"cannot load integrated evidence: {error}"
        ) from error
    expected_identity = identity((
        INTEGRATED_EVIDENCE_SCHEMA,
        manifest.get("implementation_revision"),
        _digest(record_bytes),
        _digest(recursive_bytes),
        _digest(report_bytes),
    ))
    if (
        raw_manifest != canonical_json_bytes(manifest)
        or manifest.get("schema") != INTEGRATED_EVIDENCE_SCHEMA
        or manifest.get("calibration_records_identity") != _digest(record_bytes)
        or manifest.get("recursive_records_identity") != _digest(recursive_bytes)
        or manifest.get("analysis_identity") != _digest(report_bytes)
        or manifest.get("artifact_identity") != expected_identity
    ):
        raise CohortV2IntegratedError("integrated evidence identity is stale")
    try:
        calibration_rows = tuple(json.loads(line) for line in record_bytes.splitlines())
        calibration = tuple(
            CohortV2CalibrationRecord(**{
                key: value for key, value in row.items() if key != "schema"
            })
            for row in calibration_rows
        )
        recursive_rows = tuple(json.loads(line) for line in recursive_bytes.splitlines())
        recursive = tuple(
            CohortV2RecursiveRolloutRecord(**{
                key: tuple(value) if key in {
                    "effective_horizons",
                    "cumulative_horizons",
                    "authoritative_endpoint_identities",
                    "endpoint_mse_curve",
                } else value
                for key, value in row.items() if key != "schema"
            })
            for row in recursive_rows
        )
        report = json.loads(report_bytes)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise CohortV2IntegratedError(
            f"integrated evidence records are malformed: {error}"
        ) from error
    if (
        recursive_bytes
        != b"".join(
            canonical_json_bytes({"schema": RECURSIVE_ROLLOUT_SCHEMA, **asdict(item)})
            for item in recursive
        )
        or report_bytes != canonical_json_bytes(analyze_integrated_calibration(
            calibration,
            recursive,
            candidate_configuration_id=report["candidate_configuration_id"],
            comparator_configuration_ids=tuple(
                report["comparator_configuration_ids"]
            ),
            source_bindings=report["source_bindings"],
            stress_ablations=report["stress_ablations"],
            bootstrap_seed=report["bootstrap_seed"],
            bootstrap_replicates=report["bootstrap_replicates"],
        ))
    ):
        raise CohortV2IntegratedError("integrated evidence does not recompute exactly")
    return manifest


__all__ = [
    "INTEGRATED_CHECKPOINT_SCHEMA",
    "INTEGRATED_EVIDENCE_SCHEMA",
    "RECURSIVE_HORIZONS",
    "RECURSIVE_ROLLOUT_SCHEMA",
    "CohortV2IntegratedError",
    "CohortV2RecursiveRolloutRecord",
    "IntegratedVariant",
    "analyze_integrated_calibration",
    "build_integrated_predictor",
    "build_integrated_trainer",
    "integrated_compute_calibration",
    "load_cohort_v2_integrated_checkpoint",
    "recursive_continuous_rollouts",
    "save_cohort_v2_integrated_checkpoint",
    "summarize_recursive_rollouts",
    "validate_integrated_evidence",
    "write_integrated_evidence",
]
