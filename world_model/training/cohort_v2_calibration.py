"""Rollout-level pre-confirmatory calibration analysis for cohort v2."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from statistics import mean, stdev
from typing import Mapping

from world_model.model import identity
from world_model.training.grid_artifacts import canonical_json_bytes


CALIBRATION_SCHEMA = "cohort_v2_preconfirmatory_calibration_v1"


class CohortV2CalibrationError(ValueError):
    """The pilot design, rollout metrics, or persisted report is invalid."""


@dataclass(frozen=True, slots=True)
class CohortV2CalibrationRecord:
    configuration_id: str
    exposure_role: str
    attempt_id: str
    scenario_lineage_identity: str
    coverage_stratum: str
    checkpoint_identity: str
    seed: int
    state_count: int
    mean_endpoint_prediction_error: float
    mean_endpoint_violation_rate: float
    mean_policy_compute_per_simulated_frame: float
    mean_full_compute_per_simulated_frame: float


@dataclass(frozen=True, slots=True)
class CohortV2StressGapRecord:
    stress_id: str
    exposure_role: str
    attempt_id: str
    scenario_lineage_identity: str
    coverage_stratum: str
    reference_configuration_id: str
    stressed_configuration_id: str
    metric: str
    degradation_gap: float


def _finite(value: float, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise CohortV2CalibrationError(f"{field} must be finite")
    return float(value)


def _validate_records(
    records: tuple[CohortV2CalibrationRecord, ...],
    stress: tuple[CohortV2StressGapRecord, ...],
) -> None:
    if not records or not stress:
        raise CohortV2CalibrationError("calibration or stress records are empty")
    keys = set()
    for item in records:
        if (
            not item.configuration_id
            or item.exposure_role not in ("calibration", "model_selection")
            or not item.attempt_id
            or not item.scenario_lineage_identity
            or not item.coverage_stratum
            or not item.checkpoint_identity
            or type(item.seed) is not int
            or item.seed < 0
            or type(item.state_count) is not int
            or item.state_count <= 0
        ):
            raise CohortV2CalibrationError("calibration record provenance is malformed")
        for field in (
            "mean_endpoint_prediction_error",
            "mean_endpoint_violation_rate",
            "mean_policy_compute_per_simulated_frame",
            "mean_full_compute_per_simulated_frame",
        ):
            value = _finite(getattr(item, field), field)
            if value < 0.0:
                raise CohortV2CalibrationError(f"{field} must be nonnegative")
        key = (item.configuration_id, item.exposure_role, item.attempt_id)
        if key in keys:
            raise CohortV2CalibrationError("calibration record keys are not unique")
        keys.add(key)
    stress_keys = set()
    for item in stress:
        if (
            not item.stress_id
            or item.exposure_role != "calibration"
            or not item.attempt_id
            or not item.scenario_lineage_identity
            or not item.coverage_stratum
            or not item.reference_configuration_id
            or not item.stressed_configuration_id
            or not item.metric
        ):
            raise CohortV2CalibrationError("stress-gap provenance is malformed")
        _finite(item.degradation_gap, "degradation gap")
        key = (item.stress_id, item.attempt_id)
        if key in stress_keys:
            raise CohortV2CalibrationError("stress-gap keys are not unique")
        stress_keys.add(key)


def _percentile(values: tuple[float, ...], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: tuple[float, ...]) -> dict[str, float | int]:
    if not values:
        raise CohortV2CalibrationError("metric summary is empty")
    return {
        "count": len(values),
        "maximum": max(values),
        "mean": mean(values),
        "minimum": min(values),
        "sample_standard_deviation": 0.0 if len(values) == 1 else stdev(values),
    }


def _bootstrap_mean_interval(
    values: tuple[float, ...], *, seed: int, replicates: int
) -> tuple[float, float]:
    if not values or replicates <= 0:
        raise CohortV2CalibrationError("bootstrap input is empty or invalid")
    generator = random.Random(seed)
    sampled = []
    for _ in range(replicates):
        sampled.append(mean(tuple(generator.choice(values) for _ in values)))
    return _percentile(tuple(sampled), 0.025), _percentile(tuple(sampled), 0.975)


def _by_attempt(
    records: tuple[CohortV2CalibrationRecord, ...],
    configuration_id: str,
    role: str,
) -> dict[str, CohortV2CalibrationRecord]:
    return {
        item.attempt_id: item
        for item in records
        if item.configuration_id == configuration_id and item.exposure_role == role
    }


def analyze_cohort_v2_calibration(
    records: tuple[CohortV2CalibrationRecord, ...],
    stress: tuple[CohortV2StressGapRecord, ...],
    *,
    candidate_configuration_id: str,
    eligible_comparator_ids: tuple[str, ...],
    source_bindings: Mapping[str, object],
    missing_integrations: tuple[str, ...],
    downstream_work: tuple[str, ...] = (),
    bootstrap_seed: int = 20260826,
    bootstrap_replicates: int = 10_000,
) -> dict[str, object]:
    """Select on model-selection records, then calibrate on rollout-level units."""
    _validate_records(records, stress)
    if (
        not candidate_configuration_id
        or not eligible_comparator_ids
        or len(set(eligible_comparator_ids)) != len(eligible_comparator_ids)
        or candidate_configuration_id in eligible_comparator_ids
        or type(bootstrap_seed) is not int
        or bootstrap_seed < 0
        or bootstrap_replicates <= 0
    ):
        raise CohortV2CalibrationError("frozen calibration design is malformed")
    configurations = {item.configuration_id for item in records}
    required = {candidate_configuration_id, *eligible_comparator_ids}
    if not required.issubset(configurations):
        raise CohortV2CalibrationError("frozen configuration evidence is incomplete")

    def selection_key(configuration_id: str) -> tuple[float, float, float, int]:
        values = tuple(
            item
            for item in records
            if item.configuration_id == configuration_id
            and item.exposure_role == "model_selection"
        )
        if not values:
            raise CohortV2CalibrationError("comparator model-selection evidence is empty")
        return (
            mean(tuple(item.mean_endpoint_prediction_error for item in values)),
            mean(tuple(item.mean_endpoint_violation_rate for item in values)),
            mean(tuple(item.mean_policy_compute_per_simulated_frame for item in values)),
            eligible_comparator_ids.index(configuration_id),
        )

    strongest = min(eligible_comparator_ids, key=selection_key)
    candidate = _by_attempt(records, candidate_configuration_id, "calibration")
    comparator = _by_attempt(records, strongest, "calibration")
    if set(candidate) != set(comparator) or len(candidate) < 2:
        raise CohortV2CalibrationError(
            "candidate and comparator require the same independent calibration rollouts"
        )
    attempts = tuple(sorted(candidate))
    endpoint_gains = tuple(
        comparator[key].mean_endpoint_prediction_error
        - candidate[key].mean_endpoint_prediction_error
        for key in attempts
    )
    violation_increases = tuple(
        candidate[key].mean_endpoint_violation_rate
        - comparator[key].mean_endpoint_violation_rate
        for key in attempts
    )
    gain_interval = _bootstrap_mean_interval(
        endpoint_gains, seed=bootstrap_seed, replicates=bootstrap_replicates
    )
    violation_interval = _bootstrap_mean_interval(
        violation_increases,
        seed=bootstrap_seed + 1,
        replicates=bootstrap_replicates,
    )
    comparator_error = mean(tuple(
        comparator[key].mean_endpoint_prediction_error for key in attempts
    ))
    gain_half_width = (gain_interval[1] - gain_interval[0]) / 2.0
    practical_effect = max(0.10 * comparator_error, gain_half_width)
    physical_margin = max(0.0, violation_interval[1])
    compute_values = tuple(
        item.mean_policy_compute_per_simulated_frame
        for item in records
        if item.exposure_role == "calibration"
        and item.configuration_id in required
    )
    budgets = tuple(dict.fromkeys(
        _percentile(compute_values, probability)
        for probability in (0.25, 0.5, 0.75, 1.0)
    ))

    configuration_summaries = []
    for configuration_id in sorted(required):
        values = tuple(
            item
            for item in records
            if item.configuration_id == configuration_id
            and item.exposure_role == "calibration"
        )
        if set(item.attempt_id for item in values) != set(attempts):
            raise CohortV2CalibrationError(
                f"{configuration_id} does not cover every calibration rollout"
            )
        configuration_summaries.append({
            "configuration_id": configuration_id,
            "endpoint_prediction_error": _summary(tuple(
                item.mean_endpoint_prediction_error for item in values
            )),
            "endpoint_violation_rate": _summary(tuple(
                item.mean_endpoint_violation_rate for item in values
            )),
            "full_compute_per_simulated_frame": _summary(tuple(
                item.mean_full_compute_per_simulated_frame for item in values
            )),
            "policy_compute_per_simulated_frame": _summary(tuple(
                item.mean_policy_compute_per_simulated_frame for item in values
            )),
        })

    stress_summaries = []
    for stress_id in sorted({item.stress_id for item in stress}):
        values = tuple(
            item.degradation_gap for item in stress if item.stress_id == stress_id
        )
        if {item.attempt_id for item in stress if item.stress_id == stress_id} != set(attempts):
            raise CohortV2CalibrationError(
                f"stress proxy {stress_id} does not cover every calibration rollout"
            )
        interval = _bootstrap_mean_interval(
            values,
            seed=bootstrap_seed + 2 + len(stress_summaries),
            replicates=bootstrap_replicates,
        )
        first = next(item for item in stress if item.stress_id == stress_id)
        stress_summaries.append({
            "bootstrap_95_interval": list(interval),
            "metric": first.metric,
            "reference_configuration_id": first.reference_configuration_id,
            "stress_id": stress_id,
            "stressed_configuration_id": first.stressed_configuration_id,
            "summary": _summary(values),
        })

    sensitivity = []
    for effect_multiplier in (0.5, 1.0, 1.5):
        for margin_multiplier in (0.0, 1.0, 2.0):
            threshold = practical_effect * effect_multiplier
            margin = physical_margin * margin_multiplier
            sensitivity.append({
                "calibration_only": True,
                "effect_threshold": threshold,
                "mean_gain_exceeds_threshold": mean(endpoint_gains) >= threshold,
                "mean_violation_increase_within_margin": (
                    mean(violation_increases) <= margin
                ),
                "physical_violation_margin": margin,
            })

    sufficient = not missing_integrations
    disposition = {
        "status": "sufficient_evidence_to_freeze_issue_34" if sufficient else "insufficient_evidence",
        "additional_calibration_work_required": [] if sufficient else list(missing_integrations),
    }
    return {
        "analysis_version": CALIBRATION_SCHEMA,
        "calibration_is_confirmatory": False,
        "candidate_configuration_id": candidate_configuration_id,
        "configuration_summaries": configuration_summaries,
        "disposition": disposition,
        "eligible_comparator_ids": list(eligible_comparator_ids),
        "endpoint_scope": {
            "authoritative_carrier_reintroduced_for_each_evaluated_state": True,
            "complete_rollout_gameplay_assessed": False,
            "evaluation_mode": "teacher_forced_local_successor_prediction",
            "recursive_fixed_h1_accumulation_assessed": False,
        },
        "exposure_audit": {
            "consumed_roles": ["calibration", "model_selection"],
            "final_evaluation_artifacts_accessed": False,
            "final_evaluation_derived_artifacts_accessed": False,
            "training_rollouts_counted_as_calibration_replicates": False,
        },
        "independent_calibration_replicates": len(attempts),
        "paired_primary_endpoint_gain": {
            "bootstrap_95_interval": list(gain_interval),
            "summary": _summary(endpoint_gains),
        },
        "paired_physical_violation_increase": {
            "bootstrap_95_interval": list(violation_interval),
            "summary": _summary(violation_increases),
        },
        "proposals_for_issue_34": {
            "compute_budgets_policy_units_per_simulated_frame": list(budgets),
            "failed_run_treatment": (
                "Source/provenance failure aborts the analysis. A model execution failure "
                "is retained as a failed replicate, is not replaced or excluded, and makes "
                "that configuration fail the constrained-gain rule at the affected budget."
            ),
            "physical_violation_margin": physical_margin,
            "practical_effect_threshold_absolute_endpoint_error_reduction": practical_effect,
            "replicate_design": {
                "analysis_unit": "complete_rollout",
                "fixed_replicate_count": len(attempts),
                "precision_target": (
                    "report the frozen bootstrap interval; its calibration half-width is "
                    f"{gain_half_width:.17g}"
                ),
            },
            "seed_policy": {
                "analysis_bootstrap_seed": bootstrap_seed,
                "bootstrap_replicates": bootstrap_replicates,
                "checkpoint_training_seed_is_frozen_per_configuration": True,
                "no_outcome_conditioned_seed_changes": True,
            },
            "strongest_comparator_id": strongest,
        },
        "sensitivity": sensitivity,
        "source_bindings": dict(source_bindings),
        "stress_test_degradation_proxies": stress_summaries,
        "downstream_work": list(downstream_work),
    }


def _digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _record_payload(item: object) -> dict[str, object]:
    return {"schema": CALIBRATION_SCHEMA, **asdict(item)}


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_cohort_v2_calibration(
    root: Path,
    records: tuple[CohortV2CalibrationRecord, ...],
    stress: tuple[CohortV2StressGapRecord, ...],
    *,
    candidate_configuration_id: str,
    eligible_comparator_ids: tuple[str, ...],
    source_bindings: Mapping[str, object],
    missing_integrations: tuple[str, ...],
    implementation_revision: str,
    downstream_work: tuple[str, ...] = (),
    bootstrap_seed: int = 20260826,
    bootstrap_replicates: int = 10_000,
) -> dict[str, object]:
    if not implementation_revision:
        raise CohortV2CalibrationError("implementation revision is empty")
    report = analyze_cohort_v2_calibration(
        records,
        stress,
        candidate_configuration_id=candidate_configuration_id,
        eligible_comparator_ids=eligible_comparator_ids,
        source_bindings=source_bindings,
        missing_integrations=missing_integrations,
        downstream_work=downstream_work,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    records_bytes = b"".join(canonical_json_bytes(_record_payload(item)) for item in records)
    stress_bytes = b"".join(canonical_json_bytes(_record_payload(item)) for item in stress)
    report_bytes = canonical_json_bytes(report)
    manifest = {
        "analysis": "report.json",
        "analysis_identity": _digest(report_bytes),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "calibration_artifact_identity": identity((
            CALIBRATION_SCHEMA,
            implementation_revision,
            _digest(records_bytes),
            _digest(stress_bytes),
            _digest(report_bytes),
        )),
        "candidate_configuration_id": candidate_configuration_id,
        "eligible_comparator_ids": list(eligible_comparator_ids),
        "implementation_revision": implementation_revision,
        "missing_integrations": list(missing_integrations),
        "downstream_work": list(downstream_work),
        "records": "replicate_metrics.jsonl",
        "records_identity": _digest(records_bytes),
        "schema": CALIBRATION_SCHEMA,
        "source_bindings": dict(source_bindings),
        "stress_records": "stress_gaps.jsonl",
        "stress_records_identity": _digest(stress_bytes),
    }
    root = Path(root)
    _atomic_write(root / "replicate_metrics.jsonl", records_bytes)
    _atomic_write(root / "stress_gaps.jsonl", stress_bytes)
    _atomic_write(root / "report.json", report_bytes)
    _atomic_write(root / "manifest.json", canonical_json_bytes(manifest))
    validate_cohort_v2_calibration(root)
    return manifest


def _load_jsonl(data: bytes, cls):
    try:
        rows = tuple(json.loads(line) for line in data.splitlines())
        return tuple(cls(**{key: value for key, value in row.items() if key != "schema"}) for row in rows)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise CohortV2CalibrationError(f"calibration JSONL is malformed: {error}") from error


def validate_cohort_v2_calibration(root: Path) -> dict[str, object]:
    root = Path(root)
    try:
        manifest_bytes = (root / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        records_bytes = (root / manifest["records"]).read_bytes()
        stress_bytes = (root / manifest["stress_records"]).read_bytes()
        report_bytes = (root / manifest["analysis"]).read_bytes()
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise CohortV2CalibrationError(f"cannot load calibration artifact: {error}") from error
    expected_artifact_identity = identity((
        CALIBRATION_SCHEMA,
        manifest.get("implementation_revision"),
        _digest(records_bytes),
        _digest(stress_bytes),
        _digest(report_bytes),
    ))
    if (
        manifest.get("schema") != CALIBRATION_SCHEMA
        or manifest_bytes != canonical_json_bytes(manifest)
        or manifest.get("records_identity") != _digest(records_bytes)
        or manifest.get("stress_records_identity") != _digest(stress_bytes)
        or manifest.get("analysis_identity") != _digest(report_bytes)
        or manifest.get("calibration_artifact_identity")
        != expected_artifact_identity
    ):
        raise CohortV2CalibrationError("calibration artifact identity is stale")
    records = _load_jsonl(records_bytes, CohortV2CalibrationRecord)
    stress = _load_jsonl(stress_bytes, CohortV2StressGapRecord)
    expected = analyze_cohort_v2_calibration(
        records,
        stress,
        candidate_configuration_id=manifest["candidate_configuration_id"],
        eligible_comparator_ids=tuple(manifest["eligible_comparator_ids"]),
        source_bindings=manifest["source_bindings"],
        missing_integrations=tuple(manifest["missing_integrations"]),
        downstream_work=tuple(manifest.get("downstream_work", ())),
        bootstrap_seed=manifest["bootstrap_seed"],
        bootstrap_replicates=manifest["bootstrap_replicates"],
    )
    if report_bytes != canonical_json_bytes(expected):
        raise CohortV2CalibrationError("calibration report does not recompute")
    return manifest


__all__ = [
    "CALIBRATION_SCHEMA",
    "CohortV2CalibrationError",
    "CohortV2CalibrationRecord",
    "CohortV2StressGapRecord",
    "analyze_cohort_v2_calibration",
    "validate_cohort_v2_calibration",
    "write_cohort_v2_calibration",
]
