"""Frozen issue-15 confirmatory analysis over complete final rollouts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from statistics import mean
from typing import Mapping

from world_model.model import identity
from world_model.training.cohort_v2_integrated import (
    CohortV2RecursiveRolloutRecord,
)
from world_model.training.grid_artifacts import canonical_json_bytes


SCHEMA = "cohort_v2_oracle_symbol_confirmatory_v1"
CANDIDATE_ID = "integrated_aggregated_joint_controller"


class CohortV2ConfirmatoryError(ValueError):
    """The frozen confirmatory inputs or evidence are invalid."""


def audit_final_entity_capacity(reader, *, max_entities: int) -> tuple[dict[str, object], ...]:
    """Check the frozen carrier width before any predictor evaluation."""
    rows = []
    for rollout in reader.rollouts:
        counts = tuple(
            len(frame.engine_state["entities"]) for frame in rollout.frame_records
        )
        maximum = max(counts)
        passed = maximum <= max_entities
        rows.append({
            "attempt_id": rollout.attempt_id,
            "coverage_stratum": rollout.coverage_stratum,
            "frame_record_count": len(rollout.frame_records),
            "maximum_entity_count": maximum,
            "declared_entity_slots": max_entities,
            "overflow_frame_count": sum(value > max_entities for value in counts),
            "failure_code": None if passed else "entity_slot_capacity_exceeded",
            "passed": passed,
        })
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class CohortV2ConfirmatoryRecord:
    protocol_identity: str
    release_identity: str
    partition_identity: str
    code_revision: str
    configuration_id: str
    comparison_role: str
    budget: float
    checkpoint_identity: str
    attempt_id: str
    seed: int
    exposure_role: str
    coverage_stratum: str
    state_count: int
    mean_endpoint_prediction_error: float
    mean_endpoint_violation_rate: float
    mean_policy_compute_per_simulated_frame: float
    mean_full_compute_per_simulated_frame: float

    def __post_init__(self) -> None:
        if self.comparison_role not in ("candidate", "comparator"):
            raise CohortV2ConfirmatoryError("comparison role is invalid")
        if self.exposure_role != "final_evaluation" or self.state_count <= 0:
            raise CohortV2ConfirmatoryError("confirmatory record is not a final rollout")
        values = (
            self.budget,
            self.mean_endpoint_prediction_error,
            self.mean_endpoint_violation_rate,
            self.mean_policy_compute_per_simulated_frame,
            self.mean_full_compute_per_simulated_frame,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise CohortV2ConfirmatoryError("confirmatory metrics must be finite and nonnegative")
        if any(
            not isinstance(value, str) or not value
            for value in (
                self.protocol_identity,
                self.release_identity,
                self.partition_identity,
                self.code_revision,
                self.configuration_id,
                self.checkpoint_identity,
                self.attempt_id,
                self.coverage_stratum,
            )
        ):
            raise CohortV2ConfirmatoryError("confirmatory source binding is missing")


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap(values: tuple[float, ...], seed: int, replicates: int) -> list[float]:
    generator = random.Random(seed)
    return [
        mean(tuple(generator.choice(values) for _ in values))
        for _ in range(replicates)
    ]


def analyze_cohort_v2_confirmatory(
    records: tuple[CohortV2ConfirmatoryRecord, ...],
    recursive: tuple[CohortV2RecursiveRolloutRecord, ...],
    protocol: Mapping[str, object],
    *,
    source_bindings: Mapping[str, object],
    capacity_audit: tuple[Mapping[str, object], ...] = (),
) -> dict[str, object]:
    if (
        protocol.get("schema") != "cohort_v2_prospective_statistical_protocol_v1"
        or protocol.get("status") != "frozen_before_final_evaluation"
    ):
        raise CohortV2ConfirmatoryError("issue-34 protocol is not frozen")
    protocol_identity = protocol.get("artifact_identity")
    policy = protocol.get("replicate_and_seed_policy")
    matrix = protocol.get("experiment_matrix")
    analysis = protocol.get("statistical_analysis")
    if not isinstance(protocol_identity, str) or not isinstance(policy, Mapping):
        raise CohortV2ConfirmatoryError("protocol identity or replicate policy is missing")
    if not isinstance(matrix, Mapping) or not isinstance(analysis, Mapping):
        raise CohortV2ConfirmatoryError("protocol experiment or analysis is missing")
    issue_15 = matrix.get("confirmatory_oracle_symbol_issue_15")
    if not isinstance(issue_15, Mapping):
        raise CohortV2ConfirmatoryError("issue-15 matrix is missing")
    comparisons = issue_15.get("comparisons")
    attempts = policy.get("fixed_attempt_ids")
    replicates = policy.get("analysis_bootstrap_replicates")
    if (
        not isinstance(comparisons, list)
        or len(comparisons) != 2
        or not isinstance(attempts, list)
        or len(attempts) != 6
        or replicates != 10_000
    ):
        raise CohortV2ConfirmatoryError("frozen confirmatory matrix is incomplete")

    failed_capacity = tuple(
        dict(item) for item in capacity_audit if item.get("passed") is False
    )
    if failed_capacity:
        if records or recursive:
            raise CohortV2ConfirmatoryError(
                "capacity failure cannot be combined with computed final metrics"
            )
        if (
            len(capacity_audit) != len(attempts)
            or {item.get("attempt_id") for item in capacity_audit} != set(attempts)
            or any(
                item.get("failure_code") != "entity_slot_capacity_exceeded"
                for item in failed_capacity
            )
        ):
            raise CohortV2ConfirmatoryError(
                "capacity audit does not cover the frozen final attempts"
            )
        decisions = [{
            "budget": float(comparison["budget"]),
            "strongest_comparator_id": str(comparison["strongest_comparator_id"]),
            "status": "candidate_model_execution_failure",
            "candidate_execution_failure_count": len(failed_capacity),
            "paired_complete_rollout_count": 0,
            "matched_compute_support": False,
            "endpoint_gain": "unavailable",
            "physical_violation_increase": "unavailable",
            "budget_rule_passed": False,
        } for comparison in comparisons]
        return {
            "schema": SCHEMA,
            "protocol_identity": protocol_identity,
            "candidate_configuration_id": CANDIDATE_ID,
            "decision": "unsupported",
            "decision_rationale": (
                "candidate_model_execution_failed_on_the_frozen_final_attempts"
            ),
            "budget_decisions": decisions,
            "paired_rollout_effects": [],
            "bootstrap": {
                "method": "not_run_due_to_candidate_model_execution_failure",
                "replicates": 0,
            },
            "fixed_h15_complete_rollout": {
                "status": "not_run_due_to_candidate_model_execution_failure",
                "recursive_physical_violation_status": "unavailable",
            },
            "sensitivity": [],
            "failed_missing_or_excluded_runs": list(failed_capacity),
            "failed_run_treatment": (
                "retained_without_replacement_or_exclusion_and_failed_both_budgets"
            ),
            "source_bindings": dict(source_bindings),
        }

    decisions = []
    sensitivity = []
    record_rows = []
    for comparison in comparisons:
        if not isinstance(comparison, Mapping):
            raise CohortV2ConfirmatoryError("frozen comparison is malformed")
        budget = float(comparison["budget"])
        comparator_id = str(comparison["strongest_comparator_id"])
        selected = tuple(item for item in records if item.budget == budget)
        candidate = {
            item.attempt_id: item
            for item in selected
            if item.comparison_role == "candidate"
            and item.configuration_id == CANDIDATE_ID
        }
        comparator = {
            item.attempt_id: item
            for item in selected
            if item.comparison_role == "comparator"
            and item.configuration_id == comparator_id
        }
        if (
            set(candidate) != set(attempts)
            or set(comparator) != set(attempts)
            or len(selected) != 12
            or any(item.protocol_identity != protocol_identity for item in selected)
        ):
            raise CohortV2ConfirmatoryError(
                "each budget must contain the frozen six paired final rollouts"
            )
        ordered_attempts = tuple(attempts)
        gains = tuple(
            comparator[item].mean_endpoint_prediction_error
            - candidate[item].mean_endpoint_prediction_error
            for item in ordered_attempts
        )
        violations = tuple(
            candidate[item].mean_endpoint_violation_rate
            - comparator[item].mean_endpoint_violation_rate
            for item in ordered_attempts
        )
        gain_samples = _bootstrap(
            gains, int(comparison["gain_bootstrap_seed"]), replicates
        )
        violation_samples = _bootstrap(
            violations, int(comparison["violation_bootstrap_seed"]), replicates
        )
        gain_interval = (
            _percentile(gain_samples, 0.025),
            _percentile(gain_samples, 0.975),
        )
        violation_interval = (
            _percentile(violation_samples, 0.025),
            _percentile(violation_samples, 0.975),
        )
        candidate_compute = mean(
            item.mean_policy_compute_per_simulated_frame for item in candidate.values()
        )
        comparator_compute = mean(
            item.mean_policy_compute_per_simulated_frame for item in comparator.values()
        )
        matched_support = candidate_compute <= budget and comparator_compute <= budget
        threshold = float(
            comparison["practical_effect_threshold_absolute_endpoint_error_reduction"]
        )
        margin = float(comparison["physical_violation_margin"])
        gain_passed = gain_interval[0] >= threshold
        violation_passed = violation_interval[1] <= margin
        passed = matched_support and gain_passed and violation_passed
        decisions.append({
            "budget": budget,
            "strongest_comparator_id": comparator_id,
            "paired_complete_rollout_count": len(ordered_attempts),
            "candidate_mean_policy_compute_per_simulated_frame": candidate_compute,
            "comparator_mean_policy_compute_per_simulated_frame": comparator_compute,
            "matched_compute_support": matched_support,
            "mean_endpoint_gain": mean(gains),
            "endpoint_gain_two_sided_95_interval": list(gain_interval),
            "endpoint_gain_one_sided_97_5_lower_bound": gain_interval[0],
            "practical_effect_threshold": threshold,
            "gain_rule_passed": gain_passed,
            "mean_physical_violation_increase": mean(violations),
            "physical_violation_increase_two_sided_95_interval": list(
                violation_interval
            ),
            "physical_violation_increase_one_sided_97_5_upper_bound": (
                violation_interval[1]
            ),
            "physical_violation_margin": margin,
            "violation_rule_passed": violation_passed,
            "budget_rule_passed": passed,
        })
        for threshold_multiplier in (0.5, 1.5):
            for margin_multiplier in (0.0, 2.0):
                sensitivity.append({
                    "budget": budget,
                    "threshold_multiplier": threshold_multiplier,
                    "margin_multiplier": margin_multiplier,
                    "gain_rule_passed": gain_interval[0] >= threshold * threshold_multiplier,
                    "violation_rule_passed": violation_interval[1] <= margin * margin_multiplier,
                    "descriptive_only": True,
                })
        record_rows.extend({
            "budget": budget,
            "attempt_id": attempt,
            "endpoint_gain": gain,
            "physical_violation_increase": violation,
        } for attempt, gain, violation in zip(
            ordered_attempts, gains, violations, strict=True
        ))

    recursive_attempts = {item.attempt_id for item in recursive}
    if (
        recursive_attempts != set(attempts)
        or len(recursive) != len(attempts)
        or any(
            item.exposure_role != "final_evaluation" or item.requested_horizon != 15
            for item in recursive
        )
    ):
        raise CohortV2ConfirmatoryError(
            "fixed-h15 recursive diagnostics must cover the six final rollouts"
        )
    supported = any(item["budget_rule_passed"] for item in decisions)
    return {
        "schema": SCHEMA,
        "protocol_identity": protocol_identity,
        "candidate_configuration_id": CANDIDATE_ID,
        "decision": "supported" if supported else "unsupported",
        "decision_rationale": (
            "at_least_one_budget_passed_the_frozen_constrained_gain_rule"
            if supported
            else "no_budget_passed_the_frozen_constrained_gain_rule"
        ),
        "budget_decisions": decisions,
        "paired_rollout_effects": record_rows,
        "bootstrap": {
            "method": "nonparametric_paired_complete_rollout_with_replacement",
            "replicates": replicates,
            "two_sided_interval": 0.95,
            "bonferroni_one_sided_bound": 0.975,
        },
        "fixed_h15_complete_rollout": {
            "complete_rollout_count": len(recursive),
            "mean_terminal_mse": mean(item.terminal_mse for item in recursive),
            "mean_error_auc": mean(item.error_auc for item in recursive),
            "mean_total_compute": mean(item.total_compute for item in recursive),
            "recursive_physical_violation_status": "unavailable",
        },
        "sensitivity": sensitivity,
        "failed_missing_or_excluded_runs": [],
        "source_bindings": dict(source_bindings),
    }


def _digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_cohort_v2_confirmatory_evidence(
    root: Path,
    records: tuple[CohortV2ConfirmatoryRecord, ...],
    recursive: tuple[CohortV2RecursiveRolloutRecord, ...],
    report: Mapping[str, object],
    *,
    implementation_revision: str,
    capacity_audit: tuple[Mapping[str, object], ...] = (),
) -> dict[str, object]:
    record_bytes = b"".join(
        canonical_json_bytes({"schema": SCHEMA, **asdict(item)}) for item in records
    )
    recursive_bytes = b"".join(
        canonical_json_bytes({"schema": SCHEMA, **asdict(item)}) for item in recursive
    )
    report_bytes = canonical_json_bytes(dict(report))
    capacity_bytes = b"".join(
        canonical_json_bytes({"schema": SCHEMA, **dict(item)})
        for item in capacity_audit
    )
    manifest = {
        "schema": SCHEMA,
        "implementation_revision": implementation_revision,
        "capacity_audit": "final_entity_capacity_audit.jsonl",
        "capacity_audit_identity": _digest(capacity_bytes),
        "confirmatory_records": "confirmatory_records.jsonl",
        "confirmatory_records_identity": _digest(record_bytes),
        "recursive_records": "fixed_h15_recursive_records.jsonl",
        "recursive_records_identity": _digest(recursive_bytes),
        "report": "report.json",
        "report_identity": _digest(report_bytes),
    }
    manifest["artifact_identity"] = identity((
        SCHEMA,
        implementation_revision,
        manifest["confirmatory_records_identity"],
        manifest["recursive_records_identity"],
        manifest["capacity_audit_identity"],
        manifest["report_identity"],
    ))
    root = Path(root)
    if root.exists():
        raise CohortV2ConfirmatoryError(f"immutable output already exists: {root}")
    _atomic_write(root / "confirmatory_records.jsonl", record_bytes)
    _atomic_write(root / "fixed_h15_recursive_records.jsonl", recursive_bytes)
    _atomic_write(root / "final_entity_capacity_audit.jsonl", capacity_bytes)
    _atomic_write(root / "report.json", report_bytes)
    _atomic_write(root / "manifest.json", canonical_json_bytes(manifest))
    return manifest


def validate_cohort_v2_confirmatory_evidence(
    root: Path,
    protocol: Mapping[str, object],
) -> dict[str, object]:
    root = Path(root)
    try:
        manifest_raw = (root / "manifest.json").read_bytes()
        manifest = json.loads(manifest_raw)
        record_bytes = (root / manifest["confirmatory_records"]).read_bytes()
        recursive_bytes = (root / manifest["recursive_records"]).read_bytes()
        capacity_bytes = (root / manifest["capacity_audit"]).read_bytes()
        report_bytes = (root / manifest["report"]).read_bytes()
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise CohortV2ConfirmatoryError(
            f"cannot load confirmatory evidence: {error}"
        ) from error
    records = tuple(
        CohortV2ConfirmatoryRecord(**{
            key: value for key, value in json.loads(line).items() if key != "schema"
        })
        for line in record_bytes.splitlines()
    )
    recursive = tuple(
        CohortV2RecursiveRolloutRecord(**{
            key: tuple(value) if key in {
                "effective_horizons", "cumulative_horizons",
                "authoritative_endpoint_identities", "endpoint_mse_curve",
            } else value
            for key, value in json.loads(line).items() if key != "schema"
        })
        for line in recursive_bytes.splitlines()
    )
    report = json.loads(report_bytes)
    capacity_audit = tuple(
        {key: value for key, value in json.loads(line).items() if key != "schema"}
        for line in capacity_bytes.splitlines()
    )
    expected_report = analyze_cohort_v2_confirmatory(
        records,
        recursive,
        protocol,
        source_bindings=report["source_bindings"],
        capacity_audit=capacity_audit,
    )
    expected_identity = identity((
        SCHEMA,
        manifest["implementation_revision"],
        _digest(record_bytes),
        _digest(recursive_bytes),
        _digest(capacity_bytes),
        _digest(report_bytes),
    ))
    if (
        manifest.get("schema") != SCHEMA
        or canonical_json_bytes(manifest) != manifest_raw
        or report != expected_report
        or manifest.get("artifact_identity") != expected_identity
        or manifest.get("confirmatory_records_identity") != _digest(record_bytes)
        or manifest.get("recursive_records_identity") != _digest(recursive_bytes)
        or manifest.get("capacity_audit_identity") != _digest(capacity_bytes)
        or manifest.get("report_identity") != _digest(report_bytes)
    ):
        raise CohortV2ConfirmatoryError(
            "confirmatory evidence differs from exact recomputation"
        )
    return manifest


__all__ = [
    "CANDIDATE_ID",
    "CohortV2ConfirmatoryError",
    "CohortV2ConfirmatoryRecord",
    "audit_final_entity_capacity",
    "analyze_cohort_v2_confirmatory",
    "validate_cohort_v2_confirmatory_evidence",
    "write_cohort_v2_confirmatory_evidence",
]
