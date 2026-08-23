"""Validate and publish the immutable cohort-v2 production release for issue #53."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Final

from scripts.cohort_v2_macro_semantics import (
    DERIVATION_SPEC_IDENTITY as MACRO_SPEC_IDENTITY,
    derive_capture_macro_labels,
    validate_capture_macro_derivation,
)
from scripts.cohort_v2_micro_relations import (
    DERIVATION_SPEC_IDENTITY as MICRO_SPEC_IDENTITY,
    derive_capture_micro_relations,
    validate_capture_micro_relation_derivation,
)
from scripts.cohort_v2_physical_violations import (
    DERIVATION_SPEC_IDENTITY as VIOLATION_SPEC_IDENTITY,
    derive_capture_physical_violations,
    validate_capture_physical_violation_derivation,
)
from scripts.cohort_v2_production_plans import (
    COLLECTION_IDENTITY,
    PARAMETER_IDENTITY,
    ROLE_ORDER,
    validate_issue_52_payloads,
)
from scripts.cohort_v2_replay import (
    _compare_contact_geometry,
    _compare_discrete_state_semantics,
    _compare_occurrences,
    _contact_geometry_occurrences,
    _event_occurrences,
    _final_lifecycle_projection,
    _first_launch_step,
    _relation_occurrences,
    exact_socket_comparison_rules_v1,
    semantic_identity,
)
from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_json
from scripts.observation_trace import validate_observation_trace
from scripts.physics_capture_v2 import (
    load_physics_capture_v2,
    normalized_initial_engine_state_identity,
)
from scripts.physics_capture_v2_persistence import validate_physics_capture_v2_artifact


RELEASE_IDENTITY: Final = "representative-cohort-v2-release-v1:issue-53"
PUBLICATION_IDENTITY: Final = "representative-cohort-v2-publication-v1:issue-53"
DERIVATION_INDEX_IDENTITY: Final = (
    "cohort-v2-authoritative-derivation-index-v1:issue-53"
)
SEALED_BUNDLE_IDENTITY: Final = "issue-53-final-evaluation-sealed-bundle-v1"
BUNDLE_IDENTITY: Final = "issue-53-cohort-v2-release-bundle-v1"
CENTRAL_LABELS: Final = (
    "contact",
    "supports",
    "steady-state",
    "structure-unstable",
    "excess_penetration",
    "unsupported_stationary_or_floating_body",
)


@dataclass(frozen=True, slots=True)
class ReleaseContract:
    version: int
    collection_identity: str
    parameter_identity: str
    release_identity: str
    publication_identity: str
    derivation_index_identity: str
    sealed_bundle_identity: str
    bundle_identity: str
    scenario_inventory_identity: str

    def schema(self, stem: str) -> str:
        return f"{stem}_v{self.version}"


V1_CONTRACT: Final = ReleaseContract(
    1,
    COLLECTION_IDENTITY,
    PARAMETER_IDENTITY,
    RELEASE_IDENTITY,
    PUBLICATION_IDENTITY,
    DERIVATION_INDEX_IDENTITY,
    SEALED_BUNDLE_IDENTITY,
    BUNDLE_IDENTITY,
    "cohort-v2-production-scenario-inventory-v1:issue-53",
)
V2_CONTRACT: Final = ReleaseContract(
    2,
    "cohort-v2-production-collection-plan-v2:issue-53:stable-only",
    "cohort-v2-production-parameter-plan-v2:issue-53:stable-only",
    "representative-cohort-v2-release-v2:issue-53:stable-only",
    "representative-cohort-v2-publication-v2:issue-53:stable-only",
    "cohort-v2-authoritative-derivation-index-v2:issue-53:stable-only",
    "issue-53-final-evaluation-sealed-bundle-v2:stable-only",
    "issue-53-cohort-v2-release-bundle-v2:stable-only",
    "cohort-v2-production-scenario-inventory-v2:issue-53:stable-only-release",
)
V3_CONTRACT: Final = ReleaseContract(
    3,
    (
        "cohort-v2-production-collection-plan-v3:issue-53:stable-only:"
        "anchor-order-correction"
    ),
    (
        "cohort-v2-production-parameter-plan-v3:issue-53:stable-only:"
        "anchor-order-correction"
    ),
    "representative-cohort-v2-release-v3:issue-53:stable-only:anchor-order-correction",
    "representative-cohort-v2-publication-v3:issue-53:stable-only:anchor-order-correction",
    "cohort-v2-authoritative-derivation-index-v3:issue-53:anchor-order-correction",
    "issue-53-final-evaluation-sealed-bundle-v3:anchor-order-correction",
    "issue-53-cohort-v2-release-bundle-v3:anchor-order-correction",
    "cohort-v2-production-scenario-inventory-v3:issue-53:anchor-order-correction",
)
V4_CONTRACT: Final = ReleaseContract(
    4,
    "cohort-v2-production-collection-plan-v4:issue-53:mixed-termination",
    "cohort-v2-production-parameter-plan-v4:issue-53:mixed-termination",
    "representative-cohort-v2-release-v4:issue-53:mixed-termination",
    "representative-cohort-v2-publication-v4:issue-53:mixed-termination",
    "cohort-v2-authoritative-derivation-index-v4:issue-53:mixed-termination",
    "issue-53-final-evaluation-sealed-bundle-v4:mixed-termination",
    "issue-53-cohort-v2-release-bundle-v4:mixed-termination",
    "cohort-v2-production-scenario-inventory-v4:issue-53:release",
)
V5_CONTRACT: Final = ReleaseContract(
    5,
    "cohort-v2-production-collection-plan-v5:issue-53:mixed-termination:workflow-time",
    "cohort-v2-production-parameter-plan-v5:issue-53:mixed-termination:workflow-time",
    "representative-cohort-v2-release-v5:issue-53:mixed-termination",
    "representative-cohort-v2-publication-v5:issue-53:mixed-termination",
    "cohort-v2-authoritative-derivation-index-v5:issue-53:mixed-termination",
    "issue-53-final-evaluation-sealed-bundle-v5:mixed-termination",
    "issue-53-cohort-v2-release-bundle-v5:mixed-termination",
    "cohort-v2-production-scenario-inventory-v5:issue-53:release",
)


def release_contract_for_collection(
    collection_plan: Mapping[str, Any],
) -> ReleaseContract:
    identity = collection_plan.get("identity")
    if identity == V1_CONTRACT.collection_identity:
        return V1_CONTRACT
    if identity == V2_CONTRACT.collection_identity:
        return V2_CONTRACT
    if identity == V3_CONTRACT.collection_identity:
        return V3_CONTRACT
    if identity == V4_CONTRACT.collection_identity:
        return V4_CONTRACT
    if identity == V5_CONTRACT.collection_identity:
        return V5_CONTRACT
    raise CohortV2ReleaseError("Issue-53 collection plan identity is unsupported")


def _contract_version_for_identity(collection_identity: str) -> int:
    if collection_identity == V5_CONTRACT.collection_identity:
        return 5
    if collection_identity == V4_CONTRACT.collection_identity:
        return 4
    if collection_identity == V3_CONTRACT.collection_identity:
        return 3
    if collection_identity == V2_CONTRACT.collection_identity:
        return 2
    return 1


def planned_termination_for_assignment(
    assignment: Mapping[str, Any], intervention: Mapping[str, Any]
) -> str:
    expectations = assignment.get("termination_expectations")
    if isinstance(expectations, Mapping):
        value = expectations.get(intervention["id"])
        if not isinstance(value, str):
            raise CohortV2ReleaseError(
                "Issue-53 assignment termination expectation is missing"
            )
        return value
    value = intervention.get("intended_termination_class")
    if not isinstance(value, str):
        raise CohortV2ReleaseError("Issue-53 intervention termination is missing")
    return value


class CohortV2ReleaseError(ValueError):
    """Issue-53 production evidence is incomplete, stale, or cross-bound."""


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CohortV2ReleaseError(f"Cannot load {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise CohortV2ReleaseError(f"{label} must be an object")
    return value


def production_intervention_identity(
    intervention_id: str,
    collection_identity: str = COLLECTION_IDENTITY,
) -> str:
    version = _contract_version_for_identity(collection_identity)
    return semantic_identity(
        f"cohort-v2-production-intervention-v{version}",
        collection_identity,
        intervention_id,
    )


def production_attempt_identity(
    exposure_role: str,
    intervention_id: str,
    collection_identity: str = COLLECTION_IDENTITY,
) -> str:
    version = _contract_version_for_identity(collection_identity)
    return semantic_identity(
        f"cohort-v2-production-attempt-v{version}",
        collection_identity,
        exposure_role,
        intervention_id,
    )


def replay_attempt_identity(
    exposure_role: str,
    intervention_id: str,
    collection_identity: str = COLLECTION_IDENTITY,
) -> str:
    version = _contract_version_for_identity(collection_identity)
    return semantic_identity(
        f"cohort-v2-production-replay-attempt-v{version}",
        collection_identity,
        exposure_role,
        intervention_id,
    )


def _expected_slots(collection_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    collection_identity = release_contract_for_collection(
        collection_plan
    ).collection_identity
    interventions = {
        item["id"]: item for item in collection_plan["interventions"]
    }
    slots = []
    for assignment in collection_plan["assignments"]:
        for intervention_id in assignment["intervention_ids"]:
            slots.append(
                {
                    "exposure_role": assignment["exposure_role"],
                    "dataset_partition": assignment["dataset_partition"],
                    "scenario_manifest_identity": assignment[
                        "scenario_manifest_identity"
                    ],
                    "scenario_lineage_identity": assignment[
                        "scenario_lineage_identity"
                    ],
                    "level_instance_identity": assignment["level_instance_identity"],
                    "scenario_template_identity": assignment[
                        "scenario_template_identity"
                    ],
                    "benchmark_condition_identity": assignment[
                        "benchmark_condition_identity"
                    ],
                    "intervention": interventions[intervention_id],
                    "expected_termination": planned_termination_for_assignment(
                        assignment, interventions[intervention_id]
                    ),
                    "attempt_id": production_attempt_identity(
                        assignment["exposure_role"],
                        intervention_id,
                        collection_identity,
                    ),
                }
            )
    return slots


def validate_issue_53_execution_report(
    report: Mapping[str, Any], collection_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the exact one-attempt-per-slot outcome-independent ledger."""
    contract = release_contract_for_collection(collection_plan)
    if report.get("schema") != contract.schema("issue_53_production_execution_report"):
        raise CohortV2ReleaseError("Issue-53 execution report schema is unsupported")
    if (
        report.get("collection_plan_identity") != contract.collection_identity
        or report.get("production_parameter_plan_identity") != contract.parameter_identity
        or report.get("outcome_independent_accounting") is not True
        or report.get("retry_count") != 0
    ):
        raise CohortV2ReleaseError("Issue-53 report is not bound to the frozen plans")
    expected = _expected_slots(collection_plan)
    ledger = report.get("attempt_ledger")
    if not isinstance(ledger, list) or len(ledger) != len(expected):
        raise CohortV2ReleaseError("Issue-53 report does not account for all 24 slots")
    for observed, slot in zip(ledger, expected):
        exact = {
            "attempt_id": slot["attempt_id"],
            "exposure_role": slot["exposure_role"],
            "dataset_partition": slot["dataset_partition"],
            "scenario_manifest_identity": slot["scenario_manifest_identity"],
            "scenario_lineage_identity": slot["scenario_lineage_identity"],
            "level_instance_identity": slot["level_instance_identity"],
            "scenario_template_identity": slot["scenario_template_identity"],
            "benchmark_condition_identity": slot["benchmark_condition_identity"],
            "intervention_id": slot["intervention"]["id"],
            "intervention_identity": production_intervention_identity(
                slot["intervention"]["id"], contract.collection_identity
            ),
            "intended_coverage_stratum": slot["intervention"][
                "intended_coverage_stratum"
            ],
            "intervention_source": slot["intervention"]["intervention_source"],
            "expected_termination": slot["expected_termination"],
        }
        if not isinstance(observed, Mapping) or any(
            observed.get(field) != value for field, value in exact.items()
        ):
            raise CohortV2ReleaseError(
                f"Issue-53 attempt ordering or binding differs at {slot['attempt_id']}"
            )
        if observed.get("status") not in {"accepted", "rejected", "failed"}:
            raise CohortV2ReleaseError("Issue-53 attempt status is unknown")
        if observed["status"] == "accepted":
            if not isinstance(observed.get("artifact_path"), str):
                raise CohortV2ReleaseError("Accepted issue-53 attempt has no artifact")
        elif not all(
            isinstance(observed.get(field), str) and observed[field]
            for field in ("reason", "failure_code", "quarantine_path")
        ):
            raise CohortV2ReleaseError(
                "Failed issue-53 attempt lacks typed quarantine accounting"
            )
    counts = Counter(item["status"] for item in ledger)
    if report.get("counts") != {
        "planned": len(expected),
        "attempted": len(ledger),
        "accepted": counts["accepted"],
        "rejected": counts["rejected"],
        "failed": counts["failed"],
        "quarantined": counts["rejected"] + counts["failed"],
    }:
        raise CohortV2ReleaseError("Issue-53 execution counts differ from its ledger")
    return dict(report)


def _artifact_path(runtime_root: Path, value: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (runtime_root / path).resolve()
    root = runtime_root.resolve()
    if resolved != root and root not in resolved.parents:
        raise CohortV2ReleaseError("Issue-53 artifact path escapes the runtime root")
    if not resolved.is_dir():
        raise CohortV2ReleaseError(f"Issue-53 rollout artifact is missing: {resolved}")
    return resolved


def _inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]


def _validate_rollout(
    artifact: Path,
    entry: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    metadata = _load_object(artifact / "metadata.json", "rollout metadata")
    validate_physics_capture_v2_artifact(artifact, metadata)
    capture = load_physics_capture_v2(artifact / "physics_capture_v2.json")
    expected_bindings = {
        "scenario_template_id": entry["scenario_template_identity"],
        "level_instance_id": entry["level_instance_identity"],
        "scenario_lineage_id": entry["scenario_lineage_identity"],
        "rollout_id": entry["attempt_id"],
        "intervention_id": entry["intervention_identity"],
    }
    if (
        capture.source_bindings != expected_bindings
        or metadata.get("scenario_manifest_identity")
        != entry["scenario_manifest_identity"]
        or capture.configured_fixed_step_capture_stride != 1
    ):
        raise CohortV2ReleaseError(
            f"Issue-53 rollout source binding is stale: {entry['attempt_id']}"
        )
    observation = validate_observation_trace(artifact / "observation-trace")
    expected_observation_bindings = {
        "scenario_template_identity": entry["scenario_template_identity"],
        "level_instance_identity": entry["level_instance_identity"],
        "source_scenario_lineage_identity": entry["scenario_lineage_identity"],
        "rollout_identity": entry["attempt_id"],
    }
    if (
        observation["source_bindings"] != expected_observation_bindings
        or observation["exposure_role"] != entry["exposure_role"]
        or metadata.get("observation_trace_manifest_identity")
        != observation["identity"]
    ):
        raise CohortV2ReleaseError(
            f"Issue-53 observation binding is stale: {entry['attempt_id']}"
        )
    return capture, observation


def compare_production_replay(
    original_artifact: Path,
    replay_artifact: Path,
    *,
    original_attempt_id: str,
    replay_attempt_id: str,
    original_action: Mapping[str, Any],
    replay_action: Mapping[str, Any],
    exposure_role: str,
    contract: ReleaseContract = V1_CONTRACT,
) -> dict[str, Any]:
    """Compare one production rollout with its one exact-socket replay."""
    original = load_physics_capture_v2(original_artifact / "physics_capture_v2.json")
    replay = load_physics_capture_v2(replay_artifact / "physics_capture_v2.json")
    original_observation = validate_observation_trace(
        original_artifact / "observation-trace"
    )
    replay_observation = validate_observation_trace(replay_artifact / "observation-trace")
    rules = exact_socket_comparison_rules_v1()
    maximum_delta = rules["maximum_relative_fixed_step_delta"]
    components = []

    def component(name: str, passed: bool, details: Mapping[str, Any]) -> None:
        components.append(
            {
                "component": name,
                "status": "equality" if passed else "mismatch",
                "details": dict(details),
            }
        )

    component(
        "attempt_identity",
        original_attempt_id != replay_attempt_id
        and original.capture_id != replay.capture_id
        and original.shot_id != replay.shot_id,
        {"original": original_attempt_id, "replay": replay_attempt_id},
    )
    original_bindings = dict(original.source_bindings)
    replay_bindings = dict(replay.source_bindings)
    original_bindings.pop("rollout_id", None)
    replay_bindings.pop("rollout_id", None)
    component(
        "source_binding",
        original_bindings == replay_bindings,
        {"exposure_role": exposure_role},
    )
    component(
        "intervention",
        original_action == replay_action,
        {"socket_command": original_action.get("socket_command")},
    )
    component(
        "initial_engine_state",
        normalized_initial_engine_state_identity(original)
        == normalized_initial_engine_state_identity(replay),
        {},
    )
    component(
        "fixed_step_capture_contract",
        original.configured_fixed_step_capture_stride
        == replay.configured_fixed_step_capture_stride
        == 1,
        {},
    )
    components.extend(
        (
            _compare_discrete_state_semantics(
                original.record, replay.record, maximum_delta
            ),
            _compare_occurrences(
                "event_semantics",
                _event_occurrences(original.record),
                _event_occurrences(replay.record),
                maximum_delta,
            ),
            _compare_occurrences(
                "contact_semantics",
                _relation_occurrences(original.record, "contact"),
                _relation_occurrences(replay.record, "contact"),
                maximum_delta,
            ),
            _compare_contact_geometry(
                _contact_geometry_occurrences(original.record),
                _contact_geometry_occurrences(replay.record),
                maximum_delta,
                rules["maximum_contact_separation_delta"],
            ),
            _compare_occurrences(
                "support_semantics",
                _relation_occurrences(original.record, "support"),
                _relation_occurrences(replay.record, "support"),
                maximum_delta,
            ),
        )
    )
    component(
        "terminal_entity_lifecycle",
        _final_lifecycle_projection(original.record)
        == _final_lifecycle_projection(replay.record),
        {},
    )
    original_origin = _first_launch_step(original.record)
    replay_origin = _first_launch_step(replay.record)
    original_terminal = original.record["terminal_evidence"]
    replay_terminal = replay.record["terminal_evidence"]
    terminal_delta = abs(
        original_terminal["fixed_step"]
        - original_origin
        - replay_terminal["fixed_step"]
        + replay_origin
    )
    component(
        "termination_semantics",
        original_terminal["reason"] == replay_terminal["reason"]
        and terminal_delta <= maximum_delta,
        {"fixed_step_delta": terminal_delta},
    )
    component(
        "observation_synchronization_and_access",
        all(
            original_observation[key] == replay_observation[key]
            for key in (
                "schema",
                "exposure_role",
                "observation_configuration",
                "access_policy",
            )
        ),
        {"exposure_role": exposure_role},
    )
    passed = all(
        item["status"] in {"equality", "tolerated", "not_required"}
        for item in components
    )
    return {
        "schema": contract.schema("cohort_v2_production_replay_verdict"),
        "original_attempt_identity": original_attempt_id,
        "replay_attempt_identity": replay_attempt_id,
        "exposure_role": exposure_role,
        "comparison_rules_identity": rules["identity"],
        "components": components,
        "passed": passed,
    }


def production_replay_report(
    verdicts: Sequence[Mapping[str, Any]],
    *,
    contract: ReleaseContract = V1_CONTRACT,
) -> dict[str, Any]:
    if len(verdicts) != 4 or [item.get("exposure_role") for item in verdicts] != list(
        ROLE_ORDER
    ):
        raise CohortV2ReleaseError("Production replay must cover all four exposure roles")
    if any(item.get("passed") is not True for item in verdicts):
        raise CohortV2ReleaseError("A production deterministic replay verdict failed")
    report = {
        "schema": contract.schema("cohort_v2_production_replay_report"),
        "identity": "",
        "collection_plan_identity": contract.collection_identity,
        "comparison_rules_identity": exact_socket_comparison_rules_v1()["identity"],
        "proof_count": 4,
        "retry_count": 0,
        "verdicts": [dict(item) for item in verdicts],
        "passed": True,
    }
    report["identity"] = semantic_identity(
        f"cohort-v2-production-replay-report-v{contract.version}",
        contract.collection_identity,
        *(item["replay_attempt_identity"] for item in verdicts),
    )
    return report


def _copy_rollout(source: Path, destination: Path) -> None:
    if destination.exists():
        raise CohortV2ReleaseError(f"Immutable rollout destination exists: {destination}")
    shutil.copytree(source, destination)


def _write_derivations(
    destination: Path,
    capture: Any,
    *,
    source_reference: str,
    release_identity: str = RELEASE_IDENTITY,
) -> list[dict[str, str]]:
    builders = (
        ("micro", derive_capture_micro_relations, validate_capture_micro_relation_derivation),
        ("macro", derive_capture_macro_labels, validate_capture_macro_derivation),
        (
            "physical-violations",
            derive_capture_physical_violations,
            validate_capture_physical_violation_derivation,
        ),
    )
    artifacts = []
    for name, derive, validate in builders:
        value = derive(
            capture,
            source_reference=source_reference,
            source_capture_bundle_identity=release_identity,
        )
        validate(
            value,
            capture,
            source_reference=source_reference,
            source_capture_bundle_identity=release_identity,
        )
        path = destination / f"{name}.json"
        write_immutable_cohort_v2_json(value, path)
        artifacts.append({"kind": name, "identity": value["identity"], "path": path.name})
    return artifacts


def _quality_report(
    report: Mapping[str, Any], replay_report: Mapping[str, Any]
) -> dict[str, Any]:
    version = _contract_version_for_identity(
        str(report.get("collection_plan_identity"))
    )
    ledger = report["attempt_ledger"]
    accepted = [item for item in ledger if item["status"] == "accepted"]
    shortfalls = [
        {
            "attempt_id": item["attempt_id"],
            "intended_coverage_stratum": item["intended_coverage_stratum"],
            "realized_coverage_strata": item.get("realized_coverage_strata", []),
        }
        for item in accepted
        if item["intended_coverage_stratum"]
        not in item.get("realized_coverage_strata", [])
    ]
    termination_mismatches = [
        {
            "attempt_id": item["attempt_id"],
            "expected": item["expected_termination"],
            "observed": item.get("terminal_reason"),
        }
        for item in accepted
        if item.get("terminal_reason") != item["expected_termination"]
    ]
    passed = (
        len(accepted) == report["counts"]["planned"]
        and not shortfalls
        and not termination_mismatches
        and replay_report.get("passed") is True
        and replay_report.get("proof_count") == 4
    )
    return {
        "schema": f"cohort_v2_production_quality_report_v{version}",
        "collection_plan_identity": report["collection_plan_identity"],
        "production_parameter_plan_identity": report[
            "production_parameter_plan_identity"
        ],
        "counts": dict(report["counts"]),
        "accepted_by_exposure_role": dict(
            sorted(Counter(item["exposure_role"] for item in accepted).items())
        ),
        "accepted_by_benchmark_condition": dict(
            sorted(
                Counter(item["benchmark_condition_identity"] for item in accepted).items()
            )
        ),
        "accepted_by_intervention_source": dict(
            sorted(Counter(item["intervention_source"] for item in accepted).items())
        ),
        "assigned_stratum_counts": dict(
            sorted(
                Counter(item["intended_coverage_stratum"] for item in accepted).items()
            )
        ),
        "observed_termination_counts": dict(
            sorted(
                Counter(
                    str(item.get("terminal_reason") or "unavailable")
                    for item in accepted
                ).items()
            )
        ),
        "coverage_shortfalls": shortfalls,
        "termination_mismatches": termination_mismatches,
        "replay_proof_count": replay_report.get("proof_count", 0),
        "required_capability_counts": {
            **{name: len(accepted) for name in CENTRAL_LABELS},
            "agent_observation": len(accepted),
            "canonical_observation_access_restriction": len(accepted),
            "fixed_step_capture": len(accepted),
            "complete_raw_contact_intervals": len(accepted),
            "atomic_rollout_validation": len(accepted),
            "typed_failure_and_quarantine_accounting": report["counts"]["planned"],
            "source_bound_derivations": len(accepted),
            "version_bounded_deterministic_replay": replay_report.get(
                "proof_count", 0
            ),
        },
        "systematic_exporter_defects": [],
        "passed": passed,
    }


def _public_quality_report(
    report: Mapping[str, Any],
    replay_report: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    """Project quality outcomes to the three non-final exposure roles."""
    ledger = [
        item
        for item in report["attempt_ledger"]
        if item["exposure_role"] != "final_evaluation"
    ]
    accepted = [item for item in ledger if item["status"] == "accepted"]
    counts = Counter(item["status"] for item in ledger)
    non_final_replays = [
        item
        for item in replay_report.get("verdicts", [])
        if item.get("exposure_role") != "final_evaluation"
        and item.get("passed") is True
    ]
    shortfalls = [
        {
            "attempt_id": item["attempt_id"],
            "intended_coverage_stratum": item["intended_coverage_stratum"],
            "realized_coverage_strata": item.get("realized_coverage_strata", []),
        }
        for item in accepted
        if item["intended_coverage_stratum"]
        not in item.get("realized_coverage_strata", [])
    ]
    mismatches = [
        {
            "attempt_id": item["attempt_id"],
            "expected": item["expected_termination"],
            "observed": item.get("terminal_reason"),
        }
        for item in accepted
        if item.get("terminal_reason") != item["expected_termination"]
    ]
    accepted_count = len(accepted)
    return {
        "schema": quality["schema"],
        "collection_plan_identity": quality["collection_plan_identity"],
        "production_parameter_plan_identity": quality[
            "production_parameter_plan_identity"
        ],
        "outcome_scope": "non_final_only",
        "sealed_final_evaluation_outcomes": True,
        "counts": {
            "planned": len(ledger),
            "attempted": len(ledger),
            "accepted": counts["accepted"],
            "rejected": counts["rejected"],
            "failed": counts["failed"],
            "quarantined": counts["rejected"] + counts["failed"],
        },
        "accepted_by_exposure_role": dict(
            sorted(Counter(item["exposure_role"] for item in accepted).items())
        ),
        "accepted_by_benchmark_condition": dict(
            sorted(
                Counter(
                    item["benchmark_condition_identity"] for item in accepted
                ).items()
            )
        ),
        "accepted_by_intervention_source": dict(
            sorted(Counter(item["intervention_source"] for item in accepted).items())
        ),
        "assigned_stratum_counts": dict(
            sorted(
                Counter(
                    item["intended_coverage_stratum"] for item in accepted
                ).items()
            )
        ),
        "observed_termination_counts": dict(
            sorted(
                Counter(
                    str(item.get("terminal_reason") or "unavailable")
                    for item in accepted
                ).items()
            )
        ),
        "coverage_shortfalls": shortfalls,
        "termination_mismatches": mismatches,
        "replay_proof_count": len(non_final_replays),
        "required_capability_counts": {
            **{name: accepted_count for name in CENTRAL_LABELS},
            "agent_observation": accepted_count,
            "canonical_observation_access_restriction": accepted_count,
            "fixed_step_capture": accepted_count,
            "complete_raw_contact_intervals": accepted_count,
            "atomic_rollout_validation": accepted_count,
            "typed_failure_and_quarantine_accounting": len(ledger),
            "source_bound_derivations": accepted_count,
            "version_bounded_deterministic_replay": len(non_final_replays),
        },
        "systematic_exporter_defects": [
            item
            for item in quality.get("systematic_exporter_defects", [])
            if item.get("exposure_role") != "final_evaluation"
        ],
        "passed": quality["passed"],
    }


def validate_published_issue_53_evidence(
    output: Path, sealed_output: Path
) -> dict[str, Any]:
    """Revalidate immutable public membership and the sealed final boundary."""
    public = Path(output)
    sealed = Path(sealed_output)
    collection = _load_object(public / "collection-plan.json", "collection plan")
    parameters = _load_object(
        public / "production-parameter-plan.json", "production parameter plan"
    )
    contract = release_contract_for_collection(collection)
    if parameters.get("identity") != contract.parameter_identity:
        raise CohortV2ReleaseError("Issue-53 publication plan identities differ")
    bundle = _load_object(public / "bundle-manifest.json", "issue-53 bundle")
    if (
        bundle.get("schema") != contract.schema("issue_53_cohort_v2_release_bundle")
        or bundle.get("identity") != contract.bundle_identity
    ):
        raise CohortV2ReleaseError("Issue-53 public bundle identity is stale")
    public_members = [
        item["path"]
        for item in _inventory(public)
        if item["path"] != "bundle-manifest.json"
    ]
    if bundle.get("artifacts") != public_members:
        raise CohortV2ReleaseError("Issue-53 public bundle membership is stale")
    sealed_manifest = _load_object(
        sealed / "sealed-bundle-manifest.json", "sealed issue-53 bundle"
    )
    if (
        sealed_manifest.get("identity") != contract.sealed_bundle_identity
        or sealed_manifest.get("ordinary_workflow_access") is not False
    ):
        raise CohortV2ReleaseError("Issue-53 sealed final boundary is stale")
    sealed_members = [
        item["path"]
        for item in _inventory(sealed)
        if item["path"] != "sealed-bundle-manifest.json"
    ]
    if sealed_manifest.get("artifacts") != sealed_members:
        raise CohortV2ReleaseError("Issue-53 sealed bundle membership is stale")
    quality = _load_object(
        public / "production-quality-report.json", "production quality report"
    )
    if bundle.get("passed") is not quality.get("passed"):
        raise CohortV2ReleaseError("Issue-53 bundle and quality dispositions differ")
    if bundle["passed"]:
        publication = _load_object(
            public / "cohort-v2-publication.json", "cohort-v2 publication"
        )
        release = _load_object(public / "cohort-v2-release.json", "cohort-v2 release")
        derivations = _load_object(
            public / "authoritative-derivation-index.json", "derivation index"
        )
        if (
            publication.get("identity") != contract.publication_identity
            or release.get("identity") != contract.release_identity
            or derivations.get("identity") != contract.derivation_index_identity
            or set(derivations.get("accepted_labels", {})) != set(CENTRAL_LABELS)
            or publication.get("disposition") != "complete"
            or release.get("disposition") != "complete"
            or sealed_manifest.get("passed") is not True
        ):
            raise CohortV2ReleaseError("Issue-53 complete release identities are stale")
        for rollout in release.get("primary_rollouts", []):
            root = public / rollout["path"]
            if not root.is_dir() or rollout.get("files") != _inventory(root):
                raise CohortV2ReleaseError(
                    "Issue-53 primary rollout inventory is stale"
                )
    return {
        "schema": contract.schema("issue_53_cohort_v2_release_validation_result"),
        "bundle_identity": contract.bundle_identity,
        "sealed_final_evaluation_bundle_identity": contract.sealed_bundle_identity,
        "disposition": "complete" if bundle["passed"] else "incomplete",
        "passed": bundle["passed"],
    }


def publish_issue_53_evidence(
    *,
    repository_root: Path,
    runtime_root: Path,
    output: Path,
    sealed_output: Path,
    final_access_audit: Mapping[str, Any],
    plan_root: Path | None = None,
) -> dict[str, Any]:
    """Publish a complete release, or immutable incomplete accounting, from one run."""
    repository_root = Path(repository_root).resolve()
    runtime_root = Path(runtime_root).resolve()
    output = Path(output)
    sealed_output = Path(sealed_output)
    if output.exists() or sealed_output.exists():
        raise CohortV2ReleaseError("Issue-53 immutable publication output already exists")
    selected_plan_root = (
        repository_root / "data/runtime_evidence/issue-52"
        if plan_root is None
        else Path(plan_root).resolve()
    )
    collection_path = selected_plan_root / "collection-plan.json"
    parameters_path = selected_plan_root / "production-parameter-plan.json"
    collection = _load_object(collection_path, "issue-53 collection plan")
    parameters = _load_object(parameters_path, "issue-53 parameter plan")
    contract = release_contract_for_collection(collection)
    if contract.version == 1:
        validate_issue_52_payloads(
            {
                "collection-plan.json": collection,
                "production-parameter-plan.json": parameters,
            },
            repository_root,
        )
    elif contract.version == 2:
        from scripts.cohort_v2_production_plans_v2 import validate_plan_v2_evidence

        validate_plan_v2_evidence(selected_plan_root, repository_root=repository_root)
    elif contract.version == 3:
        from scripts.cohort_v2_production_plans_v3 import validate_plan_v3_evidence

        validate_plan_v3_evidence(selected_plan_root, repository_root=repository_root)
    elif contract.version == 4:
        from scripts.cohort_v2_production_plans_v4 import validate_plan_v4_evidence

        validate_plan_v4_evidence(selected_plan_root, repository_root=repository_root)
    else:
        from scripts.cohort_v2_production_plans_v5 import validate_plan_v5_evidence

        validate_plan_v5_evidence(selected_plan_root, repository_root=repository_root)
    if parameters.get("identity") != contract.parameter_identity:
        raise CohortV2ReleaseError("Selected production plans have mismatched identities")
    report = validate_issue_53_execution_report(
        _load_object(runtime_root / "production-execution-report.json", "execution report"),
        collection,
    )
    replay_report = _load_object(
        runtime_root / "production-replay-report.json", "production replay report"
    )
    if replay_report.get("passed") is True:
        expected_replay = production_replay_report(
            replay_report.get("verdicts", []), contract=contract
        )
        if replay_report != expected_replay:
            raise CohortV2ReleaseError("Production replay report identity is stale")
    elif (
        replay_report.get("schema") != contract.schema("cohort_v2_production_replay_report")
        or replay_report.get("collection_plan_identity") != contract.collection_identity
        or not isinstance(replay_report.get("proof_count"), int)
        or not 0 <= replay_report["proof_count"] < 4
        or not isinstance(replay_report.get("verdicts"), list)
    ):
        raise CohortV2ReleaseError("Incomplete production replay report is malformed")
    quality = _quality_report(report, replay_report)

    output.parent.mkdir(parents=True, exist_ok=True)
    sealed_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".issue-53-public-", dir=output.parent) as public_temp, tempfile.TemporaryDirectory(
        prefix=".issue-53-sealed-", dir=sealed_output.parent
    ) as sealed_temp:
        public = Path(public_temp) / "bundle"
        sealed = Path(sealed_temp) / "bundle"
        public.mkdir()
        sealed.mkdir()
        write_immutable_cohort_v2_json(
            _public_quality_report(report, replay_report, quality),
            public / "production-quality-report.json",
        )
        write_immutable_cohort_v2_json(
            quality, sealed / "production-quality-report.json"
        )
        write_immutable_cohort_v2_json(
            replay_report, sealed / "production-replay-report.json"
        )
        write_immutable_cohort_v2_json(
            dict(final_access_audit), sealed / "final-access-audit.json"
        )
        shutil.copyfile(
            runtime_root / "authorities/authorized-final-access-manifest.json",
            sealed / "authorized-final-access-manifest.json",
        )
        shutil.copyfile(
            runtime_root / "player-provenance.json", public / "player-provenance.json"
        )

        scenario_inventory = []
        for assignment in collection["assignments"]:
            role = assignment["exposure_role"]
            filename = role.replace("_", "-")
            if role == "final_evaluation":
                authority_root = sealed / "scenario-authority"
                authority_root.mkdir(parents=True, exist_ok=True)
                manifest_path = authority_root / "final-evaluation.json"
                xml_path = authority_root / "final-evaluation.xml"
                shutil.copyfile(
                    runtime_root / "authorities/manifests/final-evaluation.json",
                    manifest_path,
                )
                shutil.copyfile(
                    runtime_root / "authorities/xml/final-evaluation.xml",
                    xml_path,
                )
                manifest_reference = contract.sealed_bundle_identity
            else:
                authority_root = public / "scenario-authorities"
                authority_root.mkdir(parents=True, exist_ok=True)
                manifest_path = authority_root / f"{filename}.json"
                xml_path = authority_root / f"{filename}.xml"
                shutil.copyfile(
                    runtime_root / f"authorities/manifests/{filename}.json",
                    manifest_path,
                )
                shutil.copyfile(
                    runtime_root / f"authorities/xml/{filename}.xml",
                    xml_path,
                )
                manifest_reference = manifest_path.relative_to(public).as_posix()
            scenario_inventory.append(
                {
                    "exposure_role": role,
                    "scenario_manifest_identity": assignment[
                        "scenario_manifest_identity"
                    ],
                    "scenario_lineage_identity": assignment[
                        "scenario_lineage_identity"
                    ],
                    "level_instance_identity": assignment["level_instance_identity"],
                    "scenario_template_identity": assignment[
                        "scenario_template_identity"
                    ],
                    "manifest_reference": manifest_reference,
                }
            )
        write_immutable_cohort_v2_json(
            {
                "schema": contract.schema("cohort_v2_production_scenario_inventory"),
                "identity": contract.scenario_inventory_identity,
                "collection_plan_identity": contract.collection_identity,
                "entries": scenario_inventory,
            },
            public / "scenario-inventory.json",
        )

        public_ledger = []
        primary_inventory = []
        derivation_artifacts = []
        for entry in report["attempt_ledger"]:
            final = entry["exposure_role"] == "final_evaluation"
            target_root = sealed if final else public
            published_entry = dict(entry)
            if entry["status"] == "accepted":
                artifact = _artifact_path(runtime_root, entry["artifact_path"])
                capture, _ = _validate_rollout(artifact, entry)
                rollout_relative = Path("primary-rollouts") / entry["attempt_id"]
                rollout_target = target_root / rollout_relative
                _copy_rollout(artifact, rollout_target)
                derivation_relative = Path("derivations") / entry["attempt_id"]
                derived = _write_derivations(
                    target_root / derivation_relative,
                    capture,
                    source_reference=(
                        f"{rollout_relative.as_posix()}/physics_capture_v2.json"
                    ),
                    release_identity=contract.release_identity,
                )
                published_entry["artifact_path"] = (
                    contract.sealed_bundle_identity
                    if final
                    else rollout_relative.as_posix()
                )
                if not final:
                    primary_inventory.append(
                        {
                            "attempt_id": entry["attempt_id"],
                            "exposure_role": entry["exposure_role"],
                            "capture_id": capture.capture_id,
                            "path": rollout_relative.as_posix(),
                            "files": _inventory(rollout_target),
                        }
                    )
                    derivation_artifacts.extend(
                        {
                            "attempt_id": entry["attempt_id"],
                            "exposure_role": entry["exposure_role"],
                            "kind": item["kind"],
                            "identity": item["identity"],
                            "path": (derivation_relative / item["path"]).as_posix(),
                        }
                        for item in derived
                    )
            else:
                quarantine_value = entry.get("quarantine_path")
                if not isinstance(quarantine_value, str) or not quarantine_value:
                    raise CohortV2ReleaseError(
                        "Rejected issue-53 attempt has no quarantine path"
                    )
                quarantine = _artifact_path(runtime_root, quarantine_value)
                quarantine_relative = Path("quarantine") / entry["attempt_id"]
                _copy_rollout(quarantine, target_root / quarantine_relative)
                published_entry["quarantine_path"] = (
                    contract.sealed_bundle_identity
                    if final
                    else quarantine_relative.as_posix()
                )
                published_entry["failure_manifest_path"] = (
                    contract.sealed_bundle_identity
                    if final
                    else (
                        quarantine_relative
                        / Path(str(entry["failure_manifest_path"])).name
                    ).as_posix()
                )
            if final:
                published_entry["status"] = "sealed"
                published_entry["reason"] = None
                published_entry["failure_code"] = None
                published_entry["terminal_reason"] = None
                published_entry["terminal_span_fixed_steps"] = None
                published_entry["realized_coverage_strata"] = []
            public_ledger.append(published_entry)
        public_accounting = {
            **report,
            "attempt_ledger": public_ledger,
            "counts": _public_quality_report(
                report, replay_report, quality
            )["counts"],
            "outcome_scope": "non_final_only",
            "sealed_final_evaluation_outcomes": True,
        }
        write_immutable_cohort_v2_json(
            public_accounting, public / "production-attempt-accounting.json"
        )
        shutil.copyfile(collection_path, public / "collection-plan.json")
        shutil.copyfile(parameters_path, public / "production-parameter-plan.json")
        shutil.copyfile(
            selected_plan_root / "partition-exposure-manifest.json"
            if (selected_plan_root / "partition-exposure-manifest.json").is_file()
            else repository_root
            / "data/runtime_evidence/issue-47/partition-exposure-manifest.json",
            public / "partition-exposure-manifest.json",
        )

        if not quality["passed"]:
            sealed_manifest = {
                "schema": contract.schema("issue_53_final_evaluation_sealed_bundle"),
                "identity": contract.sealed_bundle_identity,
                "collection_plan_identity": contract.collection_identity,
                "authorized_workflow_identity": final_access_audit.get(
                    "workflow_manifest_identity"
                ),
                "attempt_ids": [
                    item["attempt_id"]
                    for item in report["attempt_ledger"]
                    if item["exposure_role"] == "final_evaluation"
                ],
                "artifacts": [item["path"] for item in _inventory(sealed)],
                "ordinary_workflow_access": False,
                "disposition": "incomplete",
                "passed": False,
            }
            write_immutable_cohort_v2_json(
                sealed_manifest, sealed / "sealed-bundle-manifest.json"
            )
            bundle = {
                "schema": contract.schema("issue_53_cohort_v2_release_bundle"),
                "identity": contract.bundle_identity,
                "disposition": "incomplete",
                "collection_plan_identity": contract.collection_identity,
                "production_parameter_plan_identity": contract.parameter_identity,
                "sealed_final_evaluation_bundle_identity": contract.sealed_bundle_identity,
                "artifacts": [item["path"] for item in _inventory(public)],
                "passed": False,
            }
            write_immutable_cohort_v2_json(bundle, public / "bundle-manifest.json")
            os.replace(sealed, sealed_output)
            os.replace(public, output)
            return validate_published_issue_53_evidence(output, sealed_output)

        sealed_manifest = {
            "schema": contract.schema("issue_53_final_evaluation_sealed_bundle"),
            "identity": contract.sealed_bundle_identity,
            "collection_plan_identity": contract.collection_identity,
            "authorized_workflow_identity": final_access_audit.get(
                "workflow_manifest_identity"
            ),
            "attempt_ids": [
                item["attempt_id"]
                for item in report["attempt_ledger"]
                if item["exposure_role"] == "final_evaluation"
            ],
            "artifacts": [
                item["path"] for item in _inventory(sealed)
            ],
            "ordinary_workflow_access": False,
            "passed": True,
        }
        write_immutable_cohort_v2_json(
            sealed_manifest, sealed / "sealed-bundle-manifest.json"
        )
        derivation_index = {
            "schema": contract.schema("cohort_v2_authoritative_derivation_index"),
            "identity": contract.derivation_index_identity,
            "source_cohort_release_identity": contract.release_identity,
            "accepted_labels": {
                "contact": MICRO_SPEC_IDENTITY,
                "supports": MICRO_SPEC_IDENTITY,
                "steady-state": MACRO_SPEC_IDENTITY,
                "structure-unstable": MACRO_SPEC_IDENTITY,
                "excess_penetration": VIOLATION_SPEC_IDENTITY,
                "unsupported_stationary_or_floating_body": VIOLATION_SPEC_IDENTITY,
            },
            "artifacts": derivation_artifacts,
            "sealed_final_evaluation_bundle_identity": contract.sealed_bundle_identity,
        }
        write_immutable_cohort_v2_json(
            derivation_index, public / "authoritative-derivation-index.json"
        )
        release = {
            "schema": contract.schema("representative_cohort_v2_release"),
            "identity": contract.release_identity,
            "release_version": contract.version,
            "capability_declaration_identity": collection["authority"][
                "capability_declaration_identity"
            ],
            "source_pilot_report_identity": collection["authority"][
                "accepted_pilot_report_identity"
            ],
            "collection_plan": {
                "identity": contract.collection_identity,
                "path": "collection-plan.json",
            },
            "production_parameter_plan": {
                "identity": contract.parameter_identity,
                "path": "production-parameter-plan.json",
            },
            "partition_manifest": {
                "identity": collection["authority"]["partition_manifest_identity"],
                "path": "partition-exposure-manifest.json",
            },
            "scenario_inventory": {
                "identity": contract.scenario_inventory_identity,
                "path": "scenario-inventory.json",
            },
            "player_provenance_path": "player-provenance.json",
            "primary_rollouts": primary_inventory,
            "sealed_final_evaluation": {
                "bundle_identity": contract.sealed_bundle_identity,
                "attempt_ids": sealed_manifest["attempt_ids"],
            },
            "quality_report_path": "production-quality-report.json",
            "attempt_accounting_path": "production-attempt-accounting.json",
            "production_replay": {
                "identity": replay_report["identity"],
                "proof_count": replay_report["proof_count"],
                "sealed_bundle_identity": contract.sealed_bundle_identity,
            },
            "authoritative_derivation_index": {
                "identity": contract.derivation_index_identity,
                "path": "authoritative-derivation-index.json",
            },
            "disposition": "complete",
        }
        write_immutable_cohort_v2_json(release, public / "cohort-v2-release.json")
        publication = {
            "schema": contract.schema("representative_cohort_v2_publication"),
            "identity": contract.publication_identity,
            "cohort_release_identity": contract.release_identity,
            "cohort_release_path": "cohort-v2-release.json",
            "authoritative_derivation_index_identity": contract.derivation_index_identity,
            "sealed_final_evaluation_bundle_identity": contract.sealed_bundle_identity,
            "disposition": "complete",
        }
        write_immutable_cohort_v2_json(
            publication, public / "cohort-v2-publication.json"
        )
        bundle = {
            "schema": contract.schema("issue_53_cohort_v2_release_bundle"),
            "identity": contract.bundle_identity,
            "publication_identity": contract.publication_identity,
            "cohort_release_identity": contract.release_identity,
            "authoritative_derivation_index_identity": contract.derivation_index_identity,
            "sealed_final_evaluation_bundle_identity": contract.sealed_bundle_identity,
            "artifacts": [item["path"] for item in _inventory(public)],
            "passed": True,
        }
        write_immutable_cohort_v2_json(bundle, public / "bundle-manifest.json")
        os.replace(sealed, sealed_output)
        os.replace(public, output)
    return validate_published_issue_53_evidence(output, sealed_output)
