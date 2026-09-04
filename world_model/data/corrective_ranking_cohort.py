"""Frozen plans for issue #68's fresh broad-action corrective cohort."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math
from typing import Any, Final

from world_model.data.successor_cohort import ACTION_BOUNDS, GENERATOR_FAMILIES
from world_model.planning.gameplay import SlingshotActionBounds
from world_model.training.action_ranking_probe import (
    BROAD_ACTION_DESIGN_ID,
    broad_action_candidates,
)


PLAN_SCHEMA: Final = "issue_68_corrective_collection_plan_v2"
PILOT_REPORT_SCHEMA: Final = "issue_68_corrective_pilot_report_v2"
RELEASE_SCHEMA: Final = "issue_68_corrective_cohort_release_v2"
ROLES: Final = ("calibration", "model_selection")
PILOT_STATES_PER_ROLE: Final = 12
DEFAULT_PRODUCTION_STATES_PER_ROLE: Final = 200
FIXED_RETRY_LIMIT: Final = 2
FAILURE_COST: Final = 1_000_000_000.0
ISSUE_63_DISCRIMINATING_STATE_FRACTION: Final = 10 / 24
MINIMUM_DISCRIMINATION_IMPROVEMENT: Final = 0.10

_PHASE_SEED_BASES: Final = {
    "pilot": {"calibration": 68_300_000, "model_selection": 68_400_000},
    "production": {
        "calibration": 680_300_000,
        "model_selection": 680_400_000,
    },
}


class CorrectiveRankingCohortError(ValueError):
    """The issue-68 collection contract is inconsistent."""


def realized_endpoint_cost(
    *,
    active_pigs: int,
    active_blocks: int,
    pig_contact: bool,
    block_contact: bool,
    support_change: bool,
    pig_displacement: float,
    block_displacement: float,
) -> tuple[float, dict[str, float]]:
    """Keep removal lexicographically primary, then reward bounded progress."""

    if (
        type(active_pigs) is not int
        or active_pigs < 0
        or type(active_blocks) is not int
        or active_blocks < 0
        or type(pig_contact) is not bool
        or type(block_contact) is not bool
        or type(support_change) is not bool
        or not math.isfinite(pig_displacement)
        or pig_displacement < 0.0
        or not math.isfinite(block_displacement)
        or block_displacement < 0.0
    ):
        raise CorrectiveRankingCohortError("realized endpoint inputs are invalid")
    components = {
        "pig_contact": 0.40 if pig_contact else 0.0,
        "support_change": 0.25 if support_change else 0.0,
        "block_contact": 0.10 if block_contact else 0.0,
        "pig_displacement": 0.15 * min(round(pig_displacement, 3), 1.0),
        "block_displacement": 0.08 * min(round(block_displacement, 3), 1.0),
    }
    total = sum(components.values())
    progress = {**components, "total": total}
    count_cost = active_pigs * 1000 + active_blocks
    return count_cost + 0.999 * (1.0 - total), progress


def action_bounds() -> SlingshotActionBounds:
    return SlingshotActionBounds(
        tuple(ACTION_BOUNDS["drag_x"]),
        tuple(ACTION_BOUNDS["drag_y"]),
        tuple(ACTION_BOUNDS["tap_time_ms"]),
        int(ACTION_BOUNDS["release_time_ms"]),
    )


def release_identity(plan_identity: str) -> str:
    if not plan_identity:
        raise CorrectiveRankingCohortError("release requires a plan identity")
    return f"issue-68-corrective-release-v2:{plan_identity}"


def _cost_contract() -> dict[str, Any]:
    return {
        "identity": "issue-68-realized-endpoint-cost-v2",
        "lower_is_better": True,
        "primary_count_cost": "1000 * active_pigs + active_blocks",
        "fractional_progress_tie_break": {
            "formula": "0.999 * (1 - bounded_progress)",
            "pig_contact_weight": 0.40,
            "support_change_weight": 0.25,
            "block_contact_weight": 0.10,
            "pig_displacement_weight": 0.15,
            "block_displacement_weight": 0.08,
            "displacement_cap_world_units": 1.0,
            "displacement_round_decimals": 3,
        },
        "failure_cost": FAILURE_COST,
    }


def _design() -> dict[str, Any]:
    candidates = broad_action_candidates("declared-source-state", action_bounds())
    return {
        "identity": BROAD_ACTION_DESIGN_ID,
        "source_bound": True,
        "outcome_independent_membership": True,
        "candidate_count": len(candidates),
        "strata": [
            {
                "ordinal": item.ordinal,
                "action_stratum": item.action_stratum,
                "drag_x": item.action.drag_x,
                "drag_y": item.action.drag_y,
                "tap_time_ms": item.action.tap_time_ms,
            }
            for item in candidates
        ],
    }


def _states(phase: str, states_per_role: int) -> list[dict[str, Any]]:
    states = []
    global_ordinal = 0
    for role_index, role in enumerate(ROLES):
        for role_ordinal in range(states_per_role):
            family = GENERATOR_FAMILIES[(role_ordinal + role_index) % 2]
            state_identity = (
                f"issue-68-{phase}-v2-{role}-state-{role_ordinal + 1:04d}"
            )
            candidates = broad_action_candidates(state_identity, action_bounds())
            states.append({
                "identity": state_identity,
                "slot_identity": f"{state_identity}:scenario-lineage",
                "ordinal": global_ordinal,
                "role_ordinal": role_ordinal,
                "exposure_role": role,
                "generator_family": family["name"],
                "generation_seed": (
                    _PHASE_SEED_BASES[phase][role] + role_ordinal
                ),
                "candidate_design_identity": BROAD_ACTION_DESIGN_ID,
                "carrier_anchor_candidate_ordinal": (role_ordinal % 12) + 1,
                "candidates": [
                    {
                        "identity": item.identity,
                        "ordinal": item.ordinal,
                        "action_stratum": item.action_stratum,
                        "drag_x": item.action.drag_x,
                        "drag_y": item.action.drag_y,
                        "tap_time_ms": item.action.tap_time_ms,
                    }
                    for item in candidates
                ],
            })
            global_ordinal += 1
    return states


def build_plan(
    phase: str,
    *,
    states_per_role: int,
    pilot_report_identity: str | None = None,
    median_candidate_seconds: float | None = None,
    median_candidate_bytes: int | None = None,
) -> dict[str, Any]:
    if phase not in _PHASE_SEED_BASES:
        raise CorrectiveRankingCohortError("collection phase is unsupported")
    if type(states_per_role) is not int or states_per_role < 1:
        raise CorrectiveRankingCohortError("states per role must be positive")
    if phase == "pilot":
        if states_per_role != PILOT_STATES_PER_ROLE or pilot_report_identity is not None:
            raise CorrectiveRankingCohortError("pilot size and ancestry are fixed")
    elif not pilot_report_identity:
        raise CorrectiveRankingCohortError("production requires the passing pilot")
    candidate_count = states_per_role * len(ROLES) * 12
    identity = (
        f"issue-68-{phase}-plan-v2:states-per-role-{states_per_role}"
    )
    plan = {
        "schema": PLAN_SCHEMA,
        "identity": identity,
        "phase": phase,
        "source_issue": 68,
        "non_final_only": True,
        "exposure_roles": list(ROLES),
        "role_counts": {role: states_per_role for role in ROLES},
        "generator_families": [dict(item) for item in GENERATOR_FAMILIES],
        "action_bounds": dict(ACTION_BOUNDS),
        "action_design": _design(),
        "realized_cost_contract": _cost_contract(),
        "fixed_retry_limit": FIXED_RETRY_LIMIT,
        "failure_treatment": {
            "candidate_failure_cost": FAILURE_COST,
            "candidate_replacement": False,
            "state_replacement": False,
        },
        "stopping_rule": "attempt every frozen candidate through the fixed retry limit",
        "states": _states(phase, states_per_role),
        "pilot_report_identity": pilot_report_identity,
        "resource_decision": {
            "independent_states_per_role": states_per_role,
            "scheduled_candidate_executions": candidate_count,
            "median_candidate_seconds_from_pilot": median_candidate_seconds,
            "median_candidate_bytes_from_pilot": median_candidate_bytes,
            "estimated_collection_hours_single_worker": (
                None
                if median_candidate_seconds is None
                else median_candidate_seconds * candidate_count / 3600.0
            ),
            "estimated_collection_bytes": (
                None
                if median_candidate_bytes is None
                else median_candidate_bytes * candidate_count
            ),
            "membership_selected_without_production_outcomes": True,
        },
        "disjoint_seed_reservations": {
            "issue_68": [
                min(_PHASE_SEED_BASES[phase].values()),
                max(_PHASE_SEED_BASES[phase].values()) + states_per_role - 1,
            ],
            "future_issue_64_must_exclude_issue_68_seeds": True,
        },
        "supersedes": {
            "pilot_plan_identity": (
                "issue-68-pilot-plan-v1:states-per-role-4"
            ),
            "pilot_report_identity": (
                "issue-68-pilot-report-v1:"
                "issue-68-pilot-plan-v1:states-per-role-4"
            ),
            "disposition": "failed_insufficient_outcome_discrimination",
            "candidate_outcomes_reused": False,
        },
        "model_selection_access_rule": (
            "issue-69 must freeze its training/calibration decision rule before "
            "loading model-selection bundles"
        ),
        "outcome_conditioned_membership": False,
        "final_evaluation_opened": False,
    }
    return validate_plan(plan, phase=phase)


def build_pilot_plan() -> dict[str, Any]:
    return build_plan("pilot", states_per_role=PILOT_STATES_PER_ROLE)


def build_production_plan(
    pilot_report: Mapping[str, Any],
    *,
    states_per_role: int = DEFAULT_PRODUCTION_STATES_PER_ROLE,
) -> dict[str, Any]:
    report = validate_pilot_report(pilot_report)
    recommended = int(
        report["sample_size_justification"]["recommended_states_per_role"]
    )
    if states_per_role < recommended:
        raise CorrectiveRankingCohortError(
            f"production requires at least {recommended} independent states per role"
        )
    runtime = report["runtime_cost"]
    return build_plan(
        "production",
        states_per_role=states_per_role,
        pilot_report_identity=str(report["identity"]),
        median_candidate_seconds=float(runtime["median_seconds_per_candidate"]),
        median_candidate_bytes=int(runtime["median_bytes_per_candidate"]),
    )


def validate_plan(
    plan: Mapping[str, Any], *, phase: str | None = None
) -> dict[str, Any]:
    value = dict(plan)
    required = {
        "schema", "identity", "phase", "source_issue", "non_final_only",
        "exposure_roles", "role_counts", "generator_families", "action_bounds",
        "action_design", "realized_cost_contract", "fixed_retry_limit",
        "failure_treatment",
        "stopping_rule", "states", "pilot_report_identity", "resource_decision",
        "disjoint_seed_reservations", "supersedes", "model_selection_access_rule",
        "outcome_conditioned_membership", "final_evaluation_opened",
    }
    actual_phase = value.get("phase")
    if (
        set(value) != required
        or value.get("schema") != PLAN_SCHEMA
        or actual_phase not in _PHASE_SEED_BASES
        or (phase is not None and actual_phase != phase)
        or value.get("source_issue") != 68
        or value.get("non_final_only") is not True
        or value.get("exposure_roles") != list(ROLES)
        or value.get("generator_families") != [dict(item) for item in GENERATOR_FAMILIES]
        or value.get("action_bounds") != ACTION_BOUNDS
        or value.get("action_design") != _design()
        or value.get("realized_cost_contract") != _cost_contract()
        or value.get("fixed_retry_limit") != FIXED_RETRY_LIMIT
        or value.get("failure_treatment") != {
            "candidate_failure_cost": FAILURE_COST,
            "candidate_replacement": False,
            "state_replacement": False,
        }
        or value.get("outcome_conditioned_membership") is not False
        or value.get("final_evaluation_opened") is not False
    ):
        raise CorrectiveRankingCohortError("corrective collection plan differs")
    role_counts = value.get("role_counts")
    if (
        not isinstance(role_counts, Mapping)
        or set(role_counts) != set(ROLES)
        or len(set(role_counts.values())) != 1
    ):
        raise CorrectiveRankingCohortError("corrective role counts differ")
    states_per_role = role_counts[ROLES[0]]
    if type(states_per_role) is not int or states_per_role < 1:
        raise CorrectiveRankingCohortError("corrective role count is invalid")
    if value.get("identity") != (
        f"issue-68-{actual_phase}-plan-v2:states-per-role-{states_per_role}"
    ):
        raise CorrectiveRankingCohortError("corrective plan identity differs")
    expected_states = _states(str(actual_phase), states_per_role)
    if value.get("states") != expected_states:
        raise CorrectiveRankingCohortError("corrective states or candidates differ")
    resources = value.get("resource_decision")
    expected_reservation = {
        "issue_68": [
            min(_PHASE_SEED_BASES[str(actual_phase)].values()),
            max(_PHASE_SEED_BASES[str(actual_phase)].values())
            + states_per_role
            - 1,
        ],
        "future_issue_64_must_exclude_issue_68_seeds": True,
    }
    if (
        value.get("stopping_rule")
        != "attempt every frozen candidate through the fixed retry limit"
        or value.get("disjoint_seed_reservations") != expected_reservation
        or value.get("supersedes") != {
            "pilot_plan_identity": "issue-68-pilot-plan-v1:states-per-role-4",
            "pilot_report_identity": (
                "issue-68-pilot-report-v1:"
                "issue-68-pilot-plan-v1:states-per-role-4"
            ),
            "disposition": "failed_insufficient_outcome_discrimination",
            "candidate_outcomes_reused": False,
        }
        or value.get("model_selection_access_rule")
        != (
            "issue-69 must freeze its training/calibration decision rule before "
            "loading model-selection bundles"
        )
        or not isinstance(resources, Mapping)
        or set(resources) != {
            "independent_states_per_role",
            "scheduled_candidate_executions",
            "median_candidate_seconds_from_pilot",
            "median_candidate_bytes_from_pilot",
            "estimated_collection_hours_single_worker",
            "estimated_collection_bytes",
            "membership_selected_without_production_outcomes",
        }
        or resources.get("independent_states_per_role") != states_per_role
        or resources.get("scheduled_candidate_executions")
        != states_per_role * len(ROLES) * 12
        or resources.get("membership_selected_without_production_outcomes")
        is not True
    ):
        raise CorrectiveRankingCohortError("corrective freeze rules differ")
    counts = Counter(state["exposure_role"] for state in expected_states)
    seeds = [state["generation_seed"] for state in expected_states]
    if dict(counts) != dict(role_counts) or len(seeds) != len(set(seeds)):
        raise CorrectiveRankingCohortError("corrective roles or seeds overlap")
    if actual_phase == "pilot":
        if (
            states_per_role != PILOT_STATES_PER_ROLE
            or value["pilot_report_identity"] is not None
            or resources["median_candidate_seconds_from_pilot"] is not None
            or resources["median_candidate_bytes_from_pilot"] is not None
            or resources["estimated_collection_hours_single_worker"] is not None
            or resources["estimated_collection_bytes"] is not None
        ):
            raise CorrectiveRankingCohortError("pilot membership differs")
    else:
        seconds = resources["median_candidate_seconds_from_pilot"]
        artifact_bytes = resources["median_candidate_bytes_from_pilot"]
        if (
            not isinstance(value["pilot_report_identity"], str)
            or type(seconds) not in (int, float)
            or not float(seconds) >= 0.0
            or type(artifact_bytes) is not int
            or artifact_bytes < 0
            or resources["estimated_collection_hours_single_worker"]
            != float(seconds)
            * resources["scheduled_candidate_executions"]
            / 3600.0
            or resources["estimated_collection_bytes"]
            != artifact_bytes * resources["scheduled_candidate_executions"]
        ):
            raise CorrectiveRankingCohortError("production lacks pilot ancestry")
    return value


def validate_pilot_report(report: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(report)
    if (
        value.get("schema") != PILOT_REPORT_SCHEMA
        or value.get("identity")
        != f"issue-68-pilot-report-v2:{value.get('pilot_plan_identity')}"
        or value.get("planned_state_count") != PILOT_STATES_PER_ROLE * len(ROLES)
        or value.get("scheduled_candidate_count")
        != PILOT_STATES_PER_ROLE * len(ROLES) * 12
        or value.get("completed_candidate_count")
        != value.get("scheduled_candidate_count")
        or value.get("passed") is not True
        or value.get("outcome_conditioned_membership") is not False
        or value.get("final_evaluation_opened") is not False
    ):
        raise CorrectiveRankingCohortError("corrective pilot report did not pass")
    justification = value.get("sample_size_justification")
    if (
        not isinstance(justification, Mapping)
        or justification.get("recommended_states_per_role")
        != DEFAULT_PRODUCTION_STATES_PER_ROLE
    ):
        raise CorrectiveRankingCohortError("corrective sample-size decision differs")
    return value


def validate_role_access(
    role: str,
    *,
    release_manifest_identity: str,
    decision_freeze: Mapping[str, Any] | None = None,
) -> None:
    """Enforce the #69 calibration-before-model-selection exposure order."""

    if role not in ROLES or not release_manifest_identity:
        raise CorrectiveRankingCohortError("corrective role access is invalid")
    if role == "calibration":
        return
    if (
        not isinstance(decision_freeze, Mapping)
        or decision_freeze.get("schema")
        != "issue_69_corrective_decision_freeze_v1"
        or decision_freeze.get("issue_68_release_identity")
        != release_manifest_identity
        or decision_freeze.get("information_roles")
        != ["training", "calibration"]
        or decision_freeze.get("frozen") is not True
        or decision_freeze.get("model_selection_opened") is not False
        or decision_freeze.get("final_evaluation_opened") is not False
    ):
        raise CorrectiveRankingCohortError(
            "freeze the issue-69 training/calibration decision before "
            "model-selection access"
        )


__all__ = [
    "CorrectiveRankingCohortError",
    "DEFAULT_PRODUCTION_STATES_PER_ROLE",
    "FAILURE_COST",
    "FIXED_RETRY_LIMIT",
    "ISSUE_63_DISCRIMINATING_STATE_FRACTION",
    "MINIMUM_DISCRIMINATION_IMPROVEMENT",
    "PILOT_REPORT_SCHEMA",
    "PILOT_STATES_PER_ROLE",
    "PLAN_SCHEMA",
    "RELEASE_SCHEMA",
    "ROLES",
    "action_bounds",
    "build_pilot_plan",
    "build_production_plan",
    "release_identity",
    "realized_endpoint_cost",
    "validate_pilot_report",
    "validate_plan",
    "validate_role_access",
]
