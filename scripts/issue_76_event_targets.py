"""Observed event summaries and the frozen development-data readiness rule.

Call only after individual capture validation. These summaries neither infer
events after a trace ends nor authorize model fitting or fresh evaluation.
"""
from collections import Counter


def observed_target(summary, terminal, case, step_seconds):
    first, last = summary["first_fixed_step"], summary["last_fixed_step"]
    if not summary["observed_window_valid"] or last < first:
        raise ValueError("event target requires an intact observed interval")
    if terminal is None:
        if not summary["censored"] or summary["terminal_observed"]:
            raise ValueError("nonterminal event target must be intact right-censoring")
        kind, event_seconds = "right_censored", None
    else:
        if (summary["censored"] or not summary["terminal_observed"]
                or terminal["fixed_step"] != last):
            raise ValueError("event target terminal differs from its observed endpoint")
        kind = {"level_clear": "native_clear", "level_fail": "native_fail",
                "stable_entered": "stable_without_clear"}[terminal["reason"]]
        event_seconds = (terminal["fixed_step"] - first) * step_seconds
    gap = case["capture_time"] - case["decision_time"]
    duration = (last - first) * step_seconds
    return {"stop_kind": kind, "native_level_clear": kind == "native_clear",
            "terminal_evidence": terminal, "first_fixed_step": first, "last_fixed_step": last,
            "native_step_seconds": step_seconds,
            "decision_time_seconds": case["decision_time"],
            "capture_start_time_seconds": case["capture_time"],
            "decision_to_capture_seconds": gap, "observed_capture_seconds": duration,
            "observed_end_seconds_from_decision": gap + duration,
            "terminal_seconds_from_capture": event_seconds,
            "terminal_seconds_from_decision": None if event_seconds is None else gap + event_seconds,
            "later_clear_by_deadline_label": None}


def data_readiness(inventory, rows):
    """Count every assignment; multiple clear actions do not multiply lineages."""
    expected = {row["identity"] for row in inventory["assignments"]}
    if len(rows) != len(expected) or {row["member_identity"] for row in rows} != expected:
        raise ValueError("readiness requires exactly one audit row per assigned capture")
    assignments = {row["identity"]: row for row in inventory["assignments"]}
    sources = {source["identity"]: source for source in inventory["source_members"]}
    cells = {}
    clear_by_role = {}
    for source in sources.values():
        cells.setdefault((source["exposure_role"], source["generator_family"]), [])
        clear_by_role.setdefault(source["exposure_role"], set())
    for row in rows:
        source = sources[assignments[row["member_identity"]]["source_member_identity"]]
        role = source["exposure_role"]
        cells[role, source["generator_family"]].append(row)
        if row["capture_contract_validated"] and row["target"]["native_level_clear"]:
            clear_by_role[role].add(source["base_cluster"])
    rule = inventory["data_readiness"]
    summaries = []
    for (role, family), assigned in cells.items():
        valid = [row for row in assigned if row["capture_contract_validated"]]
        fraction = len(valid) / len(assigned)
        summaries.append({"exposure_role": role, "generator_family": family,
                          "assigned": len(assigned), "valid": len(valid),
                          "unavailable": len(assigned) - len(valid), "valid_fraction": fraction,
                          "valid_fraction_passed": fraction >= rule["minimum_valid_capture_fraction_each_role_family"],
                          "stop_counts": dict(Counter(row["target"]["stop_kind"] for row in valid))})
    minimum_keys = {"training": "minimum_clear_training_lineages",
                    "calibration": "minimum_clear_calibration_lineages",
                    "model_selection": "minimum_clear_model_selection_lineages"}
    roles = [{"exposure_role": role, "clear_base_clusters": sorted(clear_by_role[role]),
              "clear_lineages": len(clear_by_role[role]), "required": rule[key],
              "passed": len(clear_by_role[role]) >= rule[key]} for role, key in minimum_keys.items()]
    passed = all(cell["valid_fraction_passed"] for cell in summaries) and all(role["passed"] for role in roles)
    return {"cells": summaries, "roles": roles, "data_ready": passed,
            "supports_separate_training_protocol_preparation": passed,
            "paired_ranking_admissibility_used_for_individual_supervision": False,
            "model_training_authorized": False, "fresh_access": False, "advancement_authorized": False}
