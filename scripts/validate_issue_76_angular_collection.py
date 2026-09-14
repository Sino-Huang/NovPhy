"""Validate the entire angular pilot, retaining failed assignments and lineages."""
import argparse
import gzip
import json
from pathlib import Path
import time

from scripts import run_issue_76_angular_replay as angular
from scripts.issue_76_native_outcomes import terminal_evidence
from scripts.issue_76_replay_headroom import endpoint_counts, summarize


def failed_outcome(reason):
    return {"active_pigs": None, "active_blocks": None, "observed_count_cost": None,
            "terminal_reason": None, "settled_count_cost": None,
            "ranking_failure": True, "ranking_cost": 1e9, "native_level_clear": False,
            "invalid_outcome_reason": reason}


def outcome(result):
    segment, = result["segments"]
    terminal = terminal_evidence(segment)
    native = Path(segment["native_root"])
    manifest = angular.supervisor.files.read(native / "native-manifest.json")
    with gzip.open(native / manifest["chunks"][-1]["path"], "rt") as stream:
        sample = json.load(stream)["fixed_step_samples"][-1]
    if sample["fixed_step"] != manifest["last_fixed_step"]:
        raise ValueError("outcome endpoint differs from validated native endpoint")
    value = {"terminal_evidence": terminal, "fixed_step": sample["fixed_step"],
             "native_root": str(native), **endpoint_counts(sample, terminal)}
    if segment["summary"]["censored"]:
        value.update(settled_count_cost=None, ranking_failure=True, ranking_cost=1e9,
                     native_level_clear=False, invalid_outcome_reason="native_time_window_limit")
    return value


def group_report(members, cases, rows, limits):
    groups = []
    for source in dict.fromkeys(member["source_member_identity"] for member in members):
        assigned = [member for member in members if member["source_member_identity"] == source]
        reference, = [member for member in assigned if member["original_action_reference"]]
        comparisons = []
        for member in assigned:
            if reference["identity"] not in cases or member["identity"] not in cases:
                compared = {"comparable": False, "failures": ["capture_contract_not_validated"]}
            else:
                compared = angular.validation.comparison.compare(
                    cases[reference["identity"]], cases[member["identity"]], limits)
            comparisons.append({"member_identity": member["identity"], **compared})
        selected = [row for row in rows if row["source_member_identity"] == source]
        groups.append({"source_member_identity": source, "comparisons": comparisons,
                       "paired_ranking_admissible": all(row["comparable"] for row in comparisons),
                       "rows": selected, **summarize(selected)})
    return groups


def viability(groups):
    comparable = sum(group["paired_ranking_admissible"] for group in groups)
    clear = sum(group["paired_ranking_admissible"] and group["native_level_clears"] > 0
                for group in groups)
    return {"comparable_lineages": comparable, "comparable_lineages_with_grid_clear": clear,
            "action_design_viable": comparable >= 4 and clear >= 3,
            "criterion": "at least four comparable lineages and three with a native-clear grid action",
            "training_authorized": False, "fresh_access": False, "advancement_authorized": False}


def validate(root=angular.metadata.ROOT, output=angular.metadata.OUTPUT):
    started = time.monotonic()
    plan = angular.load_plan(root)
    files = angular.supervisor.files
    budget = files.read(root / "capture-budget.json")
    limits = plan["inventory"]["limits"]
    if budget["running"] or budget["stopped"] or angular.supervisor.stop_reason(limits, budget, 0, smoke=False):
        raise ValueError("angular collection has not finished within its resource envelope")
    members = plan["inventory"]["members"]
    expected = {member["identity"] for member in members}
    for folder in ("results", "supervision"):
        if {path.stem for path in (root / folder).glob("*.json")} != expected:
            raise ValueError(f"{folder} does not contain the full assigned inventory")
    correction = files.read(root / "accounting-correction.json")
    failed_id = correction["failed_result_preserved"]["member_identity"]
    for folder, key in (("results", "failed_result_preserved"),
                        ("supervision", "failed_supervisor_receipt_preserved")):
        if files.read(root / folder / (failed_id + ".json")) != correction[key]:
            raise ValueError("the audited failed assignment was changed")
    prior = files.read(root / "smoke-validation.json")["validation_wall_seconds"]

    def check_time():
        if prior + time.monotonic() - started > limits["offline_preparation_wall_seconds"]:
            raise ValueError("angular validation/outcomes exceeded their combined CPU wall allowance")

    cases, rows = {}, []
    for member in members:
        check_time()
        identity = member["identity"]
        result = files.read(root / "results" / (identity + ".json"))
        receipt = files.read(root / "supervision" / (identity + ".json"))
        error = None
        try:
            case = angular.validation.read_case(root, plan, member)
            values = outcome(result)
            cases[identity] = case
        except ValueError as failure:
            error = str(failure)
            values = failed_outcome(error)
        rows.append({"member_identity": identity,
                     **{key: member[key] for key in
                        ("source_member_identity", "candidate_ordinal", "original_action_reference")},
                     "action": member["actions"][0], "capture_contract_validated": error is None,
                     "validation_error": error, "recorded_complete": result["complete"],
                     "recorded_failure": result["failure"], "supervisor_receipt": receipt, **values})
        print(f"Audited angular assignment {len(rows)}/{len(members)}: {identity}; valid={error is None}", flush=True)
    groups = group_report(members, cases, rows, plan["comparability_limits"])
    check_time()
    elapsed = time.monotonic() - started
    report = {"identity": "issue-76-angular-collection-validation-v1",
              "execution_plan_identity": plan["identity"],
              "source_text": {name: (files.ROOT / name).read_text() for name in
                              ("scripts/validate_issue_76_angular_collection.py",
                               "tests/test_issue_76_angular_collection.py")},
              "member_identities": [member["identity"] for member in members],
              "all_assigned_records_audited": True, "validated_captures": len(cases),
              "all_capture_contracts_validated": len(cases) == len(members),
              "groups": groups, "viability": viability(groups), "capture_budget": budget,
              "accounting_correction_identity": correction["identity"],
              "validation_and_outcome_wall_seconds": elapsed, "prior_smoke_validation_seconds": prior,
              "remaining_offline_preparation_seconds": limits["offline_preparation_wall_seconds"] - prior - elapsed,
              "decision_image_paths": {key: case["decision_image_path"] for key, case in cases.items()},
              "model_scores_measured": False, "new_optimizer_updates": 0,
              "fresh_access": False, "advancement_authorized": False}
    angular.validation.immutable(root / "collection-validation.json", report)
    angular.validation.immutable(output / "collection-validation.json", report)
    print(report["viability"], flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    if parser.parse_args().validate:
        validate()
    else:
        print("No-write: --validate audits all completed angular assignments; no scoring or training.")
