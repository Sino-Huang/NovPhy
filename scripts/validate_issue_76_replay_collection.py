"""Validate every assigned replay with the unchanged prospective smoke checks."""
import argparse
import time

from scripts import validate_issue_76_action_replay as smoke


def compare_groups(members, cases, limits):
    groups = []
    for source in dict.fromkeys(member["source_member_identity"] for member in members):
        assigned = [member for member in members if member["source_member_identity"] == source]
        reference, = [member for member in assigned if member["original_action_reference"]]
        comparisons = []
        for member in assigned:
            result = smoke.comparison.compare(cases[reference["identity"]], cases[member["identity"]], limits)
            comparisons.append({"member_identity": member["identity"],
                                "original_action_reference": member["original_action_reference"],
                                **result})
        groups.append({"source_member_identity": source,
                       "paired_ranking_admissible": all(row["comparable"] for row in comparisons),
                       "comparisons": comparisons})
    return groups


def validate(root=smoke.run.ROOT):
    started = time.monotonic()
    plan = smoke.run.load_plan(root)
    members = plan["inventory"]["members"]
    budget = smoke.run.files.read(root / "capture-budget.json")
    limits = plan["inventory"]["limits"]
    if budget["running"] or budget["stopped"] or smoke.run.stop_reason(limits, budget, 0, smoke=False):
        raise ValueError("collection did not finish cleanly within its resource envelope")
    expected = {member["identity"] for member in members}
    for folder in ("results", "supervision"):
        if {path.stem for path in (root / folder).glob("*.json")} != expected:
            raise ValueError(f"{folder} inventory differs from all assigned members")
    cases = {}
    for member in members:
        cases[member["identity"]] = smoke.read_case(root, plan, member)
        print(f"Validated capture {len(cases)}/{len(members)}: {member['identity']}", flush=True)
        if time.monotonic() - started > limits["offline_preparation_wall_seconds"]:
            raise ValueError("full validation exceeded offline preparation allowance")
    groups = compare_groups(members, cases, plan["comparability_limits"])
    report = {"identity": "issue-76-replay-collection-validation-v1",
              "execution_plan_identity": plan["identity"],
              "source_text": {"scripts/validate_issue_76_replay_collection.py":
                              (smoke.run.files.ROOT / "scripts/validate_issue_76_replay_collection.py").read_text()},
              "member_identities": [member["identity"] for member in members],
              "all_capture_contracts_validated": True, "groups": groups,
              "all_groups_comparable": all(group["paired_ranking_admissible"] for group in groups),
              "capture_budget": budget, "validation_wall_seconds": time.monotonic() - started,
              "decision_image_paths": {key: case["decision_image_path"] for key, case in cases.items()},
              "model_scoring_authorized": False, "new_optimizer_updates": 0,
              "fresh_access": False, "advancement_authorized": False}
    smoke.immutable(root / "collection-validation.json", report)
    smoke.immutable(smoke.OUTPUT / "collection-validation.json", report)
    print({"captures_validated": len(cases),
           "comparable_groups": sum(group["paired_ranking_admissible"] for group in groups),
           "total_groups": len(groups), "validation_wall_seconds": report["validation_wall_seconds"]}, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    if parser.parse_args().validate:
        validate()
    else:
        print("No-write: use --validate to check the completed collection; no model scoring.")
