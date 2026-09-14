"""Audit the complete assigned event inventory without capture retries or fitting."""
import argparse
from pathlib import Path
import time

from scripts import run_issue_76_event_development as run
from scripts.issue_76_event_targets import data_readiness, observed_target
from scripts import run_issue_76_display_continuation as display
from scripts import run_issue_76_wall_continuation as wall


def audit_group(root, plan, source, assignments, completed_bytes, check_time, amendment=None, wall_amendment=None):
    cases, rows = {}, []
    for assignment in assignments:
        check_time()
        member = run.metadata.materialize_assignment(source, assignment)
        identity = member["identity"]
        result, receipt, target, error = None, None, None, None
        try:
            result = run.files.read(root / "results" / (identity + ".json"))
            receipt = run.files.read(root / "supervision" / (identity + ".json"))
            if identity not in completed_bytes or receipt["attempt_artifact_bytes"] != completed_bytes[identity]:
                raise ValueError("capture is not bound to its sealed accounting record")
            if amendment is not None:
                display.validate_launch(root, amendment, assignment, result, receipt)
            if wall_amendment is not None:
                wall.validate_launch(root, wall_amendment, assignment, result, receipt)
            case = run.validation.read_case(root, plan, member)
            segment, = result["segments"]
            target = observed_target(segment["summary"], run.terminal_evidence(segment), case,
                                     plan["inventory"]["limits"]["native_step_seconds"])
            cases[identity] = case
        except (ValueError, KeyError, TypeError, OSError, EOFError) as failure:
            error = f"{type(failure).__name__}: {failure}"
        rows.append({"member_identity": identity,
                     **{key: member[key] for key in ("source_member_identity", "base_cluster",
                         "exposure_role", "generator_family", "candidate_ordinal", "original_action_reference")},
                     "action": assignment["action"], "capture_contract_validated": error is None,
                     "validation_error": error, "target": target,
                     "recorded_complete": None if result is None else result.get("complete"),
                     "recorded_failure": None if result is None else result.get("failure"),
                     "supervisor_receipt": receipt,
                     "decision_image_path": cases[identity]["decision_image_path"] if identity in cases else None})
        print(f"Audited event assignment {assignment['ordinal']}/{len(plan['inventory']['assignments'])}: "
              f"{identity}; valid={error is None}", flush=True)
    reference, = [row["identity"] for row in assignments if row["original_action_reference"]]
    comparisons = []
    for assignment in assignments:
        identity = assignment["identity"]
        if reference not in cases or identity not in cases:
            compared = {"comparable": False, "failures": ["capture_contract_not_validated"]}
        else:
            compared = run.previous.comparison.compare(cases[reference], cases[identity], plan["comparability_limits"])
        comparisons.append({"member_identity": identity, **compared})
    check_time()
    return {"source_member_identity": source["identity"], "base_cluster": source["base_cluster"],
            "exposure_role": source["exposure_role"], "generator_family": source["generator_family"],
            "comparisons": comparisons, "paired_ranking_admissible": all(c["comparable"] for c in comparisons),
            "rows": rows}


def validate(root=run.metadata.ROOT, output=run.metadata.OUTPUT):
    root, output = Path(root), Path(output)
    started = time.monotonic()
    plan = run.load_plan(root)
    inventory = plan["inventory"]
    budget = run.files.read(root / "capture-budget.json")
    if budget["running"] or budget["active_members"]:
        raise ValueError("event collection must be terminal before a full-inventory audit")
    amendment = display.load(root) if (root / display.FILENAME).exists() else None
    if budget.get("display_amendment_identity") and (
            amendment is None or budget["display_amendment_identity"] != amendment["identity"]):
        raise ValueError("capture budget lacks its display amendment")
    if amendment is not None and amendment["execution_plan_identity"] != plan["identity"]:
        raise ValueError("display amendment belongs to another execution")
    wall_amendment = wall.load(root) if (root / wall.FILENAME).exists() else None
    if budget.get("wall_amendment_identity") and (
            wall_amendment is None or budget["wall_amendment_identity"] != wall_amendment["identity"]):
        raise ValueError("capture budget lacks its wall-time amendment")
    if wall_amendment is not None and (wall_amendment["execution_plan_identity"] != plan["identity"]
            or amendment is None or wall_amendment["display_amendment_identity"] != amendment["identity"]):
        raise ValueError("wall-time amendment has a different execution or display parent")
    assigned = {row["identity"] for row in inventory["assignments"]}
    completed = set(budget["completed_attempt_bytes"])
    if not completed.issubset(assigned) or (not budget["stopped"] and completed != assigned):
        raise ValueError("event inventory is incomplete without an explicit resource stop")
    for folder in ("results", "supervision"):
        if {path.stem for path in (root / folder).glob("*.json")} - assigned:
            raise ValueError(f"unassigned {folder} exist")
    smoke = run.files.read(root / "smoke-validation.json")
    if not smoke["validated"] or smoke["execution_plan_identity"] != plan["identity"]:
        raise ValueError("validated execution-bound smoke evidence is missing")
    prior = smoke["validation_wall_seconds"]
    allowance = inventory["limits"]["offline_validation_wall_seconds"]
    resource_failure = run.resources.global_stop(inventory["limits"], active_seconds=budget["active_seconds"],
        smoke_seconds=budget["smoke_active_seconds"], rss_mib=budget["peak_cpu_rss_mib"],
        artifact_bytes=budget["artifact_bytes"], smoke=False)

    def check_time():
        if prior + time.monotonic() - started >= allowance:
            raise RuntimeError("event validation exceeded the combined offline allowance")

    check_time()
    start_path = root / "collection-validation-start.json"
    if start_path.exists():
        raise ValueError("full validation already attempted; audit existing work before repeating it")
    sources = ("scripts/validate_issue_76_event_collection.py", "scripts/issue_76_event_targets.py",
               "tests/test_issue_76_event_collection.py", "tests/test_issue_76_event_targets.py")
    run.immutable(start_path, {"execution_plan_identity": plan["identity"],
                  "prior_smoke_validation_seconds": prior, "offline_validation_allowance_seconds": allowance,
                  "source_text": {name: (run.files.ROOT / name).read_text() for name in sources}})
    groups = []
    for source in inventory["source_members"]:
        assignments = [row for row in inventory["assignments"] if row["source_member_identity"] == source["identity"]]
        group = audit_group(root, plan, source, assignments, budget["completed_attempt_bytes"], check_time,
                            amendment, wall_amendment)
        run.immutable(root / "validation-groups" / (source["identity"] + ".json"), group)
        groups.append(group)
    rows = [row for group in groups for row in group["rows"]]
    readiness = data_readiness(inventory, rows)
    # A stopped run can be described, but it cannot clear full-collection readiness.
    if budget["stopped"] or resource_failure or completed != assigned:
        readiness.update(data_ready=False, supports_separate_training_protocol_preparation=False)
    check_time()
    elapsed = time.monotonic() - started
    report = {"identity": "issue-76-event-collection-validation-v1", "execution_plan_identity": plan["identity"],
              "validation_start": run.files.read(start_path), "member_identities": [row["member_identity"] for row in rows],
              "all_assigned_records_audited": True, "capture_inventory_completed": completed == assigned,
              "collection_resource_failure": resource_failure,
              "display_amendment_identity": None if amendment is None else amendment["identity"],
              "wall_amendment_identity": None if wall_amendment is None else wall_amendment["identity"],
              "unattempted_member_identities": sorted(assigned - completed),
              "validated_captures": sum(row["capture_contract_validated"] for row in rows),
              "groups": groups, "data_readiness": readiness, "capture_budget": budget,
              "validation_and_target_wall_seconds": elapsed, "prior_smoke_validation_seconds": prior,
              "remaining_offline_validation_seconds": allowance - prior - elapsed,
              "raw_results_rewritten": False, "new_optimizer_updates": 0,
              "model_training_authorized": False, "fresh_access": False, "advancement_authorized": False}
    run.immutable(root / "collection-validation.json", report)
    run.immutable(output / "collection-validation.json", report)
    print({"validated_captures": report["validated_captures"], "data_ready": readiness["data_ready"]}, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    if parser.parse_args().validate:
        validate()
    else:
        print("No-write: --validate audits a terminal collection; no capture, fitting or fresh access.")
