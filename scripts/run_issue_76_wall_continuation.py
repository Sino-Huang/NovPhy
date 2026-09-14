"""Source-bound wall-time increase for unattempted event captures only."""
import argparse
from copy import deepcopy
from pathlib import Path

from scripts import run_issue_76_display_continuation as display

base = display.base
IDENTITY = "issue-76-event-wall-continuation-v1"
FILENAME = "wall-continuation.json"
OVERRIDES = {"shot_seconds": 360, "attempt_seconds": 600}
SOURCES = ("scripts/run_issue_76_wall_continuation.py", "tests/test_issue_76_wall_continuation.py",
           "docs/issue-76-event-wall-time-amendment.md")


def effective_plan(plan):
    result = deepcopy(plan)
    result["inventory"]["limits"].update(OVERRIDES)
    return result


def prepare(root=base.metadata.ROOT, output=base.metadata.OUTPUT):
    root, output = Path(root), Path(output)
    plan = base.load_plan(root)
    prior_display = display.load(root)
    budget = base.files.read(root / "capture-budget.json")
    if (budget["running"] or budget["active_members"] or not budget["stopped"]
            or budget["stop_reason"] != "supervisor_KeyboardInterrupt: "
            or budget["display_amendment_identity"] != prior_display["identity"]):
        raise ValueError("wall continuation requires the audited idle display-continuation boundary")
    if Path(f"/proc/{budget['supervisor_pid']}").exists():
        raise ValueError("previous supervisor is still present")
    base.resume_inventory(root, plan, {**budget, "stopped": False})
    failure = base.resources.global_stop(plan["inventory"]["limits"], active_seconds=budget["active_seconds"],
        smoke_seconds=budget["smoke_active_seconds"], rss_mib=budget["peak_cpu_rss_mib"],
        artifact_bytes=budget["artifact_bytes"], smoke=False)
    if failure:
        raise ValueError(f"wall amendment cannot erase a global resource stop: {failure}")
    for slot in range(plan["inventory"]["limits"]["workers"]):
        base.resources.check_ports_available(base.resources.worker_ports(slot))
    assignments = {row["identity"]: row for row in plan["inventory"]["assignments"]}
    prior = {}
    for identity in budget["completed_attempt_bytes"]:
        result = base.files.read(root / "results" / (identity + ".json"))
        receipt = base.files.read(root / "supervision" / (identity + ".json"))
        if receipt["stop_reason"]:
            raise ValueError("interrupted or resource-stopped worker needs a separate audit")
        display.validate_launch(root, prior_display, assignments[identity], result, receipt)
        prior[identity] = {"result": result, "receipt": receipt}
    remaining = [identity for identity in assignments if identity not in prior]
    if set(remaining) != set(budget["unattempted_member_identities"]):
        raise ValueError("remaining inventory differs from terminal accounting")
    value = {"identity": IDENTITY, "execution_plan_identity": plan["identity"],
             "display_amendment_identity": prior_display["identity"], "boundary_budget": budget,
             "prior_records": prior, "remaining_member_identities": remaining, "wall_time_overrides": OVERRIDES,
             "capture_retries": 0, "model_training_authorized": False, "fresh_access": False,
             "advancement_authorized": False,
             "source_text": {name: (base.files.ROOT / name).read_text() for name in SOURCES}}
    base.immutable(root / FILENAME, value)
    base.immutable(output / FILENAME, value)
    print(f"Wall amendment frozen: {len(prior)} retained, {len(remaining)} unattempted; no capture started.", flush=True)
    return value


def load(root):
    value = base.files.read(Path(root) / FILENAME)
    if (value["identity"] != IDENTITY or value["wall_time_overrides"] != OVERRIDES
            or value["capture_retries"] or value["model_training_authorized"]
            or value["fresh_access"] or value["advancement_authorized"]):
        raise ValueError("wall amendment exceeds its declared scope")
    for name, source in value["source_text"].items():
        if (base.files.ROOT / name).read_text() != source:
            raise ValueError(f"wall continuation source changed: {name}")
    return value


def launch_record(amendment, assignment, slot):
    if assignment["identity"] not in amendment["remaining_member_identities"] or assignment["identity"] in amendment["prior_records"]:
        raise ValueError("wall continuation cannot replay an earlier assignment")
    return {"amendment_identity": amendment["identity"], "member_identity": assignment["identity"],
            "execution_plan_identity": amendment["execution_plan_identity"], "worker_slot": slot,
            "wall_time_overrides": amendment["wall_time_overrides"], "action": assignment["action"],
            "source_member_identity": assignment["source_member_identity"]}


def worker(root, source, assignment, limits, slot):
    amendment = load(root)
    if any(limits[key] != value for key, value in OVERRIDES.items()):
        raise ValueError("worker wall-time allowance differs from amendment")
    record = launch_record(amendment, assignment, slot)
    base.immutable(Path(root) / "wall-launches" / (assignment["identity"] + ".json"), record)
    return display.worker(root, source, assignment, limits, slot)


def validate_launch(root, amendment, assignment, result, receipt):
    identity = assignment["identity"]
    if identity in amendment["prior_records"]:
        if amendment["prior_records"][identity] != {"result": result, "receipt": receipt}:
            raise ValueError("pre-wall-amendment result or receipt changed")
    else:
        expected = launch_record(amendment, assignment, receipt["worker_slot"])
        if base.files.read(Path(root) / "wall-launches" / (identity + ".json")) != expected:
            raise ValueError("capture lacks its exact wall-time amendment binding")
        if receipt["wall_seconds"] >= OVERRIDES["attempt_seconds"]:
            raise ValueError("capture exceeded amended attempt wall allowance")


def run(root=base.metadata.ROOT, output=base.metadata.OUTPUT):
    root, output = Path(root), Path(output)
    plan = base.load_plan(root)
    prior_display = display.load(root)
    amendment = load(root)
    if (amendment["execution_plan_identity"] != plan["identity"]
            or amendment["display_amendment_identity"] != prior_display["identity"]):
        raise ValueError("wall amendment has a different execution or display parent")
    budget = base.files.read(root / "capture-budget.json")
    if budget != amendment["boundary_budget"]:
        raise ValueError("wall boundary budget changed; audit before continuation")
    for identity, record in amendment["prior_records"].items():
        if (base.files.read(root / "results" / (identity + ".json")) != record["result"]
                or base.files.read(root / "supervision" / (identity + ".json")) != record["receipt"]):
            raise ValueError("earlier result or receipt changed")
    base.resume_inventory(root, plan, {**budget, "stopped": False})
    marker = root / "wall-continuation-start.json"
    if marker.exists():
        raise ValueError("wall continuation already attempted; no automatic restart")
    base.immutable(marker, {"amendment_identity": amendment["identity"],
                           "prior_active_seconds": budget["active_seconds"],
                           "prior_completed_attempts": len(budget["completed_attempt_bytes"])})
    continued = {**budget, "stopped": False, "wall_amendment_identity": amendment["identity"]}
    continued.pop("stop_reason")
    base.resources.live._replace_json(continued, root / "capture-budget.json")
    previous = base.resources.worker
    base.resources.worker = worker
    try:
        return base.run(effective_plan(plan), root, output, smoke=False)
    finally:
        base.resources.worker = previous


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.run:
        run()
    else:
        print("No-write: prepare the wall-only amendment before continuing unattempted captures.")
