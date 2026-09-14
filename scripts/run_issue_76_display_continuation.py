"""One audited display-only transition for the existing event inventory."""
import argparse
from pathlib import Path

from scripts import run_issue_76_event_development as base
from scripts.issue_76_event_resources import worker as original_worker
from scripts.issue_76_display_start import start_display

IDENTITY = "issue-76-event-display-continuation-v1"
FILENAME = "display-continuation.json"
SOURCES = ("scripts/run_issue_76_display_continuation.py", "scripts/issue_76_display_start.py",
           "tests/test_issue_76_display_continuation.py", "tests/test_issue_76_display_start.py")


def prepare(root=base.metadata.ROOT, output=base.metadata.OUTPUT):
    root, output = Path(root), Path(output)
    plan = base.load_plan(root)
    budget = base.files.read(root / "capture-budget.json")
    if (budget["running"] or budget["active_members"] or not budget["stopped"]
            or budget["stop_reason"] != "supervisor_KeyboardInterrupt: "):
        raise ValueError("display continuation requires the audited idle KeyboardInterrupt boundary")
    if Path(f"/proc/{budget['supervisor_pid']}").exists():
        raise ValueError("original supervisor is still present")
    base.resume_inventory(root, plan, {**budget, "stopped": False})
    limits = plan["inventory"]["limits"]
    failure = base.resources.global_stop(limits, active_seconds=budget["active_seconds"],
        smoke_seconds=budget["smoke_active_seconds"], rss_mib=budget["peak_cpu_rss_mib"],
        artifact_bytes=budget["artifact_bytes"], smoke=False)
    if failure:
        raise ValueError(f"display correction cannot erase a resource stop: {failure}")
    for slot in range(limits["workers"]):
        base.resources.check_ports_available(base.resources.worker_ports(slot))
    prior = {}
    for identity in budget["completed_attempt_bytes"]:
        receipt = base.files.read(root / "supervision" / (identity + ".json"))
        if receipt["stop_reason"]:
            raise ValueError("boundary includes interrupted or resource-stopped workers; separate audit required")
        prior[identity] = {"receipt": receipt, "result": base.files.read(root / "results" / (identity + ".json"))}
    remaining = [row["identity"] for row in plan["inventory"]["assignments"] if row["identity"] not in prior]
    if set(remaining) != set(budget["unattempted_member_identities"]):
        raise ValueError("unattempted inventory differs from the terminal budget")
    value = {"identity": IDENTITY, "execution_plan_identity": plan["identity"],
             "boundary_budget": budget, "prior_records": prior, "remaining_member_identities": remaining,
             "only_change": "consume the complete newline-terminated Xvnc display-number reply",
             "capture_retries": 0, "model_training_authorized": False, "fresh_access": False,
             "advancement_authorized": False,
             "source_text": {name: (base.files.ROOT / name).read_text() for name in SOURCES}}
    base.immutable(root / FILENAME, value)
    base.immutable(output / FILENAME, value)
    print(f"Display amendment frozen: {len(prior)} retained, {len(remaining)} unattempted; budget unchanged.", flush=True)
    return value


def load(root):
    value = base.files.read(Path(root) / FILENAME)
    if (value["identity"] != IDENTITY or value["capture_retries"] != 0
            or value["model_training_authorized"] or value["fresh_access"] or value["advancement_authorized"]):
        raise ValueError("display amendment scope differs")
    for name, source in value["source_text"].items():
        if (base.files.ROOT / name).read_text() != source:
            raise ValueError(f"display continuation source changed: {name}")
    return value


def launch_record(amendment, assignment, slot):
    identity = assignment["identity"]
    if identity not in amendment["remaining_member_identities"] or identity in amendment["prior_records"]:
        raise ValueError("display continuation cannot replay an earlier assignment")
    return {"amendment_identity": amendment["identity"], "execution_plan_identity": amendment["execution_plan_identity"],
            "member_identity": identity, "worker_slot": slot, "corrected_display_helper": True,
            "action": assignment["action"], "source_member_identity": assignment["source_member_identity"]}


def worker(root, source, assignment, limits, slot):
    amendment = load(root)
    record = launch_record(amendment, assignment, slot)
    # Keep this outside the attempt: play_episode itself creates that directory
    # exclusively, which remains the no-replay guard for native capture.
    base.immutable(Path(root) / "display-launches" / (assignment["identity"] + ".json"), record)
    previous = base.resources.live.start_display
    base.resources.live.start_display = start_display
    try:
        return original_worker(root, source, assignment, limits, slot)
    finally:
        base.resources.live.start_display = previous


def validate_launch(root, amendment, assignment, result, receipt):
    """Keep original evidence exact and bind later captures to the amendment."""
    identity = assignment["identity"]
    if identity in amendment["prior_records"]:
        if amendment["prior_records"][identity] != {"result": result, "receipt": receipt}:
            raise ValueError("pre-amendment capture or receipt changed")
    else:
        expected = launch_record(amendment, assignment, receipt["worker_slot"])
        if base.files.read(Path(root) / "display-launches" / (identity + ".json")) != expected:
            raise ValueError("corrected capture lacks its exact display-amendment binding")


def run(root=base.metadata.ROOT, output=base.metadata.OUTPUT):
    root, output = Path(root), Path(output)
    plan = base.load_plan(root)
    amendment = load(root)
    if amendment["execution_plan_identity"] != plan["identity"]:
        raise ValueError("display amendment belongs to a different execution")
    budget = base.files.read(root / "capture-budget.json")
    if budget != amendment["boundary_budget"]:
        raise ValueError("boundary budget changed; audit before any further continuation")
    for identity, record in amendment["prior_records"].items():
        if (base.files.read(root / "results" / (identity + ".json")) != record["result"]
                or base.files.read(root / "supervision" / (identity + ".json")) != record["receipt"]):
            raise ValueError("an earlier capture or receipt changed")
    base.resume_inventory(root, plan, {**budget, "stopped": False})
    started = root / "display-continuation-start.json"
    if started.exists():
        raise ValueError("display continuation already attempted; no automatic restart")
    base.immutable(started, {"amendment_identity": amendment["identity"],
                           "prior_active_seconds": budget["active_seconds"],
                           "prior_completed_attempts": len(budget["completed_attempt_bytes"])})
    continued = {**budget, "stopped": False, "display_amendment_identity": amendment["identity"]}
    continued.pop("stop_reason")
    base.resources.live._replace_json(continued, root / "capture-budget.json")
    previous = base.resources.worker
    base.resources.worker = worker
    try:
        return base.run(plan, root, output, smoke=False)
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
        print("No-write: prepare and audit the display-only transition before continuing unattempted assignments.")
