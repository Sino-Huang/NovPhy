"""One complete accounting rescan for the observed atomic-publication rename."""
import argparse
from pathlib import Path

from scripts import run_issue_76_angular_replay as angular

original_counter = angular.supervisor.artifact_bytes
SOURCES = ("scripts/issue_76_angular_accounting_correction.py",
           "tests/test_issue_76_angular_accounting_correction.py")


def artifact_bytes(root):
    try:
        return original_counter(root)
    except FileNotFoundError as error:
        if not any(part.startswith(".observation-trace.") for part in Path(error.filename or "").parts):
            raise
        # Discard the partial sum. Count the committed directory on a new scan;
        # never substitute zero or retry the physics capture.
        return original_counter(root)


def prepare(root=angular.metadata.ROOT, output=angular.metadata.OUTPUT):
    plan = angular.load_plan(root)
    files = angular.supervisor.files
    budget = files.read(root / "capture-budget.json")
    failed_id = "issue-76-angular-replay-071-a04"
    failure = files.read(root / "results" / (failed_id + ".json"))
    receipt = files.read(root / "supervision" / (failed_id + ".json"))
    if (budget["running"] or budget["stopped"] or failure["complete"]
            or not failure["failure"].startswith("supervisor_FileNotFoundError")
            or receipt["worker_exitcode"] != -15):
        raise ValueError("correction does not match the audited terminal failure")
    recorded = sorted(path.stem for path in (root / "results").glob("*.json"))
    unattempted = [member["identity"] for member in plan["inventory"]["members"] if member["identity"] not in recorded]
    if len(recorded) != 18 or len(unattempted) != 47:
        raise ValueError("correction requires the exact 18 retained and 47 unattempted assignments")
    value = {"identity": "issue-76-angular-accounting-correction-v1",
             "execution_plan_identity": plan["identity"], "capture_budget_before_continuation": budget,
             "failed_result_preserved": failure, "failed_supervisor_receipt_preserved": receipt,
             "recorded_member_identities": recorded, "unattempted_member_identities": unattempted,
             "source_text": {name: (files.ROOT / name).read_text() for name in SOURCES},
             "capture_retries": 0, "limits_changed": False, "assignments_changed": False,
             "accounting_rescans_per_call_max": 1, "fresh_access": False}
    angular.validation.immutable(root / "accounting-correction.json", value)
    angular.validation.immutable(output / "accounting-correction.json", value)
    print("Accounting correction archived; failed case retained; no capture started.")


def run(root=angular.metadata.ROOT):
    plan = angular.load_plan(root)
    files = angular.supervisor.files
    correction = files.read(root / "accounting-correction.json")
    for name, source in correction["source_text"].items():
        if (files.ROOT / name).read_text() != source:
            raise ValueError("accounting correction source changed")
    if (correction["execution_plan_identity"] != plan["identity"]
            or files.read(root / "capture-budget.json") != correction["capture_budget_before_continuation"]):
        raise ValueError("continuation requires its unchanged audited budget")
    failed_id = correction["failed_result_preserved"]["member_identity"]
    if (files.read(root / "results" / (failed_id + ".json")) != correction["failed_result_preserved"]
            or files.read(root / "supervision" / (failed_id + ".json")) != correction["failed_supervisor_receipt_preserved"]):
        raise ValueError("failed assignment was changed")
    recorded = sorted(path.stem for path in (root / "results").glob("*.json"))
    if recorded != correction["recorded_member_identities"]:
        raise ValueError("continuation membership changed")
    angular.supervisor.artifact_bytes = artifact_bytes
    try:
        angular.supervisor.run(plan, root, smoke=False)
    finally:
        angular.supervisor.artifact_bytes = original_counter


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
        print("No-write: explicit accounting correction, never a capture retry.")
