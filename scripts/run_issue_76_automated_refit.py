"""Explicit execution-budget amendment around the unchanged native refit."""
import argparse
from contextlib import contextmanager
from pathlib import Path

import torch

from scripts import run_issue_76_native_refit as fit
from scripts import issue_76_reserve_fit_cost as engineering


IDENTITY = "issue-76-automated-execution-v1"
AMENDMENT = "execution-amendment-v1.json"
SOURCES = ("scripts/run_issue_76_automated_refit.py",
           "docs/issue-76-automated-execution-amendment.md")
SECONDS = 24 * 3600


def budget_keys(plan):
    keys = ["data-preparation"]
    for seed in plan["seeds"]:
        keys.append(f"common-{seed}")
        for arm in ("hybrid", "pure"):
            keys.extend((f"predictor-{arm}-{seed}", f"controller-{arm}-{seed}"))
    return keys


def non_attempt_bytes(collection):
    """The finished collector already counted every file under attempts/."""
    total = 0
    for path in collection.iterdir():
        if path.name == "attempts":
            continue
        if path.is_file():
            total += path.stat().st_size
        elif path.is_dir():
            total += sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return total


def prepare_amendment(root, plan):
    target = root / AMENDMENT
    if target.exists():
        raise ValueError("execution amendment already exists; validate it, do not reset budgets")
    collection = Path(plan["collection_root"])
    if any(not fit.collection.c2.result_path(collection, m).exists() for m in plan["members"]):
        raise ValueError("finish the collection before amending this refit")
    before = {key: fit.files.read(root / "budgets" / (key + ".json")) for key in budget_keys(plan)}
    if any(v.get("running") or v["stopped"] for v in before.values()):
        raise ValueError("audit a running, unclean or stopped job before any budget amendment")
    reservation = fit.files.read(root / "engineering-cost-reservation.json")
    if reservation != engineering.reservation(plan):
        raise ValueError("engineering costs differ from the original reservation")
    capture_budget = fit.files.read(collection / "budget.json")
    value = {"schema": "issue_76_execution_amendment_v1", "identity": IDENTITY,
             "base_plan_identity": plan["identity"],
             "authorization": "2026-09-13 operator: fully automated workflow and budget changes toward a genuine #76 pass",
             "source_text": {p: (fit.files.ROOT / p).read_text() for p in SOURCES},
             "before_budgets": before, "seconds_per_job": SECONDS,
             "collection_budget": capture_budget,
             "collection_bytes": capture_budget["artifact_bytes"] + non_attempt_bytes(collection),
             "optimizer_updates_changed": False, "fresh_access": False,
             "advancement_authorized": False}
    fit.files.write(target, value)
    for key, previous in before.items():
        fit.files.write(root / "budgets" / (key + ".json"),
                        {**previous, "limit_seconds": SECONDS, "execution_amendment": IDENTITY})
    return value


def load_amendment(root, plan):
    value = fit.files.read(root / AMENDMENT)
    if (value["identity"] != IDENTITY or value["base_plan_identity"] != plan["identity"]
            or value["seconds_per_job"] != SECONDS
            or value["source_text"] != {p: (fit.files.ROOT / p).read_text() for p in SOURCES}):
        raise ValueError("execution amendment source or base plan changed")
    if value["collection_budget"] != fit.files.read(Path(plan["collection_root"]) / "budget.json"):
        raise ValueError("completed collection changed after the execution amendment")
    for key in budget_keys(plan):
        current = fit.files.read(root / "budgets" / (key + ".json"))
        previous = value["before_budgets"][key]
        if (current["limit_seconds"] != SECONDS or current.get("execution_amendment") != IDENTITY
                or current["active_seconds"] < previous["active_seconds"]
                or current.get("engineering_seconds") != previous.get("engineering_seconds")):
            raise ValueError("amended budget lost its allowance, provenance or prior cost")
    return value


@contextmanager
def execution_policy(root, amendment):
    """Only the two declared execution operations differ from frozen v2."""
    original_budget, original_size = fit.fitting_budget, fit.working_bytes

    def amended_budget(job_root, key, original_seconds, device):
        if Path(job_root) != root or key not in amendment["before_budgets"]:
            raise ValueError("job is outside the execution amendment")
        return original_budget(job_root, key, amendment["seconds_per_job"], device)

    def working_bytes(job_root, plan):
        if Path(job_root) != root:
            raise ValueError("working-data root differs from the execution amendment")
        return amendment["collection_bytes"] + sum(
            p.stat().st_size for p in root.rglob("*") if p.is_file())

    fit.fitting_budget, fit.working_bytes = amended_budget, working_bytes
    try:
        yield
    finally:
        fit.fitting_budget, fit.working_bytes = original_budget, original_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare-amendment", "dry-run", "run-refit", "prepare-data", "fit-common",
                 "fit-predictors", "fit-controllers", "diagnose", "validate"):
        modes.add_argument("--" + name, action="store_true")
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    args = parser.parse_args()
    torch.set_num_threads(1)
    root, plan = fit.ROOT, fit.load_plan()
    if args.prepare_amendment:
        prepare_amendment(root, plan)
        fit.log("execution amendment recorded: 24h/job; all prior costs and update schedules retained")
        return
    amendment = load_amendment(root, plan)
    if args.dry_run:
        fit.log("no-write execution amendment check passed; no fit or fresh access")
        return
    try:
        with execution_policy(root, amendment):
            if args.prepare_data or args.run_refit:
                fit.prepare_data(root, plan)
            if args.fit_common or args.run_refit:
                fit.train_common(root, plan, args.device)
            if args.fit_predictors or args.run_refit:
                fit.train_predictors(root, plan, args.device)
            if args.fit_controllers or args.run_refit:
                fit.train_controllers(root, plan, args.device)
            if args.diagnose or args.run_refit:
                fit.diagnose(root, plan, args.device)
            if args.validate:
                fit.publish_diagnostic(root, plan, validate=True)
    except fit.FitPaused as error:
        fit.log(str(error))
        raise SystemExit(130)
    except fit.FitBudgetExceeded as error:
        fit.log(f"RESOURCE STOP: {error}; evidence retained under {IDENTITY}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
