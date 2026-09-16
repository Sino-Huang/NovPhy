"""Audited norm-reduction repair of the finite balanced-native v2 stop."""

import argparse
from copy import deepcopy

import torch

from scripts import run_issue_76_balanced_native as parent
from scripts import run_issue_76_native_refit as fit


ROOT = fit.files.ROOT / ".local-artifacts/issue-76-balanced-native-norm-recovery-v1"
OUTPUT = parent.OUTPUT / "norm-recovery"
REPAIRED = {"scripts/issue_76_scaled_fit.py", "tests/test_issue_76_balanced_native.py"}
SOURCES = (
    "scripts/resume_issue_76_balanced_native_norm.py",
    "scripts/probe_issue_76_balanced_gradient.py",
    "tests/test_issue_76_norm_recovery.py",
    "docs/issue-76-balanced-native-norm-recovery.md",
)


def source_repair(original, current):
    if {k: v for k, v in original.items() if k != "source_text"} != {
            k: v for k, v in current.items() if k != "source_text"}:
        raise ValueError("norm recovery changed the frozen numeric/data protocol")
    if set(original["source_text"]) != set(current["source_text"]):
        raise ValueError("norm recovery changed the parent source inventory")
    changed = {name for name in original["source_text"]
               if original["source_text"][name] != current["source_text"][name]}
    if changed != REPAIRED:
        raise ValueError("norm recovery has an unrelated or missing source change")
    return {name: {"original": original["source_text"][name],
                   "repaired": current["source_text"][name]} for name in sorted(changed)}


def finite_failure(saved):
    if (saved["complete"] or not saved["failure"].startswith("RuntimeError: The total norm")
            or (saved["updates_completed"], saved["updates_applied"], saved["updates_skipped"])
               != (9114, 9041, 73)
            or not all(bool(torch.isfinite(v).all()) for v in saved["model"].values())
            or not all(bool(torch.isfinite(v).all())
                       for state in saved["optimizer"]["state"].values()
                       for v in state.values() if torch.is_tensor(v))):
        raise ValueError("norm recovery requires the finite pre-update-9114 failure")


def resume_checkpoint(current, archived):
    finite_failure(archived)
    if any(current[k] != archived[k] for k in (
            "plan_identity", "updates_requested", "updates_completed", "updates_applied",
            "updates_skipped", "failure", "complete", "backward_scale")):
        raise ValueError("live failure checkpoint differs from the audited stop")
    if (current["model"].keys() != archived["model"].keys()
            or not all(torch.equal(current["model"][k], v) for k, v in archived["model"].items())
            or current["optimizer"]["param_groups"] != archived["optimizer"]["param_groups"]
            or current["optimizer"]["state"].keys() != archived["optimizer"]["state"].keys()
            or not all(torch.equal(current["optimizer"]["state"][k][name], value)
                       if torch.is_tensor(value)
                       else current["optimizer"]["state"][k][name] == value
                       for k, state in archived["optimizer"]["state"].items()
                       for name, value in state.items())):
        raise ValueError("live model/optimizer state differs from the audited stop")
    value = deepcopy(current)
    value["failure"] = None
    return value


def make_plan():
    original = fit.files.read(parent.ROOT / "plan.json")
    repair = source_repair(original, parent.make_plan())
    checkpoint = (ROOT / "failed-checkpoint.pt" if (ROOT / "failed-checkpoint.pt").exists()
                  else fit.model_path(parent.ROOT, 760930003, False, "predictor"))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    finite_failure(saved)
    budget_path = (ROOT / "failed-budget.json" if (ROOT / "failed-budget.json").exists()
                   else parent.ROOT / "budgets" / "predictor-hybrid-760930003.json")
    budget = fit.files.read(budget_path)
    if saved["plan_identity"] != original["identity"] or budget["running"] or budget["stopped"]:
        raise ValueError("norm recovery parent identity/budget differs")
    return {
        "schema": "issue_76_balanced_native_norm_recovery_v1",
        "identity": ROOT.name,
        "parent_plan": original,
        "source_repair": repair,
        "archived_failure_checkpoint": str(ROOT / "failed-checkpoint.pt"),
        "failure": {k: saved[k] for k in (
            "updates_completed", "updates_applied", "updates_skipped", "failure", "last_loss")},
        "retained_budget": budget,
        "correction": "float64 gradient-norm reduction; unchanged scaled unit-clipping formula",
        "replay": {"update": 9114, "loss": 1.6729962825775146,
                   "scaled_gradients_finite": True, "float32_scaled_norm": "infinity",
                   "float64_scaled_norm": 1.113209733128011e31,
                   "repaired_clipped_norm": 0.9999999988982464},
        "source_text": {name: (fit.files.ROOT / name).read_text() for name in SOURCES},
        "optimizer_exposure_changed": False,
        "training_outcomes_consulted": False,
        "fresh_access": False,
        "advancement_authorized": False,
    }


def prepare():
    if ROOT.exists():
        raise ValueError("norm recovery is already prepared")
    plan = make_plan()
    checkpoint = torch.load(fit.model_path(parent.ROOT, 760930003, False, "predictor"),
                            map_location="cpu", weights_only=False)
    fit.atomic_torch(ROOT / "failed-checkpoint.pt", checkpoint)
    parent.immutable(ROOT / "failed-budget.json", plan["retained_budget"])
    parent.immutable(ROOT / "plan.json", plan)
    parent.immutable(OUTPUT / "plan.json", plan)
    fit.log("norm recovery frozen; original finite failure archived, no optimizer work")


def run(plan, device):
    budget = fit.require_finished_budget(parent.ROOT, "predictor-hybrid-760930003")
    if (budget["limit_seconds"] != plan["retained_budget"]["limit_seconds"]
            or budget["active_seconds"] < plan["retained_budget"]["active_seconds"]):
        raise ValueError("norm recovery reset or changed the retained allowance")
    marker = parent.ROOT / "norm-recovery-applied.json"
    receipt = {"recovery_plan_identity": plan["identity"],
               "archived_failure_checkpoint": plan["archived_failure_checkpoint"],
               "resume_update": 9114, "optimizer_exposure_changed": False}
    path = fit.model_path(parent.ROOT, 760930003, False, "predictor")
    current = torch.load(path, map_location="cpu", weights_only=False)
    if current["failure"]:
        archived = torch.load(ROOT / "failed-checkpoint.pt", map_location="cpu", weights_only=False)
        resumed = resume_checkpoint(current, archived)
        parent.immutable(marker, receipt)
        fit.atomic_torch(path, resumed)
    elif not marker.exists() or fit.files.read(marker) != receipt:
        raise ValueError("norm recovery lacks its applied source-repair receipt")
    parent.train_predictors(plan["parent_plan"], device)
    parent.train_controllers(plan["parent_plan"], device)
    report = parent.diagnose(plan["parent_plan"], device)
    value = {"schema": "issue_76_balanced_native_norm_recovery_report_v1",
             "recovery_plan": plan, "parent_report": report,
             "all_jobs_complete": report["all_jobs_complete"],
             "fresh_access": False, "advancement_authorized": False}
    parent.immutable(ROOT / "report.json", value)
    parent.immutable(OUTPUT / "report.json", value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "prepare", "run"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.prepare:
        prepare()
        return
    current = make_plan()
    if args.dry_run:
        print({"identity": current["identity"], "resume_update": 9114,
               "source_changes": sorted(current["source_repair"])}, flush=True)
        return
    plan = fit.files.read(ROOT / "plan.json")
    if plan != current:
        raise ValueError("norm recovery freeze changed")
    run(plan, args.device)


if __name__ == "__main__":
    main()
