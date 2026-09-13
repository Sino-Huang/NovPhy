"""Explicit arithmetic-backend correction; preserve the failed strict CPU check."""
import argparse
from datetime import datetime, timezone

import torch

from scripts import run_issue_76_fixed_development as run
from scripts import validate_issue_76_fixed_development as independent

SOURCES = ("scripts/validate_issue_76_fixed_development_backend.py",
           "tests/test_issue_76_fixed_development_backend.py",
           "scripts/test_issue_76_fixed_development.py",
           "docs/issue-76-fixed-development-backend-correction.md")


class SourceBackendTransitions:
    """Execute only the model transition on its source device; keep the host loop.

    The unchanged independent validator owns timing, recursive state, masks and
    NumPy metrics. This object receives no targets, future observations or saved
    predictions. Attribute delegation exposes the model's static MAC inventory.
    """
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def __getattr__(self, name):
        return getattr(self.model, name)

    def carrier(self, current, action, pair):
        return self.model.carrier(current.to(self.device), action.to(self.device), pair).cpu()


def prepare(plan):
    original = run.fit.files.read(run.ROOT / "validation-plan.json")
    if original != independent.validation_plan(plan):
        raise ValueError("original independent verifier source changed")
    status = run.fit.files.read(run.OUTPUT / "calibration-status.json")
    value = {"identity": "issue-76-fixed-development-backend-correction-v1",
        "execution_plan_identity": plan["identity"], "original_validation_plan": original,
        "original_strict_cpu_failure": status["strict_cpu_comparison_failure"],
        "prior_validation_wall_seconds": status["independent_validation_budget_spent_wall_seconds"],
        "transition_device": plan["device"], "metric_device": "CPU NumPy",
        "method": "unchanged independent host loop, exact source-backend float32 transitions, independent NumPy field formulas",
        "specification": plan["independent_validation"], "all_cells_members_policies_and_times_required": True,
        "metric_tolerances_changed": False, "saved_predictions_changed": False,
        "original_strict_cpu_check_passed": False, "backend_invariance_claim": False,
        "selection_or_advancement_criteria_changed": False, "fresh_access": False,
        "resource_rule": "shared 900-second validation wall budget retained; score wall plus ALL validation wall plus focused diagnostic must also fit each 900-second role allowance",
        "focused_diagnostic_wall_seconds": status["strict_cpu_comparison_failure"]["focused_diagnostic_wall_seconds"],
        "source_text": {path: (run.fit.files.ROOT / path).read_text() for path in SOURCES}}
    run.immutable(run.ROOT / "backend-validation-plan.json", value)
    run.immutable(run.OUTPUT / "backend-validation-plan.json", value)


def load_backend_plan(plan):
    value = run.fit.files.read(run.ROOT / "backend-validation-plan.json")
    if (value["execution_plan_identity"] != plan["identity"]
            or value["original_validation_plan"] != independent.validation_plan(plan)
            or value["specification"] != plan["independent_validation"]
            or value["transition_device"] != plan["device"]):
        raise ValueError("arithmetic-backend correction differs from its source freeze")
    for path, text in value["source_text"].items():
        if (run.fit.files.ROOT / path).read_text() != text:
            raise ValueError("arithmetic-backend verifier source changed")
    return value


def validate_role(plan, role):
    correction = load_backend_plan(plan)
    run.check_role_access(plan, role)
    target = run.phase_path(role, "validation.json")
    if target.exists():
        report = run.fit.files.read(target)
        if report["validation_method_identity"] != correction["identity"] or not report["validated"]:
            raise ValueError("existing role validation uses a different method")
        print(f"Retaining complete {role} same-backend validation.", flush=True)
        return
    results = run.load_results(plan, role)
    largest_score_charge = max(run.fit.require_finished_budget(run.ROOT, f"score-{item}")["active_seconds"]
                              for item in run.ROLES if run.phase_path(item, "complete.json").exists())
    summaries = []
    torch.cuda.init()
    with run.fit.fitting_budget(run.ROOT, "independent-validation", plan["limits"]["independent_validation_wall_seconds_total"], plan["device"]) as budget:
        def check():
            run.check_storage(plan, budget)
            if largest_score_charge + budget.value["active_seconds"] + correction["focused_diagnostic_wall_seconds"] >= plan["limits"]["scoring_wall_seconds_per_role"]:
                budget.value["stopped"] = True
                budget.save()
                raise run.fit.FitBudgetExceeded("combined scoring and conservative validation charge exceeds a role allowance")
        prepared = {}
        for seed in plan["inventory"]["seeds"]:
            prepared[seed] = run.load_prepared(plan, role, seed)
            independent.validate_prepared(plan, role, seed, prepared[seed], check)
        for index, (cell, result) in enumerate(zip(plan["inventory"]["models"], results, strict=True), 1):
            check()
            model, receipts = run.scoring.load_cell(plan["inventory"], cell, plan["device"])
            if receipts != result["source_receipts"]:
                raise ValueError("checkpoint source/completion receipt differs")
            summary = independent.validate_cell(plan, cell, result, prepared[cell["seed"]], SourceBackendTransitions(model), check)
            summaries.append(summary)
            del model
            budget.save()
            print(f"Independent source-backend {role} validation {index}/54 cumulative={budget.value['active_seconds']:.2f}s", flush=True)
        check()
    report = {"execution_plan_identity": plan["identity"], "validation_method_identity": correction["identity"],
        "role": role, "validated": True, "cells_validated": len(summaries), "all_assigned_members_checked": True,
        "summaries": summaries, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "metric_tolerances_changed": False, "original_strict_cpu_calibration_check_passed": False,
        "backend_invariance_claim": False, "original_cpu_failure_retained_at": "backend-validation-plan.json",
        "budget_including_failed_attempts": run.fit.require_finished_budget(run.ROOT, "independent-validation"),
        "advancement_authorized": False}
    run.immutable(target, report)
    run.immutable(run.OUTPUT / f"{role}-validation.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--validate-calibration", action="store_true")
    modes.add_argument("--validate-model-selection", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.prepare:
        prepare(run.load_plan())
        print("Source-backend technical verification correction frozen; no predictions evaluated.")
    elif args.validate_calibration or args.validate_model_selection:
        validate_role(run.load_plan(), "calibration" if args.validate_calibration else "model_selection")
    else:
        print("No-write: same-backend independent verification; original strict CPU failure retained.")


if __name__ == "__main__":
    main()
