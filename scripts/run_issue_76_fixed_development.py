"""Source-frozen, resumable fixed-only development evaluation; no fresh access."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import torch

from scripts import prepare_issue_76_fixed_development as metadata
from scripts import score_issue_76_fixed_development as scoring

fit = scoring.fit
ROOT, OUTPUT = metadata.ROOT, metadata.OUTPUT
immutable = metadata.source.immutable
SOURCES = ("scripts/run_issue_76_fixed_development.py", "tests/test_issue_76_fixed_development_execution.py")
ROLES = ("calibration", "model_selection")


def prior_cost_ledger(project):
    """Canonical job receipts once, not recursively duplicated plan summaries.

    Inspect budget metadata only. Include failed and nonselected #76 branches.
    Public validation-budget files are mirrors of the canonical private jobs.
    """
    receipts = []
    for root in sorted((project / ".local-artifacts").iterdir()):
        if not root.is_dir() or not root.name.startswith("issue-76-") or root.name == ROOT.name:
            continue
        paths = [*(root / "budgets").glob("*.json"), *root.glob("budget*.json")]
        for path in sorted(set(paths)):
            value = fit.files.read(path)
            if "active_seconds" not in value:
                raise ValueError(f"unknown cost receipt schema: {path}")
            receipts.append({"path": str(path), "receipt": value})
    mirrors = []
    for path in sorted((project / "data").glob("issue-76-*/validation-budget.json")):
        canonical = project / ".local-artifacts" / (path.parent.name + "-v1") / "budgets/validation.json"
        if not canonical.exists() or fit.files.read(path) != fit.files.read(canonical):
            raise ValueError(f"public validation cost lacks its exact canonical job: {path}")
        mirrors.append({"path": str(path), "counted_at": str(canonical)})
    reservation_path = project / ".local-artifacts/issue-76-native-refit-v2/engineering-cost-reservation.json"
    reservation = fit.files.read(reservation_path)
    return {"receipts": receipts, "public_mirrors_not_added": mirrors,
            "recorded_active_seconds_sum": sum(row["receipt"]["active_seconds"] for row in receipts),
            "timing_semantics": "sum of recorded job wall-time charges, not GPU kernel hours or full FLOPs",
            "engineering_reservation_already_debited": {"path": str(reservation_path), "receipt": reservation},
            "nested_parent_cost_summaries_added_again": False,
            "includes_failed_and_nonselected_branches": True,
            "scope": "recorded #76 capture, representation, predictor, controller and diagnostic job receipts",
            "unmetered_earlier_engineering_or_prior_issue_work_is_not_zero": True,
            "complete_matched_deployment_cost_claim": False}


def make_plan():
    inventory = fit.files.read(ROOT / "inventory.json")
    smoke_plan = fit.files.read(ROOT / "smoke-plan.json")
    smoke = fit.files.read(ROOT / "smoke-report.json")
    checkpoints = fit.files.read(ROOT / "smoke-checkpoints.json")
    if (smoke["plan_identity"] != smoke_plan["identity"] or not smoke["synthetic_only"]
            or smoke["development_records_read"] != 0 or smoke["checkpoint_instances_validated"] != 54
            or [row["cell"] for row in checkpoints] != inventory["models"]
            or not all(row["model_state_finite"] for row in checkpoints)
            or any(not row["all_predictions_finite"] for row in smoke["summaries"])):
        raise ValueError("complete source-bound synthetic preflight is required")
    for path, text in smoke_plan["source_text"].items():
        if (fit.files.ROOT / path).read_text() != text:
            raise ValueError(f"synthetic-smoke source changed: {path}")
    for key in ("smoke-checkpoints", "synthetic-smoke"):
        fit.require_finished_budget(ROOT, key)
    return dict(identity="issue-76-fixed-development-execution-v1", inventory=inventory,
        smoke_plan_identity=smoke_plan["identity"], smoke_report=smoke, checkpoint_completion_receipts=checkpoints,
        source_text={**smoke_plan["source_text"], **{path: (fit.files.ROOT / path).read_text() for path in SOURCES}},
        prior_costs=prior_cost_ledger(fit.files.ROOT), device="cuda", threads=1, batch_size=50,
        limits={"preparation_wall_seconds_per_role_seed": 300, "scoring_wall_seconds_per_role": 900,
                "independent_validation_wall_seconds_total": 900, "rss_mib": 12288,
                "cuda_allocated_mib": 8192, "new_artifact_bytes": 2**30},
        timing_semantics="wall-time upper bounds enforce the CPU/CUDA active-time allowances conservatively",
        roles=list(ROLES), score_execution_authorized=True, fresh_access=False, optimization_performed=False,
        failed_training_qualifications_preserved=True, advancement_authorized=False,
        selection_order="complete calibration; independent calibration validation; freeze common choices; model_selection",
        independent_validation={"all_cells_and_members": True, "all_fixed_policies_and_curve_times": True,
            "source_targets": "exact first-shot observed indices; separately verify compact preparation",
            "method": "independent CPU transition loop and field-metric formulas; no recursive_curves or field_metrics reuse",
            "metric_relative_tolerance": .001, "metric_absolute_tolerance": .00001,
            "failure_masks_and_available_denominators_exact": True,
            "check_work_and_source_inventory": True},
        uncertainty={"role": "model_selection", "bootstrap_replicates": 10000, "seed": 760940001,
            "confidence": .95, "unit": "paired base lineage, with every seed carried together",
            "family_weight": "equal; resample lineages within family", "seed_weight": "equal",
            "contrasts": ["frozen hybrid minus frozen pure", "frozen hybrid minus same-checkpoint continuous at same horizon",
                          "frozen hybrid minus unchanged-carrier prediction reference"],
            "descriptive_only": True, "three_seeds_not_seed_robust_advancement": True},
        unbudgeted_prior_readonly_completion_check_wall_seconds=8.451315055135638)


def load_plan():
    plan = fit.files.read(ROOT / "execution-plan.json")
    if not plan["score_execution_authorized"] or plan["fresh_access"] or plan["optimization_performed"]:
        raise ValueError("only frozen fixed development scoring is authorized")
    if plan["inventory"] != fit.files.read(ROOT / "inventory.json"):
        raise ValueError("frozen development inventory changed")
    for path, text in plan["source_text"].items():
        if (fit.files.ROOT / path).read_text() != text:
            raise ValueError(f"execution source changed: {path}")
    return plan


def phase_path(role, name):
    if role not in ROLES:
        raise ValueError("only assigned development roles are permitted")
    return ROOT / role / name


def cell_name(cell):
    return f"{cell['recipe']}-{'pure' if cell['pure'] else 'hybrid'}-{cell['seed']}.json"


def check_storage(plan, budget):
    budget.check()
    if sum(path.stat().st_size for path in ROOT.rglob("*") if path.is_file()) > plan["limits"]["new_artifact_bytes"]:
        budget.value["stopped"] = True
        budget.save()
        raise fit.FitBudgetExceeded("fixed development derived-artifact limit exceeded")


def check_role_access(plan, role):
    if role not in plan["roles"]:
        raise ValueError("unassigned/fresh role is forbidden")
    if role == "model_selection":
        frozen = fit.files.read(ROOT / "calibration-choice.json")
        if frozen["execution_plan_identity"] != plan["identity"] or frozen["choice"] != scoring.calibration_choice(
                plan["inventory"], load_results(plan, "calibration")):
            raise ValueError("model selection requires the exact prior calibration choice freeze")


def prepare_role(plan, role):
    check_role_access(plan, role)
    inventory = plan["inventory"]
    data_plan = fit.files.read(Path(inventory["data_plan_path"]))
    if data_plan["identity"] != inventory["data_plan_identity"]:
        raise ValueError("common data plan identity differs")
    entries = [entry for entry in inventory["entries"] if entry["exposure_role"] == role]
    for seed in inventory["seeds"]:
        path = phase_path(role, f"prepared-{seed}.pt")
        key = f"prepare-{role}-{seed}"
        if path.exists():
            fit.require_finished_budget(ROOT, key)
            continue
        with fit.fitting_budget(ROOT, key, plan["limits"]["preparation_wall_seconds_per_role_seed"], "cpu") as budget:
            members = []
            for entry in entries:
                check_storage(plan, budget)
                members.append(scoring.prepare_member(Path(inventory["data_root"]), data_plan, seed, entry))
            fit.atomic_torch(path, {"execution_plan_identity": plan["identity"], "seed": seed,
                                    "role": role, "members": members})
            check_storage(plan, budget)
        print(f"Prepared {role} seed {seed}: {len(members)} assigned, {sum(11250 in m['targets'] for m in members)} exact endpoints", flush=True)


def load_prepared(plan, role, seed):
    value = torch.load(phase_path(role, f"prepared-{seed}.pt"), map_location="cpu", weights_only=False)
    entries = [entry for entry in plan["inventory"]["entries"] if entry["exposure_role"] == role]
    if (value["execution_plan_identity"] != plan["identity"] or value["seed"] != seed or value["role"] != role
            or [member["entry"] for member in value["members"]] != entries):
        raise ValueError("prepared member, role, seed or source inventory differs")
    return value["members"]


def score_role(plan, role):
    check_role_access(plan, role)
    if phase_path(role, "complete.json").exists():
        load_results(plan, role)
        print(f"Retaining completed {role}; no scoring repeated.", flush=True)
        return
    access = phase_path(role, "access.json")
    if not access.exists():
        immutable(access, {"execution_plan_identity": plan["identity"], "role": role,
                           "started_utc": datetime.now(timezone.utc).isoformat(),
                           "calibration_choice_present_before_access": (ROOT / "calibration-choice.json").exists()})
    prepare_role(plan, role)
    torch.cuda.init()
    prepared = {seed: load_prepared(plan, role, seed) for seed in plan["inventory"]["seeds"]}
    with fit.fitting_budget(ROOT, f"score-{role}", plan["limits"]["scoring_wall_seconds_per_role"], plan["device"]) as budget:
        started = time.monotonic()
        for ordinal, cell in enumerate(plan["inventory"]["models"], 1):
            check_storage(plan, budget)
            path = phase_path(role, cell_name(cell))
            if path.exists():
                if fit.files.read(path)["cell"] != cell:
                    raise ValueError("existing scored cell differs from the frozen inventory")
                continue
            model, sources = scoring.load_cell(plan["inventory"], cell, plan["device"])
            result = scoring.score_cell(model, cell, prepared[cell["seed"]], plan["device"],
                batch_size=plan["batch_size"], check_resources=lambda: check_storage(plan, budget))
            result.update(execution_plan_identity=plan["identity"], role=role, source_receipts=sources)
            immutable(path, result)
            del model, result
            check_storage(plan, budget)
            budget.save()
            elapsed = time.monotonic() - started
            print(f"{role} cells={ordinal}/54 active={budget.value['active_seconds']:.2f}s ETA={elapsed / ordinal * (54 - ordinal):.1f}s", flush=True)
    immutable(phase_path(role, "complete.json"), {"execution_plan_identity": plan["identity"], "role": role,
        "cells": [cell_name(cell) for cell in plan["inventory"]["models"]],
        "scoring_budget": fit.require_finished_budget(ROOT, f"score-{role}"),
        "completed_utc": datetime.now(timezone.utc).isoformat()})


def load_results(plan, role):
    complete = fit.files.read(phase_path(role, "complete.json"))
    if (complete["execution_plan_identity"] != plan["identity"] or complete["role"] != role
            or complete["cells"] != [cell_name(cell) for cell in plan["inventory"]["models"]]):
        raise ValueError("role completion inventory differs")
    fit.require_finished_budget(ROOT, f"score-{role}")
    return [fit.files.read(phase_path(role, cell_name(cell))) for cell in plan["inventory"]["models"]]


def freeze_choice(plan):
    validation = fit.files.read(phase_path("calibration", "validation.json"))
    if (not validation["validated"] or validation["execution_plan_identity"] != plan["identity"]
            or validation["cells_validated"] != 54 or not validation["all_assigned_members_checked"]):
        raise ValueError("complete independent calibration validation is required before selection")
    choice = scoring.calibration_choice(plan["inventory"], load_results(plan, "calibration"))
    frozen = {"execution_plan_identity": plan["identity"], "choice": choice,
              "frozen_utc": datetime.now(timezone.utc).isoformat(), "model_selection_started": False}
    if (ROOT / "calibration-choice.json").exists():
        previous = fit.files.read(ROOT / "calibration-choice.json")
        if previous["choice"] != choice or previous["execution_plan_identity"] != plan["identity"]:
            raise ValueError("existing calibration choice differs")
        return
    if phase_path("model_selection", "access.json").exists():
        raise ValueError("cannot first freeze choices after model-selection access")
    immutable(ROOT / "calibration-choice.json", frozen)
    immutable(OUTPUT / "calibration-choice.json", frozen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--score-calibration", action="store_true")
    modes.add_argument("--freeze-choice", action="store_true")
    modes.add_argument("--score-model-selection", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.prepare:
        plan = make_plan()
        immutable(ROOT / "execution-plan.json", plan)
        immutable(OUTPUT / "execution-plan.json", plan)
        print("Execution plan frozen; no development records read.", flush=True)
    elif args.score_calibration or args.score_model_selection:
        score_role(load_plan(), "calibration" if args.score_calibration else "model_selection")
    elif args.freeze_choice:
        freeze_choice(load_plan())
    else:
        print("No-write: freeze execution plan, score calibration, independently validate, freeze choices, then model selection. No fresh access.")


if __name__ == "__main__":
    main()
