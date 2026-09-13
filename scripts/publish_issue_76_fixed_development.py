"""Descriptive fixed-prediction development report, never an advancement gate."""
import argparse
from collections import defaultdict
from datetime import datetime
from statistics import mean

import numpy as np
import torch

from scripts import run_issue_76_fixed_development as run
from scripts import validate_issue_76_fixed_development_backend as backend

FIELDS = ("carrier_mse", "visual_mse", "memory_mse", "presence_mse_to_engine", "center_mse_to_engine",
          "pig_count_absolute_error", "block_count_absolute_error", "carrier_absolute_bound_excess")
SOURCES = ("scripts/publish_issue_76_fixed_development.py", "tests/test_issue_76_fixed_development_publication.py")


def clustered_effect(rows, seeds, specification):
    """One paired lineage draw carries all model seeds; families keep equal weight."""
    families = list(dict.fromkeys(row["family"] for row in rows))
    if len({row["base_cluster"] for row in rows}) != len(rows):
        raise ValueError("this frozen inventory requires one assigned member per base lineage")
    groups = {family: np.asarray([row["differences"] for row in rows
                                 if row["family"] == family and row["differences"] is not None], dtype=np.float64)
              for family in families}
    available = sum(len(values) for values in groups.values())
    counts = {family: {"assigned": sum(row["family"] == family for row in rows), "available": len(groups[family])}
              for family in families}
    result = {"assigned_lineages": len(rows), "available_lineages": available, "family_denominators": counts,
              "lineages_with_prediction_failure": sum(row["prediction_failure"] for row in rows),
              "negative_difference_favors_hybrid": True, "descriptive_only": True}
    if any(not len(values) for values in groups.values()):
        return {**result, "mean_difference": None, "interval": None, "unavailable_reason": "empty_family"}
    if any(values.shape != (len(values), len(seeds)) or not np.isfinite(values).all() for values in groups.values()):
        raise ValueError("paired lineage effect lacks the complete finite seed vector")
    generator = np.random.default_rng(specification["seed"])
    estimates = np.zeros(specification["bootstrap_replicates"])
    for values in groups.values():
        draws = generator.integers(len(values), size=(len(estimates), len(values)))
        estimates += values[draws].mean(axis=(1, 2)) / len(groups)
    tail = (1 - specification["confidence"]) / 2
    return {**result, "mean_difference": mean(float(values.mean()) for values in groups.values()),
            "per_seed": {str(seed): mean(float(values[:, index].mean()) for values in groups.values())
                         for index, seed in enumerate(seeds)},
            "per_family": {family: float(values.mean()) for family, values in groups.items()},
            "interval": np.quantile(estimates, [tail, 1 - tail]).tolist(),
            "bootstrap_replicates": len(estimates), "confidence": specification["confidence"]}


def curve_summary(rows, policy, elapsed, kind, seeds, families):
    records = [(row, row["policies"][policy]["curves"][str(elapsed)][kind]) for row in rows if row["policies"]]
    records = [(row, value) for row, value in records if value is not None]
    output = {"available_target_rows": len(records), "failed_prediction_rows": sum(value["failure"] is not None for _, value in records), "fields": {}}
    for field in FIELDS:
        groups = defaultdict(list)
        for row, value in records:
            if value.get(field) is not None:
                groups[row["seed"], row["family"]].append(value[field])
        complete = all(groups[seed, family] for seed in seeds for family in families)
        output["fields"][field] = {
            "equal_family_seed_mean": mean(mean(groups[seed, family]) for seed in seeds for family in families) if complete else None,
            "available_metric_rows": sum(map(len, groups.values())), "all_family_seed_groups_available": complete}
    return output


def matrix_summary(inventory, results, role):
    entries = [entry for entry in inventory["entries"] if entry["exposure_role"] == role]
    families = list(dict.fromkeys(entry["family"] for entry in entries))
    table = []
    for recipe in inventory["recipes"]:
        for pure in (False, True):
            rows = [row for result in results if result["cell"]["recipe"] == recipe and result["cell"]["pure"] == pure
                    for row in result["rows"]]
            for pair in (run.fit.CONTINUOUS_PAIRS if pure else run.fit.PAIRS):
                policy = run.fit.policy_name(pair)
                table.append({"recipe": recipe, "pure": pure, "policy": policy,
                    "assigned_rows": len(rows), "available_endpoint_rows": sum(row["available"] for row in rows),
                    "rows_with_any_prediction_failure": sum(row["policies"][policy]["prediction_failure"] for row in rows if row["policies"]),
                    "curves": {str(elapsed): {kind: curve_summary(rows, policy, elapsed, kind, inventory["seeds"], families)
                                              for kind in ("recursive", "observed_context_local")}
                               for elapsed in run.scoring.CURVE_TIMES}})
    return table


def contrast_rows(inventory, results, choices, comparator):
    rows_by_cell = {(result["cell"]["recipe"], result["cell"]["pure"], result["cell"]["seed"]):
                   {row["member_identity"]: row for row in result["rows"]} for result in results}
    left = choices["hybrid"]
    right = choices["pure"]
    hybrid_pair = next(pair for pair in run.fit.PAIRS if run.fit.policy_name(pair) == left["policy"])
    continuous_policy = run.fit.policy_name(next(pair for pair in run.fit.CONTINUOUS_PAIRS if pair.delta == hybrid_pair.delta))
    output = []
    for entry in inventory["entries"]:
        if entry["exposure_role"] != "model_selection":
            continue
        values, failures, availability = [], False, []
        for seed in inventory["seeds"]:
            lhs = rows_by_cell[left["recipe"], False, seed][entry["member_identity"]]
            rhs = rows_by_cell[right["recipe"], True, seed][entry["member_identity"]] if comparator == "pure" else lhs
            if lhs["available"] != rhs["available"]:
                raise ValueError("paired comparison endpoint availability differs")
            availability.append(lhs["available"])
            if not lhs["available"]:
                continue
            lhs_policy = lhs["policies"][left["policy"]]
            lhs_metric = lhs_policy["curves"]["11250"]["recursive"]
            if comparator == "unchanged":
                rhs_metric = lhs["unchanged_carrier_reference"]["curves"]["11250"]
                rhs_failure = rhs_metric["failure"] is not None
            else:
                rhs_policy = rhs["policies"][right["policy"] if comparator == "pure" else continuous_policy]
                rhs_metric = rhs_policy["curves"]["11250"]["recursive"]
                rhs_failure = rhs_policy["prediction_failure"]
            failures |= lhs_policy["prediction_failure"] or rhs_failure
            values.append(lhs_metric["carrier_mse"] - rhs_metric["carrier_mse"])
        if any(availability) and not all(availability):
            raise ValueError("lineage comparison is missing a paired seed")
        output.append({"base_cluster": entry["base_cluster"], "family": entry["family"],
                       "differences": values if values else None, "prediction_failure": failures})
    return output


def publication_plan(plan):
    return {"identity": "issue-76-fixed-development-publication-v1", "execution_plan_identity": plan["identity"],
            "uncertainty": plan["uncertainty"], "field_inventory": list(FIELDS),
            "source_text": {path: (run.fit.files.ROOT / path).read_text() for path in SOURCES},
            "advancement_authorized": False, "fresh_access": False}


def publish(plan):
    if run.fit.files.read(run.ROOT / "publication-plan.json") != publication_plan(plan):
        raise ValueError("publication source/analysis freeze changed")
    correction = backend.load_backend_plan(plan)
    validated = {role: run.fit.files.read(run.phase_path(role, "validation.json")) for role in run.ROLES}
    if any(not report["validated"] or report["cells_validated"] != 54 or report["validation_method_identity"] != correction["identity"] for report in validated.values()):
        raise ValueError("complete qualified validation is required for both roles")
    choice = run.fit.files.read(run.ROOT / "calibration-choice.json")
    timeline = [run.fit.files.read(run.phase_path("calibration", "access.json"))["started_utc"],
                run.fit.files.read(run.phase_path("calibration", "complete.json"))["completed_utc"],
                validated["calibration"]["completed_utc"], choice["frozen_utc"],
                run.fit.files.read(run.phase_path("model_selection", "access.json"))["started_utc"],
                run.fit.files.read(run.phase_path("model_selection", "complete.json"))["completed_utc"],
                validated["model_selection"]["completed_utc"]]
    if any(datetime.fromisoformat(a) > datetime.fromisoformat(b) for a, b in zip(timeline, timeline[1:])):
        raise ValueError("calibration/validation/choice/model-selection access order differs")
    with run.fit.fitting_budget(run.ROOT, "independent-validation", plan["limits"]["independent_validation_wall_seconds_total"], "cpu") as budget:
        results = {role: run.load_results(plan, role) for role in run.ROLES}
        if choice["choice"] != run.scoring.calibration_choice(plan["inventory"], results["calibration"]):
            raise ValueError("frozen calibration decision is not reproducible")
        tables = {}
        for role in run.ROLES:
            run.check_storage(plan, budget)
            tables[role] = matrix_summary(plan["inventory"], results[role], role)
        contrasts = {name: clustered_effect(contrast_rows(plan["inventory"], results["model_selection"], choice["choice"]["selected"], name),
                                          plan["inventory"]["seeds"], plan["uncertainty"])
                     for name in ("pure", "same_hybrid_continuous", "unchanged")}
        report = {"execution_plan_identity": plan["identity"], "publication_plan_identity": publication_plan(plan)["identity"],
            "selected": choice["choice"]["selected"], "complete_matrix": tables, "model_selection_contrasts": contrasts,
            "access_timeline": timeline, "selection_optimism": "calibration estimates select the controls; model-selection choices were not retuned",
            "validation_method": correction["identity"], "original_strict_cpu_check_passed": False,
            "model_selection_endpoint_coverage_passed": all(value["available"] / value["assigned"] >= .9
                for value in contrasts["pure"]["family_denominators"].values()),
            "scoring_budgets": {role: run.fit.require_finished_budget(run.ROOT, f"score-{role}") for role in run.ROLES},
            "transition_work": {role: {field: sum(batch[field] for result in results[role] for batch in result["batches"])
                                       for field in ("recursive_transitions", "local_transitions", "linear_macs", "controller_calls")}
                                for role in run.ROLES},
            "work_interpretation": "linear MAC proxies, not full FLOPs or matched deployment compute",
            "preparation_budgets": {f"{role}-{seed}": run.fit.require_finished_budget(run.ROOT, f"prepare-{role}-{seed}")
                                    for role in run.ROLES for seed in plan["inventory"]["seeds"]},
            "prior_cost_ledger_reference": "execution-plan.json:prior_costs", "smoke_cost_reference": "smoke-report.json",
            "full_result_paths": {role: [str(run.phase_path(role, run.cell_name(cell))) for cell in plan["inventory"]["models"]] for role in run.ROLES},
            "failed_training_qualifications_preserved": True, "fresh_access": False, "new_optimizer_updates": 0,
            "adaptive_gameplay_measured": False, "strongest_gameplay_policy_established": False, "advancement_authorized": False}
        run.check_storage(plan, budget)
    report["cumulative_validation_and_publication_budget"] = run.fit.require_finished_budget(run.ROOT, "independent-validation")
    run.immutable(run.ROOT / "report.json", report)
    run.immutable(run.OUTPUT / "report.json", report)
    print("Complete fixed-prediction report published; advancement remains unauthorized.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.prepare:
        value = publication_plan(run.load_plan())
        run.immutable(run.ROOT / "publication-plan.json", value)
        run.immutable(run.OUTPUT / "publication-plan.json", value)
        print("Publication formulas frozen; no development predictions read.")
    elif args.publish:
        publish(run.load_plan())
    else:
        print("No-write: descriptive fixed-prediction effects only; no advancement claim.")


if __name__ == "__main__":
    main()
