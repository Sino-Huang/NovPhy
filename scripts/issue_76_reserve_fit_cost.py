"""Charge completed CUDA engineering smokes symmetrically to unused fit budgets."""
import argparse

from scripts import run_issue_76_native_refit as fit
from scripts import issue_76_expansion as files


def reservation(plan):
    folder = files.ROOT / "data/issue-76-native-refit-synthetic-smoke"
    receipts = []
    totals = {"common": 0., "predictor": 0., "controller": 0.}
    for path in sorted(folder.glob("attempt-*.json")):
        value = files.read(path)
        if value["device"] != "cuda" or value["research_fits"] != 0:
            raise ValueError("engineering receipt is not the declared CUDA synthetic test")
        costs = {}
        for key, budget in value["budget_records"].items():
            category = key.split("-", 1)[0]
            costs[key] = budget["active_seconds"]
            totals[category] += budget["active_seconds"]
        receipts.append({"path": str(path.relative_to(files.ROOT)), "costs": costs})
    if not receipts:
        raise ValueError("complete the CUDA smoke before reserving its real cost")
    count = len(plan["seeds"])
    return {"schema": "issue_76_symmetric_engineering_fit_cost_v1", "plan_identity": plan["identity"],
            "receipts": receipts, "actual_category_seconds": totals,
            "initial_seconds_per_budget": {"common": totals["common"] / count,
                "predictor": totals["predictor"] / (2 * count), "controller": totals["controller"] / (2 * count)},
            "rule": "same category debit for every paired seed/arm; no increase to the 3h/6h/3h ceilings"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    plan = fit.load_plan()
    value = reservation(plan)
    target = fit.ROOT / "engineering-cost-reservation.json"
    if args.validate:
        if files.read(target) != value:
            raise ValueError("engineering cost inventory changed after reservation")
    else:
        if target.exists():
            raise ValueError("engineering costs already reserved; never reset active fit budgets")
        keys = []
        for seed in plan["seeds"]:
            keys.append((f"common-{seed}", "common", plan["limits"]["common_seconds_per_seed"]))
            for name in ("hybrid", "pure"):
                keys.append((f"predictor-{name}-{seed}", "predictor", plan["limits"]["predictor_seconds_per_arm_seed"]))
                keys.append((f"controller-{name}-{seed}", "controller", plan["limits"]["controller_diagnostic_seconds_per_arm_seed"]))
        if any((fit.ROOT / "budgets" / (key + ".json")).exists() for key, _, _ in keys):
            raise ValueError("fitting budgets already exist; do not overwrite consumed allowances")
        for key, category, limit in keys:
            debit = value["initial_seconds_per_budget"][category]
            files.write(fit.ROOT / "budgets" / (key + ".json"), {
                "active_seconds": debit, "engineering_seconds": debit,
                "engineering_reservation": str(target), "peak_cpu_rss_mib": 0.,
                "peak_cuda_allocated_mib": 0., "peak_cuda_reserved_mib": 0.,
                "limit_seconds": limit, "stopped": False, "running": False})
        files.write(target, value)
    fit.log(f"engineering fit costs {'reserved' if args.prepare else 'validated'}: {value['actual_category_seconds']}")


if __name__ == "__main__":
    main()
