"""Exactly replay lower-rate curves and validate optimizer continuity and costs."""
from pathlib import Path

import torch

from scripts import run_issue_76_lower_rate_dynamics as run
from scripts.validate_issue_76_full_duration_dynamics import check_optimizer_steps


def check_checkpoint(saved, before, model, plan, pure):
    if (saved["plan_identity"] != plan["identity"] or not saved["complete"] or saved["failure"]
            or (saved["updates_requested"], saved["updates_completed"], saved["updates_applied"],
                saved["updates_skipped"]) != (6000, 6000, 5952, 48)):
        raise ValueError("lower-rate checkpoint source, completion or exposure differs")
    expected = run.initial_checkpoint(before, plan)
    if saved["optimizer"]["param_groups"] != expected["optimizer"]["param_groups"]:
        raise ValueError("optimizer settings differ beyond the frozen learning-rate change")
    return check_optimizer_steps(model, before["optimizer"], saved["optimizer"], plan, pure)


def main():
    torch.set_num_threads(1)
    report = run.fit.files.read(run.OUTPUT / "report.json")
    plan = run.make_plan()
    if report["plan"] != plan or run.fit.files.read(run.ROOT / "plan.json") != plan:
        raise ValueError("published or frozen lower-rate plan differs")
    references = {(r["seed"], r["pure"], r["member_identity"]): r for r in plan["reference_rows"]}
    rows, checkpoints = [], []
    with run.fit.fitting_budget(run.ROOT, "validation", 180, "cpu") as budget:
        for seed in plan["seeds"]:
            for pure in (False, True):
                saved = torch.load(run.fit.model_path(run.ROOT, seed, pure, "predictor"), map_location="cpu", weights_only=False)
                before = torch.load(plan["source_checkpoint_paths"][f"{seed}-{pure}"], map_location="cpu", weights_only=False)
                model = run.fit.new_predictor(plan, pure, "cpu").eval().requires_grad_(False)
                steps = check_checkpoint(saved, before, model, plan, pure)
                model.load_state_dict(saved["model"])
                checkpoints.append(dict(seed=seed, pure=pure, updates_applied=5952, updates_skipped=48,
                                        learning_rate=plan["learning_rate"], optimizer_steps=steps))
                for entry in plan["evaluation"]:
                    budget.check()
                    shard = run.fit.load_shard(Path(plan["data_root"]), entry, plan["data_binding"], fitting=True)
                    carrier = run.fit.load_carrier(Path(plan["data_root"]), seed, entry, plan["data_binding"])
                    rows.append({**references[seed, pure, entry["member_identity"]],
                                 "lower_rate": run.source.source.drift_curves(model, shard, carrier)})
        summaries = run.summarize(plan, rows)
        if rows != report["rows"] or summaries != report["summaries"]:
            raise ValueError("recomputed curves or decisions differ")
        if report["qualification_supported"] != all(s["qualification_supported"] for s in summaries):
            raise ValueError("aggregate qualification differs")
        if report["fresh_access"] or report["advancement_passed"]:
            raise ValueError("training diagnostics cannot claim advancement or fresh access")
        required = {"exposure-audit", "checkpoint-preflight", "diagnostic"}
        required.update(f"preparation-{seed}" for seed in plan["seeds"])
        required.update(f"predictor-{arm}-{seed}" for seed in plan["seeds"] for arm in ("hybrid", "pure"))
        if set(report["budgets"]) != required:
            raise ValueError("stage resource inventory differs")
        for key, expected in report["budgets"].items():
            actual = run.fit.require_finished_budget(run.ROOT, key)
            if actual != expected or actual["peak_cpu_rss_mib"] > 12288 or actual["peak_cuda_allocated_mib"] > 8192:
                raise ValueError("published cost or resource evidence differs")
        budget.check()
    run.immutable(run.OUTPUT / "validation.json", dict(validated=True, plan_identity=plan["identity"],
                  recomputed_rows=len(rows), recomputed_policy_curves=sum(len(r["lower_rate"]) for r in rows),
                  checkpoints=checkpoints, qualification_supported=report["qualification_supported"],
                  optimization_performed=False, fresh_access=False,
                  additional_validation_budget_path=str(run.ROOT / "budgets/validation.json")))
    print(dict(validated=True, recomputed_rows=30, recomputed_policy_curves=180,
               qualification_supported=report["qualification_supported"]), flush=True)


if __name__ == "__main__":
    main()
