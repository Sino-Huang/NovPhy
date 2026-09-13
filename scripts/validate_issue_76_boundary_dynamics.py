"""Recompute the complete boundary-balanced diagnostic without fitting."""
import torch

from scripts import run_issue_76_boundary_dynamics as run


def main():
    torch.set_num_threads(1)
    report = run.fit.files.read(run.OUTPUT / "report.json")
    plan = run.make_plan()
    if report["plan"] != plan or run.fit.files.read(run.ROOT / "plan.json") != plan:
        raise ValueError("publication or frozen boundary plan differs")
    references = {(r["seed"], r["pure"], r["member_identity"]): r["curves"] for r in plan["reference_rows"]}
    rows, checkpoints = [], []
    with run.fit.fitting_budget(run.ROOT, "validation", 180, "cpu") as budget:
        for seed in plan["seeds"]:
            for pure in (False, True):
                saved = torch.load(run.fit.model_path(run.ROOT, seed, pure, "predictor"),
                                    map_location="cpu", weights_only=False)
                if (saved["plan_identity"] != plan["identity"] or not saved["complete"] or saved["failure"]
                        or (saved["updates_requested"], saved["updates_completed"], saved["updates_applied"],
                            saved["updates_skipped"]) != (6000, 6000, 5952, 48)):
                    raise ValueError("boundary predictor source, completion, or exposure differs")
                model = run.fit.new_predictor(plan, pure, "cpu").eval().requires_grad_(False)
                model.load_state_dict(saved["model"])
                checkpoints.append(dict(seed=seed, pure=pure, updates_applied=5952, updates_skipped=48))
                for entry in plan["evaluation"]:
                    budget.check()
                    shard = run.fit.load_shard(run.previous.parent.ROOT, entry, plan["data_binding"], fitting=True)
                    carrier = run.fit.load_carrier(run.previous.parent.ROOT, seed, entry, plan["data_binding"])
                    rows.append(dict(seed=seed, pure=pure, member_identity=entry["member_identity"], family=entry["family"],
                        reference=references[seed, pure, entry["member_identity"]], boundary=run.drift_curves(model, shard, carrier)))
        summaries = run.summarize(plan, rows)
        if rows != report["rows"] or summaries != report["summaries"]:
            raise ValueError("complete recomputed curves or summaries differ")
        if report["qualification_supported"] != all(s["qualification_supported"] for s in summaries):
            raise ValueError("aggregate boundary qualification differs")
        if report["fresh_access"] or report["advancement_passed"]:
            raise ValueError("training-only diagnostic cannot claim fresh access or advancement")
        for key, expected in report["budgets"].items():
            actual = run.fit.require_finished_budget(run.ROOT, key)
            if actual != expected or actual["peak_cpu_rss_mib"] > 12288 or actual["peak_cuda_allocated_mib"] > 8192:
                raise ValueError("published fitting cost or resource evidence differs")
        budget.check()
    value = dict(validated=True, plan_identity=plan["identity"], recomputed_rows=len(rows),
        recomputed_policy_curves=sum(len(r["boundary"]) for r in rows), checkpoints=checkpoints,
        optimization_performed=False, fresh_access=False, qualification_supported=report["qualification_supported"],
        additional_validation_budget_path=str(run.ROOT / "budgets/validation.json"))
    run.previous.parent.previous.immutable(run.OUTPUT / "validation.json", value)
    print(value, flush=True)


if __name__ == "__main__":
    main()
