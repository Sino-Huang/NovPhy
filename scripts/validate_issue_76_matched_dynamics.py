"""Recompute the complete matched-dynamics publication without fitting."""
import time

import torch

from scripts import run_issue_76_matched_dynamics as run


def main():
    torch.set_num_threads(1)
    began = time.monotonic()
    report = run.fit.files.read(run.OUTPUT / "report.json")
    plan = run.make_plan()
    if report["plan"] != plan or run.fit.files.read(run.ROOT / "plan.json") != plan:
        raise ValueError("publication or frozen source plan differs")
    rows, checkpoints = [], []
    for seed in plan["seeds"]:
        for pure in (False, True):
            path = run.fit.model_path(run.ROOT, seed, pure, "predictor")
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if (saved["plan_identity"] != plan["identity"] or not saved["complete"] or saved["failure"]
                    or (saved["updates_requested"], saved["updates_completed"], saved["updates_applied"],
                        saved["updates_skipped"]) != (6000, 6000, 5952, 48)):
                raise ValueError("predictor completion, source, or exposure differs")
            model = run.fit.new_predictor(plan, pure, "cpu").eval().requires_grad_(False)
            model.load_state_dict(saved["model"])
            old = run.fit.load_predictor(run.parent.ROOT, plan["parent_plan"], seed, pure, "cpu")
            checkpoints.append(dict(seed=seed, pure=pure, path=str(path), updates_applied=5952,
                                    updates_skipped=48))
            for entry in plan["evaluation"]:
                shard = run.fit.load_shard(run.parent.ROOT, entry, plan["parent_plan"], fitting=True)
                carrier = run.fit.load_carrier(run.parent.ROOT, seed, entry, plan["parent_plan"])
                rows.append(dict(seed=seed, pure=pure, member_identity=entry["member_identity"],
                    family=entry["family"], parent=run.recursive_probe(old, shard, carrier),
                    mixed=run.recursive_probe(model, shard, carrier)))
                if time.monotonic() - began > 180:
                    raise RuntimeError("validation exceeded 180-second CPU ceiling")
    summaries = run.summarize(plan, rows)
    if rows != report["rows"] or summaries != report["summaries"]:
        raise ValueError("recomputed complete metrics or qualification decisions differ")
    if report["qualification_supported"] != all(s["qualification_supported"] for s in summaries):
        raise ValueError("aggregate qualification differs")
    if report["fresh_access"] or report["advancement_passed"]:
        raise ValueError("training-only diagnostic cannot claim fresh access or advancement")
    for key, expected in report["budgets"].items():
        actual = run.fit.require_finished_budget(run.ROOT, key)
        if actual != expected or actual["peak_cpu_rss_mib"] > 12288 or actual["peak_cuda_allocated_mib"] > 8192:
            raise ValueError("published completed cost or resource evidence differs")
    value = dict(validated=True, plan_identity=plan["identity"], recomputed_rows=len(rows),
                 checkpoints=checkpoints, optimization_performed=False, fresh_access=False,
                 additional_validation_seconds=time.monotonic() - began,
                 qualification_supported=report["qualification_supported"])
    run.parent.previous.immutable(run.OUTPUT / "validation.json", value)
    print(value, flush=True)


if __name__ == "__main__":
    main()
