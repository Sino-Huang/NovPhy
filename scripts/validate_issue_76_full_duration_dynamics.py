"""Validate all full-duration curves, counters, optimizer history, and costs."""
from collections import Counter
from pathlib import Path

import torch

from scripts import run_issue_76_full_duration_dynamics as run


def check_optimizer_steps(model, before, after, plan, pure):
    counts = Counter(run.fit.pair_for_update(pure, u).abstraction.value
                     for u in range(plan["start_update"], plan["start_update"] + plan["updates"]["predictor"])
                     if plan["training"][u % len(plan["training"])]["usable"])
    old_ids = [i for group in before["param_groups"] for i in group["params"]]
    new_ids = [i for group in after["param_groups"] for i in group["params"]]
    names = [name for name, _ in model.named_parameters()]
    if old_ids != new_ids or len(names) != len(new_ids):
        raise ValueError("optimizer parameter registration differs from the source")
    rows = []
    for name, index in zip(names, new_ids):
        mode = ("micro" if name.startswith(("micro_head.", "micro_adapter.")) else
                "macro" if name.startswith(("macro_head.", "macro_adapter.")) else None)
        expected = counts[mode] if mode else sum(counts.values())
        prior, current = float(before["state"][index]["step"]), float(after["state"][index]["step"])
        if current - prior != expected:
            raise ValueError(f"optimizer history differs for {name}: {prior} -> {current}, expected +{expected}")
        rows.append(dict(parameter=name, source_steps=prior, completed_steps=current, stage_steps=expected))
    return rows


def main():
    torch.set_num_threads(1)
    report = run.fit.files.read(run.OUTPUT / "report.json")
    plan = run.make_plan()
    if report["plan"] != plan or run.fit.files.read(run.ROOT / "plan.json") != plan:
        raise ValueError("publication or frozen full-duration plan differs")
    references = {(r["seed"], r["pure"], r["member_identity"]): r for r in plan["reference_rows"]}
    rows, checkpoints = [], []
    with run.fit.fitting_budget(run.ROOT, "validation", 180, "cpu") as budget:
        for seed in plan["seeds"]:
            for pure in (False, True):
                saved = torch.load(run.fit.model_path(run.ROOT, seed, pure, "predictor"), map_location="cpu", weights_only=False)
                if (saved["plan_identity"] != plan["identity"] or not saved["complete"] or saved["failure"]
                        or (saved["updates_requested"], saved["updates_completed"], saved["updates_applied"],
                            saved["updates_skipped"]) != (1800, 1800, 1786, 14)):
                    raise ValueError("full-duration checkpoint source, completion, or exposure differs")
                before = torch.load(plan["source_checkpoint_paths"][f"{seed}-{pure}"], map_location="cpu", weights_only=False)
                run.initial_checkpoint(before, plan)
                model = run.fit.new_predictor(plan, pure, "cpu").eval().requires_grad_(False)
                model.load_state_dict(saved["model"])
                steps = check_optimizer_steps(model, before["optimizer"], saved["optimizer"], plan, pure)
                checkpoints.append(dict(seed=seed, pure=pure, updates_applied=1786, updates_skipped=14,
                                        optimizer_steps=steps))
                for entry in plan["evaluation"]:
                    budget.check()
                    shard = run.fit.load_shard(Path(plan["data_root"]), entry, plan["data_binding"], fitting=True)
                    carrier = run.fit.load_carrier(Path(plan["data_root"]), seed, entry, plan["data_binding"])
                    rows.append({**references[seed, pure, entry["member_identity"]],
                                 "full_duration": run.source.drift_curves(model, shard, carrier)})
        summaries = run.summarize(plan, rows)
        if rows != report["rows"] or summaries != report["summaries"]:
            raise ValueError("complete recomputed curves or qualification decisions differ")
        if report["qualification_supported"] != all(s["qualification_supported"] for s in summaries):
            raise ValueError("aggregate full-duration qualification differs")
        if report["fresh_access"] or report["advancement_passed"]:
            raise ValueError("training-only diagnostic cannot claim fresh access or advancement")
        for key, expected in report["budgets"].items():
            actual = run.fit.require_finished_budget(run.ROOT, key)
            if actual != expected or actual["peak_cpu_rss_mib"] > 12288 or actual["peak_cuda_allocated_mib"] > 8192:
                raise ValueError("published cost or resource evidence differs")
        budget.check()
    value = dict(validated=True, plan_identity=plan["identity"], recomputed_rows=len(rows),
        recomputed_policy_curves=sum(len(r["full_duration"]) for r in rows), checkpoints=checkpoints,
        optimization_performed=False, fresh_access=False, qualification_supported=report["qualification_supported"],
        additional_validation_budget_path=str(run.ROOT / "budgets/validation.json"))
    run.source.previous.parent.previous.immutable(run.OUTPUT / "validation.json", value)
    print({k: v for k, v in value.items() if k != "checkpoints"}, flush=True)


if __name__ == "__main__":
    main()
