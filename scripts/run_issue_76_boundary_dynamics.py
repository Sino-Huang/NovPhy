"""Paired predictor refit with half shot-start, half uniform training starts."""
import argparse
import gc

import torch

from scripts import run_issue_76_matched_dynamics as previous
from scripts.diagnose_issue_76_hybrid_drift import drift_curves

fit = previous.fit
ROOT = fit.files.ROOT / ".local-artifacts/issue-76-boundary-dynamics-v1"
OUTPUT = fit.files.ROOT / "data/issue-76-boundary-dynamics"
SOURCES = ("scripts/run_issue_76_boundary_dynamics.py", "scripts/issue_76_matched_batches.py",
           "scripts/diagnose_issue_76_hybrid_drift.py", "scripts/run_issue_76_native_refit.py",
           "scripts/issue_76_fit_budget.py", "world_model/training/native_history_fit.py",
           "world_model/training/native_history_model.py", "docs/issue-76-boundary-dynamics-protocol.md")


def boundary_schedule(original, starts, seed):
    """Replace even rows before mixing; odd rows and lineage assignments remain."""
    schedule = {}
    for update, rows in original.items():
        changed = rows.clone()
        entry = int(rows[0, 0])
        boundaries = torch.tensor(starts[entry], dtype=torch.long)
        indices = torch.randint(len(boundaries), (len(rows[::2]),),
                                generator=fit.generator(seed, 1000000 + update))
        changed[::2, 1] = boundaries[indices]
        schedule[update] = changed
    return schedule


def make_plan():
    source = fit.files.read(previous.parent.ROOT / "plan.json")
    index = fit.data_index(previous.parent.ROOT, source)
    baseline = fit.files.read(previous.OUTPUT / "report.json")
    validation = fit.files.read(previous.OUTPUT / "validation.json")
    curves = fit.files.read(fit.files.ROOT / "data/issue-76-hybrid-drift-diagnostic/report.json")
    if (not validation["validated"] or validation["plan_identity"] != baseline["plan"]["identity"]
            or curves["source_plan_identity"] != baseline["plan"]["identity"]):
        raise ValueError("requires the validated matched-dynamics diagnostic")
    training = [e for e in index["entries"] if e["exposure_role"] == "training"]
    selected = {}
    for entry in training:
        selected.setdefault(entry["family"], entry)
    references = [{k: row[k] for k in ("seed", "pure", "member_identity", "family", "curves")}
                  for row in curves["rows"]]
    return dict(identity=ROOT.name, source_root=str(previous.parent.ROOT),
        source_plan_path=str(previous.parent.ROOT / "plan.json"),
        data_binding={k: source[k] for k in ("identity", "synthetic")},
        reference_plan_identity=baseline["plan"]["identity"], reference_rows=references,
        training=training, evaluation=list(selected.values()), seeds=source["seeds"], capacity=source["capacity"],
        updates={"predictor": 6000}, batch_size=32, learning_rate=.0003,
        predictor_seconds_per_arm_seed=1800, preparation_seconds_per_seed=300, diagnostic_seconds=600,
        new_artifact_bytes=2**30, first_step_ratio=.5, hybrid_endpoint_ratio=.8, pure_endpoint_ratio=1.1,
        required_unchanged_wins=4, fresh_access=False, advancement_authorized=False,
        reused_common_costs={str(seed): fit.require_finished_budget(previous.parent.ROOT, f"common-{seed}")
                             for seed in source["seeds"]},
        source_text={s: (fit.files.ROOT / s).read_text() for s in dict.fromkeys((*fit.SOURCES, *SOURCES))})


def train(plan, device):
    lengths = [e["frames"] if e["usable"] else 0 for e in plan["training"]]
    for seed in plan["seeds"]:
        if all(fit.retain_completed_fit(ROOT, plan, seed, pure, "predictor") for pure in (False, True)):
            continue
        with fit.fitting_budget(ROOT, f"preparation-{seed}", plan["preparation_seconds_per_seed"], "cpu") as budget:
            shards, carriers, starts = [], [], []
            for entry in plan["training"]:
                budget.check()
                shard = fit.load_shard(previous.parent.ROOT, entry, plan["data_binding"], fitting=True) if entry["usable"] else None
                shards.append(shard)
                carriers.append(fit.load_carrier(previous.parent.ROOT, seed, entry, plan["data_binding"]) if entry["usable"] else None)
                starts.append([s["start"] for s in shard["segment_ranges"]] if shard else [])
        for pure in (False, True):
            if fit.retain_completed_fit(ROOT, plan, seed, pure, "predictor"):
                continue
            name = "pure" if pure else "hybrid"
            with fit.fitting_budget(ROOT, f"predictor-{name}-{seed}", plan["predictor_seconds_per_arm_seed"], device) as budget:
                original = previous.sampled_schedule(lengths, seed, pure, plan["updates"]["predictor"], plan["batch_size"])
                schedule = previous.mixed_schedule(boundary_schedule(original, starts, seed), seed, pure)
                torch.manual_seed(seed)
                model = fit.new_predictor(plan, pure, device)

                def loss(update):
                    if update not in schedule:
                        return None
                    pair = fit.pair_for_update(pure, update)
                    batch = previous.indexed_batch(shards, carriers, schedule[update], pair.delta, symbolic=not pure)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    if pure:
                        return fit.continuous_loss(model, batch["z"], batch["available"], batch["action"], pair)
                    return fit.hybrid_loss(model, batch, pair)

                result = fit.fit_updates(fit.model_path(ROOT, seed, pure, "predictor"), model,
                    plan_identity=plan["identity"], updates=plan["updates"]["predictor"],
                    lr=plan["learning_rate"], make_loss=loss, budget=budget, device=device)
                if (result["updates_applied"], result["updates_skipped"]) != (5952, 48):
                    raise ValueError("boundary predictor update counts differ")
            del model, result, original, schedule
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        del shards, carriers, starts
        gc.collect()
        if sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()) > plan["new_artifact_bytes"]:
            raise fit.FitBudgetExceeded("boundary dynamics artifacts exceed 1 GiB")


def summarize(plan, rows):
    summaries = []
    for seed in plan["seeds"]:
        for pure in (False, True):
            local = [r for r in rows if r["seed"] == seed and r["pure"] == pure]
            policy = "fixed-750-continuous" if pure else "fixed-750-micro"
            records = {kind: {t: [next(p for p in row[kind][policy] if p["elapsed"] == t)
                                 for row in local] for t in (750, fit.ENDPOINT)}
                       for kind in ("reference", "boundary")}
            if len(local) != len(plan["evaluation"]) or not all(
                    p["available"] for points in records.values() for values in points.values() for p in values):
                summaries.append(dict(seed=seed, pure=pure, qualification_supported=False, reason="missing assigned endpoint"))
                continue
            means = {kind: {str(t): sum(p["recursive_mse"] for p in points) / len(points)
                            for t, points in by_time.items()} for kind, by_time in records.items()}
            wins = sum(p["recursive_mse"] < p["unchanged_mse"] for p in records["boundary"][fit.ENDPOINT])
            first = means["boundary"]["750"] <= plan["first_step_ratio"] * means["reference"]["750"]
            endpoint = means["boundary"][str(fit.ENDPOINT)] <= (
                plan["pure_endpoint_ratio"] if pure else plan["hybrid_endpoint_ratio"]) * means["reference"][str(fit.ENDPOINT)]
            qualified = endpoint and (pure or (first and wins >= plan["required_unchanged_wins"]))
            summaries.append(dict(seed=seed, pure=pure, policy=policy, mean_carrier_mse=means,
                unchanged_wins=wins, first_step_hypothesis_supported=first,
                endpoint_contrast_supported=endpoint, qualification_supported=qualified))
    return summaries


def diagnose(plan):
    if (OUTPUT / "report.json").exists():
        return
    references = {(r["seed"], r["pure"], r["member_identity"]): r["curves"] for r in plan["reference_rows"]}
    rows = []
    with fit.fitting_budget(ROOT, "diagnostic", plan["diagnostic_seconds"], "cpu") as budget:
        for seed in plan["seeds"]:
            for pure in (False, True):
                fit.require_finished_budget(ROOT, f"predictor-{'pure' if pure else 'hybrid'}-{seed}")
                saved = torch.load(fit.model_path(ROOT, seed, pure, "predictor"), map_location="cpu", weights_only=False)
                if not saved["complete"] or saved["plan_identity"] != plan["identity"]:
                    raise ValueError("diagnostic requires the completed source-bound predictor")
                model = fit.new_predictor(plan, pure, "cpu").eval().requires_grad_(False)
                model.load_state_dict(saved["model"])
                for entry in plan["evaluation"]:
                    budget.check()
                    shard = fit.load_shard(previous.parent.ROOT, entry, plan["data_binding"], fitting=True)
                    carrier = fit.load_carrier(previous.parent.ROOT, seed, entry, plan["data_binding"])
                    rows.append(dict(seed=seed, pure=pure, member_identity=entry["member_identity"], family=entry["family"],
                        reference=references[seed, pure, entry["member_identity"]], boundary=drift_curves(model, shard, carrier)))
                    fit.log(f"boundary dynamics qualification {len(rows)}/30")
    summaries = summarize(plan, rows)
    previous.parent.previous.immutable(OUTPUT / "report.json", dict(plan=plan, rows=rows, summaries=summaries,
        budgets={p.stem: fit.files.read(p) for p in sorted((ROOT / "budgets").glob("*.json"))},
        qualification_supported=all(s["qualification_supported"] for s in summaries),
        fresh_access=False, advancement_passed=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = make_plan()
    if args.prepare:
        previous.parent.previous.immutable(ROOT / "plan.json", plan)
        previous.parent.previous.immutable(OUTPUT / "plan.json", plan)
        fit.log("boundary dynamics plan frozen; no fitting performed")
    elif args.run:
        if fit.files.read(ROOT / "plan.json") != plan:
            raise ValueError("frozen boundary dynamics plan differs")
        train(plan, args.device)
        diagnose(plan)
    else:
        fit.log("no-write boundary plan check passed")


if __name__ == "__main__":
    main()
