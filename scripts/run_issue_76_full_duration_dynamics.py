"""Paired, source-bound continuation with full-duration recursive supervision."""
import argparse
import gc
from pathlib import Path

import torch

from scripts import run_issue_76_boundary_dynamics as source
from scripts.issue_76_full_duration_loss import trajectory_batch, full_continuous_loss, full_hybrid_loss

fit = source.fit
ROOT = fit.files.ROOT / ".local-artifacts/issue-76-full-duration-dynamics-v1"
OUTPUT = fit.files.ROOT / "data/issue-76-full-duration-dynamics"
SOURCES = ("scripts/run_issue_76_full_duration_dynamics.py", "scripts/issue_76_full_duration_loss.py",
           "scripts/run_issue_76_boundary_dynamics.py", "scripts/issue_76_matched_batches.py",
           "scripts/diagnose_issue_76_hybrid_drift.py", "docs/issue-76-full-duration-dynamics-protocol.md")


def continuation_schedule(lengths, starts, seed, pure, begin, updates):
    original = source.previous.sampled_schedule(lengths, seed, pure, begin + updates)
    original = {u: rows for u, rows in original.items() if u >= begin}
    return source.previous.mixed_schedule(source.boundary_schedule(original, starts, seed), seed, pure)


def initial_checkpoint(checkpoint, plan):
    if (checkpoint["plan_identity"] != plan["source_plan_identity"] or not checkpoint["complete"]
            or checkpoint["failure"] or (checkpoint["updates_completed"], checkpoint["updates_applied"],
                                         checkpoint["updates_skipped"]) != (6000, 5952, 48)):
        raise ValueError("continuation requires the complete bound source predictor")
    return {**checkpoint, "plan_identity": plan["identity"], "updates_requested": plan["updates"]["predictor"],
            "updates_completed": 0, "updates_applied": 0, "updates_skipped": 0,
            "complete": False, "failure": None, "last_loss": None}


def make_plan():
    parent = fit.files.read(source.ROOT / "plan.json")
    report = fit.files.read(source.OUTPUT / "report.json")
    validation = fit.files.read(source.OUTPUT / "validation.json")
    probe = fit.files.read(fit.files.ROOT / "data/issue-76-full-duration-resource-probe/report.json")
    if (report["plan"] != parent or not validation["validated"]
            or validation["plan_identity"] != parent["identity"]
            or probe["source_plan_identity"] != parent["identity"] or not probe["weights_unchanged"]):
        raise ValueError("requires validated boundary predictors and their no-update resource probe")
    references = [{k: r[k] for k in ("seed", "pure", "member_identity", "family", "boundary", "reference")}
                  for r in report["rows"]]
    return dict(identity=ROOT.name, source_plan_identity=parent["identity"], source_root=str(source.ROOT),
        source_checkpoint_paths={f"{seed}-{pure}": str(fit.model_path(source.ROOT, seed, pure, "predictor"))
                                 for seed in parent["seeds"] for pure in (False, True)},
        data_root=parent["source_root"], data_binding=parent["data_binding"],
        training=parent["training"], evaluation=parent["evaluation"], seeds=parent["seeds"], capacity=parent["capacity"],
        reference_rows=references, reused_common_costs=parent["reused_common_costs"], prior_stage_costs=report["budgets"],
        resource_probe=probe, start_update=6000, updates={"predictor": 1800}, expected_applied=1786,
        expected_skipped=14, batch_size=32, learning_rate=.0003, optimizer_state_retained=True,
        predictor_seconds_per_arm_seed=1800, preparation_seconds_per_seed=300, diagnostic_seconds=600,
        new_artifact_bytes=2**30, unchanged_mse_ratio=.8, required_unchanged_wins=4, retention_ratio=1.1,
        fresh_access=False, advancement_authorized=False,
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
                shard = fit.load_shard(Path(plan["data_root"]), entry, plan["data_binding"], fitting=True) if entry["usable"] else None
                shards.append(shard)
                carriers.append(fit.load_carrier(Path(plan["data_root"]), seed, entry, plan["data_binding"]) if entry["usable"] else None)
                starts.append([s["start"] for s in shard["segment_ranges"]] if shard else [])
        for pure in (False, True):
            if fit.retain_completed_fit(ROOT, plan, seed, pure, "predictor"):
                continue
            name = "pure" if pure else "hybrid"
            with fit.fitting_budget(ROOT, f"predictor-{name}-{seed}", plan["predictor_seconds_per_arm_seed"], device) as budget:
                schedule = continuation_schedule(lengths, starts, seed, pure, plan["start_update"], plan["updates"]["predictor"])
                path = fit.model_path(ROOT, seed, pure, "predictor")
                if not path.exists():
                    checkpoint = torch.load(plan["source_checkpoint_paths"][f"{seed}-{pure}"], map_location="cpu", weights_only=False)
                    fit.atomic_torch(path, initial_checkpoint(checkpoint, plan))
                    del checkpoint
                torch.manual_seed(seed)
                model = fit.new_predictor(plan, pure, device)

                def loss(update):
                    absolute = plan["start_update"] + update
                    if absolute not in schedule:
                        return None
                    pair = fit.pair_for_update(pure, absolute)
                    batch = trajectory_batch(shards, carriers, schedule[absolute], pair.delta, symbolic=not pure)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    if pure:
                        return full_continuous_loss(model, batch["z"], batch["available"], batch["action"], pair)
                    return full_hybrid_loss(model, batch, pair)

                result = fit.fit_updates(path, model, plan_identity=plan["identity"],
                    updates=plan["updates"]["predictor"], lr=plan["learning_rate"], make_loss=loss, budget=budget, device=device)
                if (result["updates_applied"], result["updates_skipped"]) != (plan["expected_applied"], plan["expected_skipped"]):
                    raise ValueError("full-duration stage exposure counts differ")
            del model, result, schedule
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        del shards, carriers, starts
        gc.collect()
        if sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()) > plan["new_artifact_bytes"]:
            raise fit.FitBudgetExceeded("full-duration artifacts exceed 1 GiB")


def summarize(plan, rows):
    summaries = []
    for seed in plan["seeds"]:
        for pure in (False, True):
            local = [r for r in rows if r["seed"] == seed and r["pure"] == pure]
            policies = {}
            for name in local[0]["full_duration"]:
                ends = [next(p for p in r["full_duration"][name] if p["elapsed"] == fit.ENDPOINT) for r in local]
                available = len(local) == len(plan["evaluation"]) and all(p["available"] for p in ends)
                if not available:
                    policies[name] = dict(qualification_supported=False, reason="missing assigned endpoint")
                    continue
                mse = sum(p["recursive_mse"] for p in ends) / len(ends)
                unchanged = sum(p["unchanged_mse"] for p in ends) / len(ends)
                wins = sum(p["recursive_mse"] < p["unchanged_mse"] for p in ends)
                policies[name] = dict(mean_carrier_mse=mse, unchanged_mse=unchanged, unchanged_wins=wins,
                    qualification_supported=mse <= plan["unchanged_mse_ratio"] * unchanged
                    and wins >= plan["required_unchanged_wins"])
            anchor = "fixed-750-continuous" if pure else "fixed-750-micro"
            anchors = {kind: {str(t): [next(p for p in r[kind][anchor] if p["elapsed"] == t) for r in local]
                              for t in (750, fit.ENDPOINT)} for kind in ("full_duration", "boundary", "reference")}
            if not all(p["available"] for points in anchors.values() for values in points.values() for p in values):
                summaries.append(dict(seed=seed, pure=pure, policies=policies, qualification_supported=False,
                                      reason="missing assigned anchor endpoint"))
                continue
            metrics = {kind: {t: sum(p["recursive_mse"] for p in values) / len(values)
                              for t, values in points.items()} for kind, points in anchors.items()}
            reference = min(metrics[k][str(fit.ENDPOINT)] for k in (("boundary", "reference") if pure else ("boundary",)))
            initial = metrics["full_duration"]["750"] <= plan["retention_ratio"] * metrics["boundary"]["750"]
            endpoint = metrics["full_duration"][str(fit.ENDPOINT)] <= plan["retention_ratio"] * reference
            summaries.append(dict(seed=seed, pure=pure, policies=policies, anchor_metrics=metrics,
                anchor_policy=anchor, initial_accuracy_retained=initial, endpoint_accuracy_retained=endpoint,
                qualification_supported=initial and endpoint and all(p["qualification_supported"] for p in policies.values())))
    return summaries


def diagnose(plan):
    if (OUTPUT / "report.json").exists():
        return
    references = {(r["seed"], r["pure"], r["member_identity"]): r for r in plan["reference_rows"]}
    rows = []
    with fit.fitting_budget(ROOT, "diagnostic", plan["diagnostic_seconds"], "cpu") as budget:
        for seed in plan["seeds"]:
            for pure in (False, True):
                fit.require_finished_budget(ROOT, f"predictor-{'pure' if pure else 'hybrid'}-{seed}")
                saved = torch.load(fit.model_path(ROOT, seed, pure, "predictor"), map_location="cpu", weights_only=False)
                if not saved["complete"] or saved["plan_identity"] != plan["identity"]:
                    raise ValueError("diagnostic requires the complete bound continuation")
                model = fit.new_predictor(plan, pure, "cpu").eval().requires_grad_(False)
                model.load_state_dict(saved["model"])
                for entry in plan["evaluation"]:
                    budget.check()
                    shard = fit.load_shard(Path(plan["data_root"]), entry, plan["data_binding"], fitting=True)
                    carrier = fit.load_carrier(Path(plan["data_root"]), seed, entry, plan["data_binding"])
                    rows.append({**references[seed, pure, entry["member_identity"]],
                                 "full_duration": source.drift_curves(model, shard, carrier)})
                    fit.log(f"full-duration qualification {len(rows)}/30")
    summaries = summarize(plan, rows)
    source.previous.parent.previous.immutable(OUTPUT / "report.json", dict(plan=plan, rows=rows, summaries=summaries,
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
        source.previous.parent.previous.immutable(ROOT / "plan.json", plan)
        source.previous.parent.previous.immutable(OUTPUT / "plan.json", plan)
        fit.log("full-duration plan frozen; no optimizer updates performed")
    elif args.run:
        if fit.files.read(ROOT / "plan.json") != plan:
            raise ValueError("full-duration frozen plan differs")
        train(plan, args.device)
        diagnose(plan)
    else:
        fit.log("full-duration no-write plan check passed")


if __name__ == "__main__":
    main()
