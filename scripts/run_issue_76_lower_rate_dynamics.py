"""Paired full-duration continuation with an explicitly reduced AdamW rate."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch

from scripts import run_issue_76_full_duration_dynamics as source

fit = source.fit
ROOT = fit.files.ROOT / ".local-artifacts/issue-76-lower-rate-dynamics-v1"
OUTPUT = fit.files.ROOT / "data/issue-76-lower-rate-dynamics"
immutable = source.source.previous.parent.previous.immutable
SOURCES = ("scripts/run_issue_76_lower_rate_dynamics.py", "docs/issue-76-lower-rate-dynamics-protocol.md")


def initial_checkpoint(checkpoint, plan):
    if (checkpoint["plan_identity"] != plan["source_plan_identity"] or not checkpoint["complete"]
            or checkpoint["failure"] or (checkpoint["updates_completed"], checkpoint["updates_applied"],
                                         checkpoint["updates_skipped"]) != (1800, 1786, 14)):
        raise ValueError("requires the completed full-duration source checkpoint")
    if any(group["lr"] != .0003 for group in checkpoint["optimizer"]["param_groups"]):
        raise ValueError("source optimizer rate differs from the frozen full-duration stage")
    optimizer = {**checkpoint["optimizer"], "param_groups": [
        {**group, "lr": plan["learning_rate"]} for group in checkpoint["optimizer"]["param_groups"]]}
    return {**checkpoint, "optimizer": optimizer, "plan_identity": plan["identity"],
            "updates_requested": plan["updates"]["predictor"], "updates_completed": 0,
            "updates_applied": 0, "updates_skipped": 0, "complete": False, "failure": None, "last_loss": None}


def make_plan():
    parent = fit.files.read(source.ROOT / "plan.json")
    report = fit.files.read(source.OUTPUT / "report.json")
    validation = fit.files.read(source.OUTPUT / "validation.json")
    probe = fit.files.read(fit.files.ROOT / "data/issue-76-finite-step/report.json")
    if (parent != report["plan"] or not validation["validated"]
            or validation["plan_identity"] != parent["identity"]
            or probe["source_plan_identity"] != parent["identity"] or probe["checkpoint_writes"]):
        raise ValueError("requires validated full-duration predictors and the no-checkpoint-write step probe")
    return {**parent, "identity": ROOT.name, "source_plan_identity": parent["identity"],
        "source_root": str(source.ROOT), "source_plan": parent,
        "source_checkpoint_paths": {f"{seed}-{pure}": str(fit.model_path(source.ROOT, seed, pure, "predictor"))
                                    for seed in parent["seeds"] for pure in (False, True)},
        "reference_rows": report["rows"], "prior_stage_costs": report["budgets"],
        "prior_validation_cost": fit.files.read(source.OUTPUT / "validation-budget.json"),
        "finite_step_probe": probe, "start_update": 7800, "updates": {"predictor": 6000},
        "gradient_tradeoff_probe": fit.files.read(fit.files.ROOT / "data/issue-76-gradient-tradeoff/report.json"),
        "expected_applied": 5952, "expected_skipped": 48, "learning_rate": .00003,
        "optimizer_moments_retained": True, "predictor_seconds_per_arm_seed": 3600,
        "audit_seconds": 300, "checkpoint_preflight_seconds": 180, "pure_retains_full_duration_reference": True,
        "source_text": {**parent["source_text"], **{p: (fit.files.ROOT / p).read_text() for p in SOURCES}}}


def exposure_audit(plan):
    lengths = [e["frames"] if e["usable"] else 0 for e in plan["training"]]
    rows = []
    with fit.fitting_budget(ROOT, "exposure-audit", plan["audit_seconds"], "cpu") as budget:
        starts = []
        for entry in plan["training"]:
            budget.check()
            shard = fit.load_shard(Path(plan["data_root"]), entry, plan["data_binding"], fitting=True) if entry["usable"] else None
            starts.append([s["start"] for s in shard["segment_ranges"]] if shard else [])
        for seed in plan["seeds"]:
            horizon_totals = []
            for pure in (False, True):
                budget.check()
                schedule = source.continuation_schedule(lengths, starts, seed, pure, plan["start_update"], 6000)
                expected = {u for u in range(7800, 13800) if lengths[u % len(lengths)]}
                if set(schedule) != expected or len(schedule) != 5952:
                    raise ValueError("lower-rate absolute update inventory differs")
                totals, counts = defaultdict(Counter), Counter()
                for u, values in schedule.items():
                    pair = fit.pair_for_update(pure, u)
                    totals[pair.delta].update(map(tuple, values.tolist()))
                    counts[fit.policy_name(pair)] += 1
                horizon_totals.append(totals)
                rows.append(dict(seed=seed, pure=pure, scheduled=6000, applied=len(schedule), skipped=48,
                                 sampled_starts=sum(sum(c.values()) for c in totals.values()), pair_updates=dict(counts)))
            if horizon_totals[0] != horizon_totals[1]:
                raise ValueError("paired horizon-level lineage/start multiplicities differ")
            fit.log(f"lower-rate paired exposure audited seed {seed}")
        budget.check()
    immutable(OUTPUT / "exposure.json", dict(plan_identity=plan["identity"], rows=rows,
              paired_multiplicities_equal=True, budget=fit.require_finished_budget(ROOT, "exposure-audit")))


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
            with fit.fitting_budget(ROOT, f"predictor-{'pure' if pure else 'hybrid'}-{seed}",
                                    plan["predictor_seconds_per_arm_seed"], device) as budget:
                schedule = source.continuation_schedule(lengths, starts, seed, pure, plan["start_update"], 6000)
                path = fit.model_path(ROOT, seed, pure, "predictor")
                if not path.exists():
                    saved = torch.load(plan["source_checkpoint_paths"][f"{seed}-{pure}"], map_location="cpu", weights_only=False)
                    fit.atomic_torch(path, initial_checkpoint(saved, plan))
                    del saved
                torch.manual_seed(seed)
                model = fit.new_predictor(plan, pure, device)

                def loss(update):
                    absolute = plan["start_update"] + update
                    if absolute not in schedule:
                        return None
                    pair = fit.pair_for_update(pure, absolute)
                    batch = source.trajectory_batch(shards, carriers, schedule[absolute], pair.delta, symbolic=not pure)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    return (source.full_continuous_loss(model, batch["z"], batch["available"], batch["action"], pair)
                            if pure else source.full_hybrid_loss(model, batch, pair))

                result = fit.fit_updates(path, model, plan_identity=plan["identity"], updates=6000,
                                        lr=plan["learning_rate"], make_loss=loss, budget=budget, device=device)
                if (result["updates_applied"], result["updates_skipped"]) != (5952, 48):
                    raise ValueError("lower-rate stage exposure differs")
                if any(g["lr"] != plan["learning_rate"] for g in result["optimizer"]["param_groups"]):
                    raise ValueError("loaded optimizer did not retain the reduced learning rate")
            del model, result, schedule
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        if sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()) > plan["new_artifact_bytes"]:
            raise fit.FitBudgetExceeded("lower-rate artifacts exceed 1 GiB")


def summarize(plan, rows):
    summaries = source.summarize(plan, [{**r, "full_duration": r["lower_rate"]} for r in rows])
    for summary in summaries:
        summary["anchor_metrics"]["lower_rate"] = summary["anchor_metrics"].pop("full_duration")
        if summary["pure"]:
            local = [r for r in rows if r["seed"] == summary["seed"] and r["pure"]]
            points = [next(p for p in r["full_duration"][summary["anchor_policy"]] if p["elapsed"] == fit.ENDPOINT) for r in local]
            mean = sum(p["recursive_mse"] for p in points) / len(points)
            retained = all(p["available"] for p in points) and summary["anchor_metrics"]["lower_rate"][str(fit.ENDPOINT)] <= plan["retention_ratio"] * mean
            summary.update(full_duration_pure_endpoint_mse=mean, full_duration_pure_accuracy_retained=retained,
                           qualification_supported=summary["qualification_supported"] and retained)
    return summaries


def diagnose(plan):
    references = {(r["seed"], r["pure"], r["member_identity"]): r for r in plan["reference_rows"]}
    rows = []
    with fit.fitting_budget(ROOT, "diagnostic", plan["diagnostic_seconds"], "cpu") as budget:
        for seed in plan["seeds"]:
            for pure in (False, True):
                fit.require_finished_budget(ROOT, f"predictor-{'pure' if pure else 'hybrid'}-{seed}")
                saved = torch.load(fit.model_path(ROOT, seed, pure, "predictor"), map_location="cpu", weights_only=False)
                if not saved["complete"] or saved["plan_identity"] != plan["identity"]:
                    raise ValueError("diagnostic requires the completed lower-rate predictor")
                model = fit.new_predictor(plan, pure, "cpu").eval().requires_grad_(False)
                model.load_state_dict(saved["model"])
                for entry in plan["evaluation"]:
                    budget.check()
                    shard = fit.load_shard(Path(plan["data_root"]), entry, plan["data_binding"], fitting=True)
                    carrier = fit.load_carrier(Path(plan["data_root"]), seed, entry, plan["data_binding"])
                    rows.append({**references[seed, pure, entry["member_identity"]],
                                 "lower_rate": source.source.drift_curves(model, shard, carrier)})
                    fit.log(f"lower-rate qualification {len(rows)}/30")
    summaries = summarize(plan, rows)
    immutable(OUTPUT / "report.json", dict(plan=plan, rows=rows, summaries=summaries,
              budgets={p.stem: fit.files.read(p) for p in sorted((ROOT / "budgets").glob("*.json"))},
              qualification_supported=all(s["qualification_supported"] for s in summaries),
              fresh_access=False, advancement_passed=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = make_plan()
    if args.audit:
        exposure_audit(plan)
    elif args.prepare:
        exposure = fit.files.read(OUTPUT / "exposure.json")
        if exposure["plan_identity"] != plan["identity"] or not exposure["paired_multiplicities_equal"]:
            raise ValueError("requires the completed paired exposure audit")
        preflight = fit.files.read(OUTPUT / "checkpoint-preflight.json")
        if (preflight["plan_identity"] != plan["identity"] or len(preflight["rows"]) != 6
                or not all(r["model_and_moments_retained"] and r["new_lr"] == plan["learning_rate"] for r in preflight["rows"])):
            raise ValueError("requires all six source-checkpoint preflight checks")
        immutable(ROOT / "plan.json", plan)
        immutable(OUTPUT / "plan.json", plan)
        fit.log("lower-rate plan frozen; no model fitting performed")
    elif args.run:
        if fit.files.read(ROOT / "plan.json") != plan:
            raise ValueError("frozen lower-rate plan differs")
        train(plan, args.device)
        if not (OUTPUT / "report.json").exists():
            diagnose(plan)
    else:
        fit.log("lower-rate no-write plan check passed")


if __name__ == "__main__":
    main()
