"""Continue paired lower-rate predictors using only observed shot-boundary starts."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch

from scripts import run_issue_76_lower_rate_dynamics as source
from scripts import issue_76_matched_batches as batches

fit = source.fit
ROOT = fit.files.ROOT / ".local-artifacts/issue-76-boundary-focus-v1"
OUTPUT = fit.files.ROOT / "data/issue-76-boundary-focus"
SOURCES = ("scripts/run_issue_76_boundary_focus.py", "docs/issue-76-boundary-focus-protocol.md")


def schedule(starts, seed, pure, begin, updates):
    original = {}
    for update in range(begin, begin + updates):
        entry = update % len(starts)
        if not starts[entry]:
            continue
        boundaries = torch.tensor(starts[entry])
        rows = torch.full((32, 2), entry, dtype=torch.long)
        for offset in (0, 1):
            indices = torch.randint(len(boundaries), (16,),
                                    generator=fit.generator(seed, (offset + 1) * 1000000 + update))
            rows[offset::2, 1] = boundaries[indices]
        original[update] = rows
    return batches.mixed_schedule(original, seed, pure)


def initial_checkpoint(saved, plan):
    if (saved["plan_identity"] != plan["source_plan_identity"] or not saved["complete"] or saved["failure"]
            or (saved["updates_requested"], saved["updates_completed"], saved["updates_applied"],
                saved["updates_skipped"]) != (6000, 6000, 5952, 48)):
        raise ValueError("requires the completed lower-rate source checkpoint")
    if any(group["lr"] != plan["learning_rate"] for group in saved["optimizer"]["param_groups"]):
        raise ValueError("source learning rate differs")
    return {**saved, "plan_identity": plan["identity"], "updates_requested": plan["updates"]["predictor"],
            "updates_completed": 0, "updates_applied": 0, "updates_skipped": 0,
            "complete": False, "failure": None, "last_loss": None}


def make_plan():
    parent = fit.files.read(source.ROOT / "plan.json")
    report = fit.files.read(source.OUTPUT / "report.json")
    validation = fit.files.read(source.OUTPUT / "validation.json")
    probe = fit.files.read(fit.files.ROOT / "data/issue-76-local-interventions/report.json")
    if (parent != report["plan"] or not validation["validated"] or validation["plan_identity"] != parent["identity"]
            or probe["source_plan_identity"] != parent["identity"] or probe["checkpoint_writes"]):
        raise ValueError("requires validated lower-rate predictors and paired local intervention evidence")
    return {**parent, "identity": ROOT.name, "source_plan_identity": parent["identity"],
        "source_root": str(source.ROOT), "source_plan": parent,
        "source_checkpoint_paths": {f"{seed}-{pure}": str(fit.model_path(source.ROOT, seed, pure, "predictor"))
                                    for seed in parent["seeds"] for pure in (False, True)},
        "reference_rows": report["rows"], "prior_stage_costs": report["budgets"],
        "prior_validation_cost": fit.files.read(source.OUTPUT / "validation-budget.json"),
        "local_interventions": probe, "start_update": 13800, "updates": {"predictor": 1800},
        "expected_applied": 1785, "expected_skipped": 15, "boundary_fraction": 1.,
        "predictor_seconds_per_arm_seed": 1800, "retain_lower_rate_anchor_accuracy": True,
        "source_text": {**parent["source_text"], **{p: (fit.files.ROOT / p).read_text() for p in SOURCES}}}


def prepare(plan):
    rows = []
    with fit.fitting_budget(ROOT, "preflight", 300, "cpu") as budget:
        starts = []
        for entry in plan["training"]:
            budget.check()
            shard = fit.load_shard(Path(plan["data_root"]), entry, plan["data_binding"], fitting=True) if entry["usable"] else None
            starts.append([s["start"] for s in shard["segment_ranges"]] if shard else [])
        for seed in plan["seeds"]:
            totals = []
            for pure in (False, True):
                budget.check()
                sampled = schedule(starts, seed, pure, plan["start_update"], plan["updates"]["predictor"])
                expected = {u for u in range(13800, 15600) if plan["training"][u % len(starts)]["usable"]}
                if set(sampled) != expected or len(sampled) != 1785:
                    raise ValueError("boundary-focus assigned update inventory differs")
                exposures, counts = defaultdict(Counter), Counter()
                for update, values in sampled.items():
                    pair = fit.pair_for_update(pure, update)
                    if any(start not in starts[entry] for entry, start in values.tolist()):
                        raise ValueError("non-boundary start in focused schedule")
                    exposures[pair.delta].update(map(tuple, values.tolist()))
                    counts[fit.policy_name(pair)] += 1
                totals.append(exposures)
                before = torch.load(plan["source_checkpoint_paths"][f"{seed}-{pure}"], map_location="cpu", weights_only=False)
                initial = initial_checkpoint(before, plan)
                assert initial["model"] is before["model"] and initial["optimizer"] is before["optimizer"]
                rows.append(dict(seed=seed, pure=pure, scheduled=1800, applied=len(sampled), skipped=15,
                                 sampled_starts=sum(sum(c.values()) for c in exposures.values()),
                                 pair_updates=dict(counts), all_starts_observed_boundaries=True,
                                 source_weights_and_optimizer_retained=True))
                del before, initial
            if totals[0] != totals[1]:
                raise ValueError("paired horizon-level exposure multiplicities differ")
            fit.log(f"boundary-focus preflight seed {seed}")
        budget.check()
    source.immutable(OUTPUT / "preflight.json", dict(plan_identity=plan["identity"], rows=rows,
                     paired_multiplicities_equal=True, budget=fit.require_finished_budget(ROOT, "preflight")))
    source.immutable(ROOT / "plan.json", plan)
    source.immutable(OUTPUT / "plan.json", plan)


def train(plan, device):
    for seed in plan["seeds"]:
        if all(fit.retain_completed_fit(ROOT, plan, seed, pure, "predictor") for pure in (False, True)):
            continue
        with fit.fitting_budget(ROOT, f"preparation-{seed}", 300, "cpu") as budget:
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
            arm = "pure" if pure else "hybrid"
            with fit.fitting_budget(ROOT, f"predictor-{arm}-{seed}", plan["predictor_seconds_per_arm_seed"], device) as budget:
                sampled = schedule(starts, seed, pure, plan["start_update"], plan["updates"]["predictor"])
                path = fit.model_path(ROOT, seed, pure, "predictor")
                if not path.exists():
                    before = torch.load(plan["source_checkpoint_paths"][f"{seed}-{pure}"], map_location="cpu", weights_only=False)
                    fit.atomic_torch(path, initial_checkpoint(before, plan))
                    del before
                torch.manual_seed(seed)
                model = fit.new_predictor(plan, pure, device)

                def loss(update):
                    absolute = plan["start_update"] + update
                    if absolute not in sampled:
                        return None
                    pair = fit.pair_for_update(pure, absolute)
                    batch = source.source.trajectory_batch(shards, carriers, sampled[absolute], pair.delta, symbolic=not pure)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    return (source.source.full_continuous_loss(model, batch["z"], batch["available"], batch["action"], pair)
                            if pure else source.source.full_hybrid_loss(model, batch, pair))

                result = fit.fit_updates(path, model, plan_identity=plan["identity"],
                    updates=plan["updates"]["predictor"], lr=plan["learning_rate"], make_loss=loss, budget=budget, device=device)
                if (result["updates_applied"], result["updates_skipped"]) != (1785, 15):
                    raise ValueError("boundary-focus completed exposure differs")
            del model, result, sampled
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        if sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()) > plan["new_artifact_bytes"]:
            raise fit.FitBudgetExceeded("boundary-focus artifacts exceed 1 GiB")


def summarize(plan, rows):
    summaries = source.summarize(plan, [{**r, "lower_rate": r["boundary_focus"]} for r in rows])
    for summary in summaries:
        summary["anchor_metrics"]["boundary_focus"] = summary["anchor_metrics"].pop("lower_rate")
        local = [r for r in rows if r["seed"] == summary["seed"] and r["pure"] == summary["pure"]]
        points = {t: [next(p for p in r["lower_rate"][summary["anchor_policy"]] if p["elapsed"] == t)
                      for r in local] for t in (750, fit.ENDPOINT)}
        means = {str(t): sum(p["recursive_mse"] for p in values) / len(values) for t, values in points.items()}
        retained = all(p["available"] for values in points.values() for p in values) and all(
            summary["anchor_metrics"]["boundary_focus"][str(t)] <= plan["retention_ratio"] * means[str(t)] for t in points)
        summary["anchor_metrics"]["lower_rate"] = means
        summary.update(lower_rate_anchor_accuracy_retained=retained,
                       qualification_supported=summary["qualification_supported"] and retained)
    return summaries


def diagnose(plan):
    references = {(r["seed"], r["pure"], r["member_identity"]): r for r in plan["reference_rows"]}
    rows = []
    with fit.fitting_budget(ROOT, "diagnostic", 600, "cpu") as budget:
        for seed in plan["seeds"]:
            for pure in (False, True):
                fit.require_finished_budget(ROOT, f"predictor-{'pure' if pure else 'hybrid'}-{seed}")
                saved = torch.load(fit.model_path(ROOT, seed, pure, "predictor"), map_location="cpu", weights_only=False)
                if not saved["complete"] or saved["plan_identity"] != plan["identity"]:
                    raise ValueError("diagnosis requires the complete boundary-focus predictor")
                model = fit.new_predictor(plan, pure, "cpu").eval().requires_grad_(False)
                model.load_state_dict(saved["model"])
                for entry in plan["evaluation"]:
                    budget.check()
                    shard = fit.load_shard(Path(plan["data_root"]), entry, plan["data_binding"], fitting=True)
                    carrier = fit.load_carrier(Path(plan["data_root"]), seed, entry, plan["data_binding"])
                    rows.append({**references[seed, pure, entry["member_identity"]],
                                 "boundary_focus": source.source.source.drift_curves(model, shard, carrier)})
                    fit.log(f"boundary-focus diagnostic {len(rows)}/30")
    summaries = summarize(plan, rows)
    source.immutable(OUTPUT / "report.json", dict(plan=plan, rows=rows, summaries=summaries,
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
        prepare(plan)
    elif args.run:
        if fit.files.read(ROOT / "plan.json") != plan or fit.files.read(OUTPUT / "plan.json") != plan:
            raise ValueError("frozen boundary-focus plan differs")
        if str(args.device).startswith("cuda"):
            torch.cuda.init()
        train(plan, args.device)
        if not (OUTPUT / "report.json").exists():
            diagnose(plan)
    else:
        print("No-write: six paired 1800-position boundary-only continuations; prepare and archive before running.")


if __name__ == "__main__":
    main()
