"""Prospective exact-example regrouping of native predictor training."""
import argparse
import gc

import torch

from scripts import run_issue_76_repaired_representation as parent
from scripts.issue_76_matched_batches import indexed_batch, mixed_schedule, sampled_schedule

fit = parent.fit
ROOT = fit.files.ROOT / ".local-artifacts/issue-76-matched-dynamics-v1"
OUTPUT = fit.files.ROOT / "data/issue-76-matched-dynamics"
SOURCES = ("scripts/run_issue_76_matched_dynamics.py", "scripts/issue_76_matched_batches.py",
           "docs/issue-76-matched-dynamics-protocol.md", "scripts/issue_76_fit_budget.py",
           "scripts/run_issue_76_native_refit.py", "world_model/training/native_history_fit.py",
           "world_model/training/native_history_model.py")


def make_plan():
    source = fit.files.read(parent.ROOT / "plan.json")
    training = [e for e in fit.data_index(parent.ROOT, source)["entries"]
                if e["exposure_role"] == "training"]
    selected = {}
    for entry in training:
        selected.setdefault(entry["family"], entry)
    return dict(identity=ROOT.name, parent_plan=source, source_root=str(parent.ROOT),
                training=training, evaluation=list(selected.values()),
                seeds=source["seeds"], capacity=source["capacity"],
                updates={"predictor": 6000}, learning_rate=.0003, batch_size=32,
                predictor_seconds_per_arm_seed=1800, preparation_seconds_per_seed=300,
                diagnostic_seconds=600, new_artifact_bytes=2**30,
                source_text={s: (fit.files.ROOT / s).read_text() for s in SOURCES},
                reused_common_costs={str(seed): fit.require_finished_budget(parent.ROOT, f"common-{seed}")
                                     for seed in source["seeds"]},
                hybrid_parent_mse_ratio=.8, hybrid_unchanged_wins_per_seed=4,
                pure_parent_mse_ratio=1.1, fresh_access=False, advancement_authorized=False)


def train(plan, device):
    source = plan["parent_plan"]
    lengths = [e["frames"] if e["usable"] else 0 for e in plan["training"]]
    for seed in plan["seeds"]:
        if all(fit.retain_completed_fit(ROOT, plan, seed, pure, "predictor") for pure in (False, True)):
            continue
        with fit.fitting_budget(ROOT, f"preparation-{seed}", plan["preparation_seconds_per_seed"], "cpu") as budget:
            shards, carriers = [], []
            for entry in plan["training"]:
                budget.check()
                shards.append(fit.load_shard(parent.ROOT, entry, source, fitting=True) if entry["usable"] else None)
                carriers.append(fit.load_carrier(parent.ROOT, seed, entry, source) if entry["usable"] else None)
        for pure in (False, True):
            if fit.retain_completed_fit(ROOT, plan, seed, pure, "predictor"):
                continue
            name = "pure" if pure else "hybrid"
            with fit.fitting_budget(ROOT, f"predictor-{name}-{seed}", plan["predictor_seconds_per_arm_seed"], device) as budget:
                original = sampled_schedule(lengths, seed, pure, plan["updates"]["predictor"], plan["batch_size"])
                schedule = mixed_schedule(original, seed, pure)
                torch.manual_seed(seed)
                model = fit.new_predictor(plan, pure, device)

                def loss(update):
                    if update not in schedule:
                        return None
                    pair = fit.pair_for_update(pure, update)
                    batch = indexed_batch(shards, carriers, schedule[update], pair.delta, symbolic=not pure)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    if pure:
                        return fit.continuous_loss(model, batch["z"], batch["available"], batch["action"], pair)
                    return fit.hybrid_loss(model, batch, pair)

                result = fit.fit_updates(fit.model_path(ROOT, seed, pure, "predictor"), model,
                    plan_identity=plan["identity"], updates=plan["updates"]["predictor"],
                    lr=plan["learning_rate"], make_loss=loss, budget=budget, device=device)
                if (result["updates_applied"], result["updates_skipped"]) != (len(schedule), 6000 - len(schedule)):
                    raise ValueError("matched predictor did not consume the exact scheduled updates")
            del model, result, original, schedule
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        del shards, carriers
        gc.collect()
        if sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()) > plan["new_artifact_bytes"]:
            raise fit.FitBudgetExceeded("matched dynamics artifacts exceed 1 GiB")


@torch.no_grad()
def recursive_probe(model, shard, carrier):
    segment = shard["segment_ranges"][0]
    start, stop = segment["start"], segment["stop"]
    steps = shard["tensors"]["fixed_steps"].tolist()
    ends = [i for i in range(start, stop) if steps[i] == steps[start] + fit.ENDPOINT]
    if len(ends) != 1:
        return dict(available=False)
    target = carrier[ends[0]]
    values = {}
    for pair in model.pairs:
        predicted, _ = fit.rollout(model, None, carrier[start], segment["action"], fixed_pair=pair)
        values[fit.policy_name(pair)] = float((predicted - target).square().mean())
    return dict(available=True, unchanged_carrier_mse=float((carrier[start] - target).square().mean()),
                policies=values)


def summarize(plan, rows):
    summaries = []
    for seed in plan["seeds"]:
        for pure in (False, True):
            local = [r for r in rows if r["seed"] == seed and r["pure"] == pure]
            policy = "fixed-750-continuous" if pure else "fixed-750-micro"
            available = [r for r in local if r["parent"]["available"] and r["mixed"]["available"]]
            if len(available) != len(plan["evaluation"]):
                summaries.append(dict(seed=seed, pure=pure, qualification_supported=False,
                                      reason="assigned endpoint unavailable"))
                continue
            means = {kind: sum(r[kind]["policies"][policy] for r in available) / len(available)
                     for kind in ("parent", "mixed")}
            wins = sum(r["mixed"]["policies"][policy] < r["mixed"]["unchanged_carrier_mse"] for r in available)
            ratio = plan["pure_parent_mse_ratio"] if pure else plan["hybrid_parent_mse_ratio"]
            summaries.append(dict(seed=seed, pure=pure, policy=policy, mean_carrier_mse=means,
                unchanged_wins=wins, qualification_supported=means["mixed"] <= ratio * means["parent"]
                and (pure or wins >= plan["hybrid_unchanged_wins_per_seed"])))
    return summaries


def diagnose(plan):
    if (OUTPUT / "report.json").exists():
        return
    rows = []
    with fit.fitting_budget(ROOT, "diagnostic", plan["diagnostic_seconds"], "cpu") as budget:
        for seed in plan["seeds"]:
            for pure in (False, True):
                fit.require_finished_budget(ROOT, f"predictor-{'pure' if pure else 'hybrid'}-{seed}")
                saved = torch.load(fit.model_path(ROOT, seed, pure, "predictor"), map_location="cpu", weights_only=False)
                if not saved["complete"] or saved["plan_identity"] != plan["identity"]:
                    raise ValueError("qualification requires completed source-bound predictor")
                model = fit.new_predictor(plan, pure, "cpu").eval().requires_grad_(False)
                model.load_state_dict(saved["model"])
                old = fit.load_predictor(parent.ROOT, plan["parent_plan"], seed, pure, "cpu")
                for entry in plan["evaluation"]:
                    budget.check()
                    shard = fit.load_shard(parent.ROOT, entry, plan["parent_plan"], fitting=True)
                    carrier = fit.load_carrier(parent.ROOT, seed, entry, plan["parent_plan"])
                    rows.append(dict(seed=seed, pure=pure, member_identity=entry["member_identity"],
                        family=entry["family"], parent=recursive_probe(old, shard, carrier),
                        mixed=recursive_probe(model, shard, carrier)))
                    fit.log(f"matched predictor qualification {len(rows)}/30")
    summaries = summarize(plan, rows)
    costs = {p.stem: fit.files.read(p) for p in sorted((ROOT / "budgets").glob("*.json"))}
    parent.previous.immutable(OUTPUT / "report.json", dict(plan=plan, rows=rows, summaries=summaries,
        budgets=costs, qualification_supported=all(r["qualification_supported"] for r in summaries),
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
        parent.previous.immutable(ROOT / "plan.json", plan)
        parent.previous.immutable(OUTPUT / "plan.json", plan)
        fit.log("matched dynamics plan frozen; no fitting performed")
        return
    if args.run:
        if fit.files.read(ROOT / "plan.json") != plan:
            raise ValueError("frozen matched dynamics plan differs")
        train(plan, args.device)
        diagnose(plan)
    else:
        fit.log("no-write plan check passed; use --prepare before --run")


if __name__ == "__main__":
    main()
