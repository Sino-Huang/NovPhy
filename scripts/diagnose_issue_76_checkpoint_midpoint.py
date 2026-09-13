"""Probe one fixed adjacent-checkpoint midpoint without fitting or saving models."""
import argparse
from pathlib import Path

import torch

from scripts import run_issue_76_endpoint_anchor as run

ROOT = run.fit.files.ROOT / ".local-artifacts/issue-76-checkpoint-midpoint-v1"
OUTPUT = run.fit.files.ROOT / "data/issue-76-checkpoint-midpoint/report.json"


def midpoint(left, right):
    assert left.keys() == right.keys()
    return {key: (left[key] + right[key]) / 2 for key in left}


def error_comparison(left, right, target):
    a, b = left - target, right - target
    return dict(local_anchor_mse=float(a.square().mean()), endpoint_anchor_mse=float(b.square().mean()),
                cross_error_mean=float((a * b).mean()), prediction_midpoint_mse=float(((a + b) / 2).square().mean()))


@torch.no_grad()
def compare_predictions(left, right, shard, carrier):
    segment = shard["segment_ranges"][0]
    start, stop = segment["start"], segment["stop"]
    steps = shard["tensors"]["fixed_steps"].tolist()
    lookup = {steps[i] - steps[start]: i for i in range(start, stop)}
    action = segment["action"][None]
    result = {}
    for pair in left.pairs:
        a = b = carrier[start:start + 1]
        points = []
        for elapsed in range(pair.delta, run.fit.ENDPOINT + 1, pair.delta):
            a = left.carrier(a, action, pair)
            b = right.carrier(b, action, pair)
            if elapsed not in (750, run.fit.ENDPOINT):
                continue
            if elapsed not in lookup:
                points.append(dict(elapsed=elapsed, available=False))
                continue
            target = carrier[lookup[elapsed]:lookup[elapsed] + 1]
            points.append(dict(elapsed=elapsed, available=True, **error_comparison(a, b, target)))
        result[run.fit.policy_name(pair)] = points
    return result


def summarize(plan, rows):
    summaries = run.summarize(plan, [{**r, "endpoint_anchor": r["midpoint"]} for r in rows])
    for summary in summaries:
        summary["anchor_metrics"]["midpoint"] = summary["anchor_metrics"].pop("endpoint_anchor")
        local = [r for r in rows if r["seed"] == summary["seed"] and r["pure"] == summary["pure"]]
        points = {t: [next(p for p in r["endpoint_anchor"][summary["anchor_policy"]] if p["elapsed"] == t)
                      for r in local] for t in (750, run.fit.ENDPOINT)}
        means = {str(t): sum(p["recursive_mse"] for p in values) / len(values) for t, values in points.items()}
        retained = all(p["available"] for values in points.values() for p in values) and all(
            summary["anchor_metrics"]["midpoint"][str(t)] <= plan["retention_ratio"] * means[str(t)] for t in points)
        summary["anchor_metrics"]["endpoint_anchor"] = means
        summary.update(endpoint_anchor_accuracy_retained=retained,
                       qualification_supported=summary["qualification_supported"] and retained)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    if not parser.parse_args().run:
        print("No-write: one 50/50 midpoint, all six arms/seeds and 30 records; no optimizer or checkpoint writes.")
        return
    torch.set_num_threads(1)
    report = run.fit.files.read(run.OUTPUT / "report.json")
    plan = report["plan"]
    rows = []
    with run.fit.fitting_budget(ROOT, "midpoint", 180, "cpu") as budget:
        for seed in plan["seeds"]:
            for pure in (False, True):
                saved = [torch.load(run.fit.model_path(root, seed, pure, "predictor"), map_location="cpu", weights_only=False)
                         for root in (run.source.ROOT, run.ROOT)]
                models = [run.fit.new_predictor(plan, pure, "cpu").eval().requires_grad_(False) for _ in range(3)]
                for checkpoint, model, identity in zip(saved, models, (plan["source_plan_identity"], plan["identity"])):
                    assert checkpoint["complete"] and checkpoint["failure"] is None and checkpoint["plan_identity"] == identity
                    model.load_state_dict(checkpoint["model"])
                models[2].load_state_dict(midpoint(saved[0]["model"], saved[1]["model"]))
                for entry in plan["evaluation"]:
                    budget.check()
                    shard = run.fit.load_shard(Path(plan["data_root"]), entry, plan["data_binding"], fitting=True)
                    carrier = run.fit.load_carrier(Path(plan["data_root"]), seed, entry, plan["data_binding"])
                    reference = next(r for r in report["rows"] if r["seed"] == seed and r["pure"] == pure
                                     and r["member_identity"] == entry["member_identity"])
                    comparison = compare_predictions(models[0], models[1], shard, carrier)
                    for policy, points in comparison.items():
                        for point in points:
                            if point["available"]:
                                for stage in ("local_anchor", "endpoint_anchor"):
                                    original = next(p for p in reference[stage][policy] if p["elapsed"] == point["elapsed"])
                                    assert point[stage + "_mse"] == original["recursive_mse"], "parent replay differs"
                    rows.append({**reference, "midpoint": run.source.source.source.source.drift_curves(models[2], shard, carrier),
                                 "error_comparison": comparison})
                    run.fit.log(f"checkpoint midpoint {len(rows)}/30")
                for checkpoint, model in zip(saved, models):
                    assert all(torch.equal(value, checkpoint["model"][key]) for key, value in model.state_dict().items())
        budget.check()
    summaries = summarize(plan, rows)
    run.immutable(OUTPUT, dict(source_plan_identity=plan["identity"], parent_plan_identity=plan["source_plan_identity"],
        weights=[.5, .5], evaluation=plan["evaluation"], rows=rows, summaries=summaries,
        midpoint_qualification_supported=all(s["qualification_supported"] for s in summaries),
        optimization_performed=False, checkpoint_writes=False, fresh_access=False, advancement_passed=False,
        budget=run.fit.require_finished_budget(ROOT, "midpoint"),
        replay_budget=run.fit.files.read(ROOT.parent / "issue-76-trajectory-retention-v1/budgets/replay.json"),
        source_text={p: (run.fit.files.ROOT / p).read_text() for p in (
            "scripts/diagnose_issue_76_checkpoint_midpoint.py", "docs/issue-76-checkpoint-midpoint-protocol.md")},
        interpretation="One fixed parameter midpoint; output midpoint is post-hoc diagnostic, not deployed ensemble recursion"))


if __name__ == "__main__":
    main()
