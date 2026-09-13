"""Exposure-matched parser probe: mix existing samples within each batch."""
import argparse
from collections import defaultdict

import torch

from scripts import run_issue_76_native_refit as fit
from scripts import run_issue_76_order_probe as previous


ROOT = fit.files.ROOT / ".local-artifacts/issue-76-parser-mixed-batches-v1"
OUTPUT = fit.files.ROOT / "data/issue-76-parser-mixed-batches"
SOURCES = ("scripts/run_issue_76_mixed_batches.py", "docs/issue-76-parser-mixed-batches-protocol.md")


def mix_batches(samples):
    """Transpose sample and episode axes together for every input/target field."""
    return {k: torch.stack([s[k] for s in samples]).transpose(0, 1).reshape(-1, 32, *samples[0][k].shape[1:])
            for k in samples[0]}


def make_plan():
    reference = fit.files.read(previous.OUTPUT / "report.json")
    prior = previous.make_plan()
    if reference["plan"] != prior:
        raise ValueError("published interleaved control differs from its frozen plan")
    return dict(schema="issue_76_mixed_batches_plan_v1", identity=ROOT.name,
        parent_plan_identity=prior["parent_plan"]["identity"],
        parent_source_text=prior["parent_plan"]["source_text"],
        prior_source_text=prior["source_text"], reference_report=str(previous.OUTPUT / "report.json"),
        controls=reference["results"], control_summaries=reference["summaries"],
        training=prior["training"], evaluation=prior["evaluation"],
        order=prior["order"], evaluation_rows=prior["evaluation_rows"],
        seeds=prior["parent_plan"]["seeds"], updates=1500, learning_rate=.001,
        seconds_per_seed=900, new_artifact_bytes=2**30,
        required_mae_ratio=.5, required_count_span=3., presence_must_not_worsen=True,
        source_text={s: (fit.files.ROOT / s).read_text() for s in SOURCES},
        model_deployable=False, fresh_access=False, advancement_authorized=False)


def epoch_batches(plan, parent, seed, epoch, budget):
    samples = []
    for offset in range(len(plan["training"])):
        budget.check()
        entry, rows = previous.training_sample(plan["training"], plan["order"], seed,
                                               epoch * len(plan["training"]) + offset)
        if rows is None:
            continue
        shard = fit.load_shard(fit.ROOT, entry, parent, fitting=True)
        samples.append({k: shard["tensors"][k][rows] for k in ("images", "presence", "centers")})
    return mix_batches(samples)


def run(plan):
    parent = fit.load_plan()
    usable = sum(e["usable"] for e in plan["training"])
    for seed in plan["seeds"]:
        target = ROOT / f"seed-{seed}.json"
        if target.exists():
            fit.require_finished_budget(ROOT, f"mixed-{seed}")
            continue
        torch.manual_seed(seed)
        with fit.fitting_budget(ROOT, f"mixed-{seed}", plan["seconds_per_seed"], "cuda") as budget:
            model = fit.NativeVisualParser().cuda()
            buffer, buffered_epoch = None, None

            def loss(update):
                nonlocal buffer, buffered_epoch
                epoch, offset = divmod(update, len(plan["training"]))
                if offset >= usable:
                    return None  # The two original unusable assignments, never zero-filled images.
                if epoch != buffered_epoch:
                    buffer = epoch_batches(plan, parent, seed, epoch, budget)
                    buffered_epoch = epoch
                    fit.log(f"mixed parser seed={seed} epoch={epoch + 1}/6 original-sample buffer ready")
                return fit.perception_loss(model, {k: v[offset].cuda() for k, v in buffer.items()})

            fit.fit_updates(ROOT / "checkpoints" / f"visual-mixed-{seed}.pt", model,
                plan_identity=plan["identity"], updates=plan["updates"], lr=plan["learning_rate"],
                make_loss=loss, budget=budget, device="cuda")
            model.eval().requires_grad_(False)
            controls = next(c["rows"] for c in plan["controls"] if c["seed"] == seed)
            records = []
            for ordinal, (entry, control) in enumerate(zip(plan["evaluation"], controls), 1):
                budget.check()
                if entry["member_identity"] != control["member_identity"]:
                    raise ValueError("mixed parser and paired control membership differ")
                metrics = dict(control["metrics"])
                if entry["usable"]:
                    shard = fit.load_shard(fit.ROOT, entry, parent)
                    metrics["mixed"] = previous.measurements(model, shard["tensors"],
                        plan["evaluation_rows"][entry["member_identity"]], "cuda")
                records.append({**control, "metrics": metrics})
                if ordinal % 25 == 0:
                    budget.save()
                    fit.log(f"mixed parser seed={seed} evaluation={ordinal}/{len(plan['evaluation'])} active={budget.value['active_seconds']:.1f}s")
            budget.check()
            if sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()) > plan["new_artifact_bytes"]:
                budget.value["stopped"] = True
                budget.save()
                raise fit.FitBudgetExceeded("mixed parser exceeded 1 GiB of new artifacts")
        previous.immutable(target, dict(plan_identity=plan["identity"], seed=seed, rows=records))
        del model, buffer
        torch.cuda.empty_cache()


def summarize(plan):
    summaries, results = [], []
    for seed in plan["seeds"]:
        result = fit.files.read(ROOT / f"seed-{seed}.json")
        if (result["plan_identity"] != plan["identity"] or result["seed"] != seed
                or [r["member_identity"] for r in result["rows"]] != [e["member_identity"] for e in plan["evaluation"]]):
            raise ValueError("mixed parser result source or assigned inventory differs")
        cost = fit.require_finished_budget(ROOT, f"mixed-{seed}")
        groups, probes = defaultdict(list), {}
        for row in result["rows"]:
            groups[row["exposure_role"], row["family"]].append(row)
            if row["exposure_role"] == "training" and row["usable"]:
                probes.setdefault(row["family"], row)
        cells = []
        for (role, family), rows in sorted(groups.items()):
            available = [r for r in rows if r["usable"]]
            means = {arm: {metric: sum(r["metrics"][arm][metric] for r in available) / len(available)
                     for metric in previous.METRICS} for arm in ("ordered", "interleaved", "mixed")}
            pig_errors = {arm: {k: sum(r["metrics"][arm][k] for r in available)
                for k in ("pig_positive_frames", "pig_negative_frames", "pig_false_positive_frames", "pig_false_negative_frames")}
                for arm in means}
            cells.append(dict(role=role, family=family, assigned=len(rows), usable=len(available),
                              means=means, pig_class_counts=pig_errors))
        selection = [c for c in cells if c["role"] == "model_selection"]
        means = {arm: {metric: sum(c["means"][arm][metric] for c in selection) / len(selection)
                 for metric in previous.METRICS} for arm in ("ordered", "interleaved", "mixed")}
        counts = [r["metrics"]["mixed"]["first_frame_block_count"] for r in probes.values()]
        span = max(counts) - min(counts)
        passed = (means["mixed"]["block_count_mae"] <= plan["required_mae_ratio"] * means["interleaved"]["block_count_mae"]
            and span >= plan["required_count_span"]
            and means["mixed"]["task_presence_brier"] <= means["interleaved"]["task_presence_brier"])
        summaries.append(dict(seed=seed, costs=cost, cells=cells, family_equal_model_selection=means,
                              training_probe_count_span=span, mixed_batch_hypothesis_supported=passed))
        results.append(result)
    return dict(schema="issue_76_mixed_batches_report_v1", plan=plan, summaries=summaries, results=results,
        mixed_batch_hypothesis_supported=all(s["mixed_batch_hypothesis_supported"] for s in summaries),
        model_deployable=False, trained_gameplay_evaluated=False, fresh_access=False, advancement_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "prepare", "run", "publish", "validate"):
        modes.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = make_plan()
    if args.dry_run:
        fit.log("no-write mixed-batch dry run: exact original frame/label multiset, three matched fits, no captures")
        return
    if args.prepare:
        previous.immutable(ROOT / "plan.json", plan)
        fit.log("mixed-batch protocol frozen; no fitting started")
        return
    if fit.files.read(ROOT / "plan.json") != plan:
        raise ValueError("mixed-batch source, controls or recipe changed")
    if args.run:
        run(plan)
    value = summarize(plan)
    if args.run or args.publish:
        previous.immutable(ROOT / "report.json", value)
    if args.publish:
        previous.immutable(OUTPUT / "report.json", value)
    if args.validate and fit.files.read(OUTPUT / "report.json") != value:
        raise ValueError("mixed-batch publication differs")
    fit.log(f"mixed-batch hypothesis supported={value['mixed_batch_hypothesis_supported']}; not an advancement result")


if __name__ == "__main__":
    main()
