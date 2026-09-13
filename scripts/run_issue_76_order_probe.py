"""One prospective, exposure-matched test of family-interleaved parser fitting."""
import argparse
from collections import defaultdict
from pathlib import Path

import torch

from scripts import run_issue_76_native_refit as fit


ROOT = fit.files.ROOT / ".local-artifacts/issue-76-parser-order-v1"
OUTPUT = fit.files.ROOT / "data/issue-76-parser-order"
SOURCES = ("scripts/run_issue_76_order_probe.py", "docs/issue-76-parser-order-protocol.md")
METRICS = ("block_count_mae", "pig_count_mae", "task_presence_brier", "task_center_mae")


def interleaved_order(entries):
    groups = defaultdict(list)
    for index, entry in enumerate(entries):
        groups[entry["family"]].append(index)
    if len({len(v) for v in groups.values()}) != 1:
        raise ValueError("this probe requires the assigned balanced family inventory")
    return [index for row in zip(*(groups[k] for k in sorted(groups))) for index in row]


def original_update(update, order):
    epoch, offset = divmod(update, len(order))
    return epoch * len(order) + order[offset]


def training_sample(training, order, seed, update):
    source_update = original_update(update, order)
    entry = training[source_update % len(training)]
    rows = torch.randint(entry["frames"], (32,), generator=fit.generator(seed, source_update)) if entry["usable"] else None
    return entry, rows


def evaluation_rows(frames):
    count = min(32, frames)
    return [0] if count == 1 else [i * (frames - 1) // (count - 1) for i in range(count)]


def make_plan():
    parent = fit.load_plan()
    index = fit.data_index(fit.ROOT, parent)
    training = [e for e in index["entries"] if e["exposure_role"] == "training"]
    evaluation = [e for e in index["entries"] if e["exposure_role"] in ("training", "model_selection")]
    baselines = {str(seed): fit.require_finished_budget(fit.ROOT, f"common-{seed}")
                 for seed in parent["seeds"]}
    return dict(schema="issue_76_parser_order_plan_v1", identity=ROOT.name,
        parent_plan=parent, training=training, evaluation=evaluation,
        baseline_common_budgets=baselines, order=interleaved_order(training),
        evaluation_rows={e["member_identity"]: evaluation_rows(e["frames"]) for e in evaluation},
        updates=1500, batch_size=32, learning_rate=.001, seconds_per_seed=900,
        new_artifact_bytes=2**30, source_text={s: (fit.files.ROOT / s).read_text() for s in SOURCES},
        required_model_selection_mae_ratio=.5, required_training_probe_count_span=3.,
        fresh_access=False, history_dynamics_or_controller_fitting=False,
        common_checkpoint_deployable=False, advancement_authorized=False)


def immutable(path, value):
    if path.exists():
        if fit.files.read(path) != value:
            raise ValueError(f"different existing probe evidence: {path}")
    else:
        fit.files.write(path, value)


def check_disk(plan, budget):
    if sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()) > plan["new_artifact_bytes"]:
        budget.value["stopped"] = True
        budget.save()
        raise fit.FitBudgetExceeded("parser-order probe exceeded 1 GiB of new artifacts")


@torch.no_grad()
def measurements(model, tensors, rows, device):
    images = tensors["images"][rows].to(device)
    truth = tensors["presence"][rows].to(device)
    centers = tensors["centers"][rows].to(device)
    out = model(images)
    probability = out["presence_logits"].sigmoid()
    present = truth[:, 3:11].bool()
    pig_true = truth[:, 10].bool()
    pig_pred = probability[:, 10] >= .5
    return {
        "block_count_mae": float((probability[:, 3:10].sum(1) - truth[:, 3:10].sum(1)).abs().mean()),
        "pig_count_mae": float((probability[:, 10] - truth[:, 10]).abs().mean()),
        "task_presence_brier": float((probability[:, 3:11] - truth[:, 3:11]).square().mean()),
        "task_center_mae": float((out["centers"][:, 3:11][present] - centers[:, 3:11][present]).abs().mean()) if bool(present.any()) else None,
        "first_frame_block_count": float(probability[0, 3:10].sum()),
        "first_frame_true_block_count": float(truth[0, 3:10].sum()),
        "pig_positive_frames": int(pig_true.sum()), "pig_negative_frames": int((~pig_true).sum()),
        "pig_false_positive_frames": int((pig_pred & ~pig_true).sum()),
        "pig_false_negative_frames": int((~pig_pred & pig_true).sum())}


def train_and_score(plan):
    parent = plan["parent_plan"]
    for seed in parent["seeds"]:
        result_path = ROOT / f"seed-{seed}.json"
        if result_path.exists():
            fit.require_finished_budget(ROOT, f"probe-{seed}")
            fit.log(f"retain completed parser-order seed={seed}")
            continue
        torch.manual_seed(seed)
        with fit.fitting_budget(ROOT, f"probe-{seed}", plan["seconds_per_seed"], "cuda") as budget:
            model = fit.NativeVisualParser().cuda()

            def loss(update):
                entry, rows = training_sample(plan["training"], plan["order"], seed, update)
                if rows is None:
                    return None
                shard = fit.load_shard(fit.ROOT, entry, parent, fitting=True)
                return fit.perception_loss(model, {k: shard["tensors"][k][rows].cuda()
                    for k in ("images", "presence", "centers")})

            fit.fit_updates(ROOT / "checkpoints" / f"visual-interleaved-{seed}.pt", model,
                plan_identity=plan["identity"], updates=plan["updates"], lr=plan["learning_rate"],
                make_loss=loss, budget=budget, device="cuda")
            model.eval().requires_grad_(False)
            check_disk(plan, budget)
            baseline = fit.NativeVisualParser().cuda().eval().requires_grad_(False)
            checkpoint = torch.load(fit.common_path(fit.ROOT, seed), map_location="cuda", weights_only=False)
            if checkpoint["plan_identity"] != parent["identity"] or checkpoint["seed"] != seed or checkpoint["synthetic"]:
                raise ValueError("ordered baseline checkpoint binding differs")
            baseline.load_state_dict(checkpoint["parser"])
            records = []
            for ordinal, entry in enumerate(plan["evaluation"], 1):
                budget.check()
                row = {k: entry[k] for k in ("member_identity", "family", "exposure_role", "usable")}
                row["metrics"] = {}
                if entry["usable"]:
                    shard = fit.load_shard(fit.ROOT, entry, parent)
                    rows = plan["evaluation_rows"][entry["member_identity"]]
                    row["metrics"] = {name: measurements(m, shard["tensors"], rows, "cuda")
                        for name, m in (("ordered", baseline), ("interleaved", model))}
                records.append(row)
                if ordinal % 25 == 0:
                    budget.save()
                    fit.log(f"parser-order seed={seed} evaluation={ordinal}/{len(plan['evaluation'])} active={budget.value['active_seconds']:.1f}s")
            budget.check()
            check_disk(plan, budget)
        immutable(result_path, dict(plan_identity=plan["identity"], seed=seed, rows=records))
        del model, baseline, checkpoint
        torch.cuda.empty_cache()


def summarize(plan):
    summaries, results = [], []
    for seed in plan["parent_plan"]["seeds"]:
        result = fit.files.read(ROOT / f"seed-{seed}.json")
        if result["plan_identity"] != plan["identity"] or result["seed"] != seed:
            raise ValueError("parser-order result binding differs")
        if [r["member_identity"] for r in result["rows"]] != [e["member_identity"] for e in plan["evaluation"]]:
            raise ValueError("parser-order result omitted or reordered an assigned episode")
        cost = fit.require_finished_budget(ROOT, f"probe-{seed}")
        groups = defaultdict(list)
        probes = {}
        for row in result["rows"]:
            groups[row["exposure_role"], row["family"]].append(row)
            if row["exposure_role"] == "training" and row["usable"]:
                probes.setdefault(row["family"], row)
        cells = []
        for (role, family), rows in sorted(groups.items()):
            available = [r for r in rows if r["usable"]]
            means = {arm: {metric: sum(r["metrics"][arm][metric] for r in available
                if r["metrics"][arm][metric] is not None) / sum(r["metrics"][arm][metric] is not None for r in available)
                for metric in METRICS} for arm in ("ordered", "interleaved")}
            cells.append(dict(role=role, family=family, assigned=len(rows), usable=len(available), means=means))
        selection = [c for c in cells if c["role"] == "model_selection"]
        mae = {arm: sum(c["means"][arm]["block_count_mae"] for c in selection) / len(selection)
               for arm in ("ordered", "interleaved")}
        spans = {}
        for arm in ("ordered", "interleaved"):
            counts = [r["metrics"][arm]["first_frame_block_count"] for r in probes.values()]
            spans[arm] = max(counts) - min(counts)
        passed = (mae["interleaved"] <= plan["required_model_selection_mae_ratio"] * mae["ordered"]
                  and spans["interleaved"] >= plan["required_training_probe_count_span"])
        summaries.append(dict(seed=seed, costs=cost, cells=cells,
            family_equal_model_selection_block_mae=mae, training_probe_count_spans=spans,
            ordering_hypothesis_supported=passed))
        results.append(result)
    return dict(schema="issue_76_parser_order_report_v1", plan=plan, summaries=summaries,
        results=results, ordering_hypothesis_supported=all(s["ordering_hypothesis_supported"] for s in summaries),
        trained_gameplay_evaluated=False, fresh_access=False, advancement_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "prepare", "run", "publish", "validate"):
        modes.add_argument("--" + mode, action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = make_plan()
    if args.dry_run:
        fit.log("no-write parser-order dry run: three 1500-update fits, exact original samples, no captures")
        return
    if args.prepare:
        immutable(ROOT / "plan.json", plan)
        fit.log("parser-order plan frozen; no fitting started")
        return
    if fit.files.read(ROOT / "plan.json") != plan:
        raise ValueError("parser-order source, membership or recipe changed")
    if args.run:
        train_and_score(plan)
    value = summarize(plan)
    if args.run or args.publish:
        immutable(ROOT / "report.json", value)
    if args.publish:
        immutable(OUTPUT / "report.json", value)
    if args.validate and fit.files.read(OUTPUT / "report.json") != value:
        raise ValueError("published parser-order report differs")
    fit.log(f"parser-order complete: hypothesis_supported={value['ordering_hypothesis_supported']}; not an advancement result")


if __name__ == "__main__":
    main()
