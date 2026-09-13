"""Matched parser diagnostic: balance only pig supervision using training counts."""
import argparse
from collections import defaultdict

import torch
from torch.nn import functional as F

from scripts import run_issue_76_mixed_batches as mixed
from scripts.audit_issue_76_pig_exposure import OUTPUT as AUDIT


fit, previous = mixed.fit, mixed.previous
ROOT = fit.files.ROOT / ".local-artifacts/issue-76-balanced-pig-v1"
OUTPUT = fit.files.ROOT / "data/issue-76-balanced-pig"
SOURCES = ("scripts/run_issue_76_balanced_pig.py", "docs/issue-76-balanced-pig-protocol.md")


def balanced_loss(model, batch, weights):
    output = model(batch["images"])
    original = fit.perception_loss(lambda _: output, batch)
    target = batch["presence"][:, 10]
    logits = output["presence_logits"][:, 10]
    weight = torch.where(target.bool(), weights["positive"], weights["negative"])
    pig_loss = 4 / batch["presence"].shape[1] * F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pig_loss = pig_loss + (logits.sigmoid() - target).square()
    return original + ((weight - 1) * pig_loss).mean()


def make_plan():
    parent = mixed.make_plan()
    control = fit.files.read(mixed.OUTPUT / "report.json")
    if control["plan"] != parent:
        raise ValueError("mixed control source changed")
    audit = fit.files.read(AUDIT)
    if audit["parent_plan_identity"] != parent["identity"]:
        raise ValueError("exposure audit is not for the matched control")
    return dict(schema="issue_76_balanced_pig_plan_v1", identity=ROOT.name,
        parent=parent, controls=control["results"], control_summaries=control["summaries"],
        exposure_audit=audit, seconds_per_seed=900, new_artifact_bytes=2**30,
        criteria=dict(minimum_specificity=.5, minimum_sensitivity=.95, maximum_collateral_ratio=1.1),
        source_text={s: (fit.files.ROOT / s).read_text() for s in SOURCES},
        fresh_access=False, model_deployable=False, advancement_authorized=False)


def run(plan):
    parent, native = plan["parent"], fit.load_plan()
    usable = sum(e["usable"] for e in parent["training"])
    for summary in plan["exposure_audit"]["summaries"]:
        seed = summary["seed"]
        target = ROOT / f"seed-{seed}.json"
        if target.exists():
            fit.require_finished_budget(ROOT, f"balanced-{seed}")
            continue
        torch.manual_seed(seed)
        with fit.fitting_budget(ROOT, f"balanced-{seed}", plan["seconds_per_seed"], "cuda") as budget:
            model = fit.NativeVisualParser().cuda()
            buffer, buffered_epoch = None, None

            def loss(update):
                nonlocal buffer, buffered_epoch
                epoch, offset = divmod(update, len(parent["training"]))
                if offset >= usable:
                    return None
                if epoch != buffered_epoch:
                    buffer = mixed.epoch_batches(parent, native, seed, epoch, budget)
                    buffered_epoch = epoch
                    fit.log(f"balanced pig seed={seed} epoch={epoch + 1}/6 buffer ready")
                return balanced_loss(model, {k: v[offset].cuda() for k, v in buffer.items()}, summary["balanced_weights"])

            fit.fit_updates(ROOT / "checkpoints" / f"visual-balanced-{seed}.pt", model,
                plan_identity=plan["identity"], updates=parent["updates"], lr=parent["learning_rate"],
                make_loss=loss, budget=budget, device="cuda")
            model.eval().requires_grad_(False)
            controls = next(c["rows"] for c in plan["controls"] if c["seed"] == seed)
            rows = []
            for ordinal, (entry, control) in enumerate(zip(parent["evaluation"], controls), 1):
                budget.check()
                if entry["member_identity"] != control["member_identity"]:
                    raise ValueError("paired evaluation membership differs")
                metrics = dict(control["metrics"])
                if entry["usable"]:
                    shard = fit.load_shard(fit.ROOT, entry, native)
                    metrics["balanced"] = previous.measurements(model, shard["tensors"],
                        parent["evaluation_rows"][entry["member_identity"]], "cuda")
                rows.append({**control, "metrics": metrics})
                if ordinal % 25 == 0:
                    budget.save()
                    fit.log(f"balanced pig seed={seed} evaluation={ordinal}/{len(controls)}")
            budget.check()
            if sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()) > plan["new_artifact_bytes"]:
                budget.value["stopped"] = True
                budget.save()
                raise fit.FitBudgetExceeded("balanced-pig artifacts exceeded 1 GiB")
        previous.immutable(target, dict(plan_identity=plan["identity"], seed=seed, rows=rows))
        del model, buffer
        torch.cuda.empty_cache()


def summarize(plan):
    summaries, results = [], []
    for seed in plan["parent"]["seeds"]:
        result = fit.files.read(ROOT / f"seed-{seed}.json")
        if (result["seed"] != seed or result["plan_identity"] != plan["identity"]
                or [r["member_identity"] for r in result["rows"]] != [e["member_identity"] for e in plan["parent"]["evaluation"]]):
            raise ValueError("balanced result source/membership differs")
        groups = defaultdict(list)
        for row in result["rows"]:
            groups[row["exposure_role"], row["family"]].append(row)
        cells = []
        for (role, family), rows in sorted(groups.items()):
            available = [r for r in rows if r["usable"]]
            means = {arm: {k: sum(r["metrics"][arm][k] for r in available) / len(available)
                          for k in previous.METRICS} for arm in ("mixed", "balanced")}
            counts = {arm: {k: sum(r["metrics"][arm][k] for r in available)
                for k in ("pig_positive_frames", "pig_negative_frames", "pig_false_positive_frames", "pig_false_negative_frames")}
                for arm in means}
            cells.append(dict(role=role, family=family, assigned=len(rows), usable=len(available), means=means, counts=counts))
        roles = {}
        criteria = plan["criteria"]
        for role in ("training", "model_selection"):
            subset = [c for c in cells if c["role"] == role]
            roles[role] = {}
            for arm in ("mixed", "balanced"):
                counts = {k: sum(c["counts"][arm][k] for c in subset) for k in subset[0]["counts"][arm]}
                roles[role][arm] = dict(**counts,
                    specificity=1 - counts["pig_false_positive_frames"] / counts["pig_negative_frames"],
                    sensitivity=1 - counts["pig_false_negative_frames"] / counts["pig_positive_frames"],
                    family_equal={k: sum(c["means"][arm][k] for c in subset) / len(subset) for k in previous.METRICS})
        passed = all(roles[r]["balanced"]["specificity"] >= criteria["minimum_specificity"]
            and roles[r]["balanced"]["sensitivity"] >= criteria["minimum_sensitivity"] for r in roles)
        selection = roles["model_selection"]
        passed = passed and all(selection["balanced"]["family_equal"][k] <= criteria["maximum_collateral_ratio"] * selection["mixed"]["family_equal"][k]
                                   for k in ("block_count_mae", "task_presence_brier"))
        summaries.append(dict(seed=seed, cells=cells, roles=roles, balance_hypothesis_supported=passed,
            costs=fit.require_finished_budget(ROOT, f"balanced-{seed}")))
        results.append(result)
    return dict(schema="issue_76_balanced_pig_report_v1", plan=plan, summaries=summaries, results=results,
        balance_hypothesis_supported=all(s["balance_hypothesis_supported"] for s in summaries),
        model_deployable=False, fresh_access=False, advancement_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "prepare", "run", "publish", "validate"):
        modes.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = make_plan()
    if args.dry_run:
        fit.log("no-write balanced-pig dry run; existing training exposures only, no fit/capture")
        return
    if args.prepare:
        previous.immutable(ROOT / "plan.json", plan)
        fit.log("balanced-pig source, weights, criteria and budget frozen; no fitting started")
        return
    if fit.files.read(ROOT / "plan.json") != plan:
        raise ValueError("balanced-pig source or recipe changed")
    if args.run:
        run(plan)
    report = summarize(plan)
    if args.run or args.publish:
        previous.immutable(ROOT / "report.json", report)
    if args.publish:
        previous.immutable(OUTPUT / "report.json", report)
    if args.validate and fit.files.read(OUTPUT / "report.json") != report:
        raise ValueError("balanced-pig publication differs")
    fit.log(f"balance hypothesis supported={report['balance_hypothesis_supported']}; not advancement")


if __name__ == "__main__":
    main()
