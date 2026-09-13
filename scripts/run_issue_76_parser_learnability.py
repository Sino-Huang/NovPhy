"""Bounded five-image parser learnability check; never a deployment model."""
import argparse

import torch

from scripts import run_issue_76_native_refit as fit
from scripts.run_issue_76_order_probe import immutable, measurements


ROOT = fit.files.ROOT / ".local-artifacts/issue-76-parser-learnability-v1"
OUTPUT = fit.files.ROOT / "data/issue-76-parser-learnability"
SOURCES = ("scripts/run_issue_76_parser_learnability.py",
           "scripts/run_issue_76_order_probe.py", "docs/issue-76-parser-learnability-protocol.md")


def make_plan():
    parent = fit.load_plan()
    selected = {}
    for entry in fit.data_index(fit.ROOT, parent)["entries"]:
        if entry["exposure_role"] == "training" and entry["usable"]:
            selected.setdefault(entry["family"], entry)
    return dict(schema="issue_76_parser_learnability_plan_v1", identity=ROOT.name,
        parent_plan_identity=parent["identity"], parent_source_text=parent["source_text"],
        entries=[selected[k] for k in sorted(selected)], frame_index=0, seed=fit.SEEDS[0],
        updates=1000, seconds=180, learning_rate=.001, new_artifact_bytes=256 * 2**20,
        block_mae_max=.1, task_presence_brier_max=.01,
        source_text={s: (fit.files.ROOT / s).read_text() for s in SOURCES},
        generalization_tested=False, model_deployable=False, fresh_access=False)


def succeeded(metrics, plan):
    return (metrics["block_count_mae"] <= plan["block_mae_max"]
            and metrics["task_presence_brier"] <= plan["task_presence_brier_max"])


@torch.no_grad()
def measure(model, batch):
    result = measurements(model, batch, list(range(len(batch["images"]))), "cuda")
    output = model(batch["images"])
    result.update(perception_loss=float(fit.perception_loss(model, batch)),
        predicted_block_counts=output["presence_logits"].sigmoid()[:, 3:10].sum(1).cpu().tolist(),
        true_block_counts=batch["presence"][:, 3:10].sum(1).cpu().tolist())
    return result


def run(plan):
    target = ROOT / "result.json"
    if target.exists():
        fit.require_finished_budget(ROOT, "learnability")
        return
    parent = fit.load_plan()
    torch.manual_seed(plan["seed"])
    with fit.fitting_budget(ROOT, "learnability", plan["seconds"], "cuda") as budget:
        samples = [fit.load_shard(fit.ROOT, e, parent, fitting=True)["tensors"] for e in plan["entries"]]
        batch = {k: torch.stack([s[k][plan["frame_index"]] for s in samples]).cuda()
                 for k in ("images", "presence", "centers")}
        del samples
        model = fit.NativeVisualParser().cuda()
        initial = measure(model, batch)
        fit.fit_updates(ROOT / "checkpoint.pt", model, plan_identity=plan["identity"],
            updates=plan["updates"], lr=plan["learning_rate"],
            make_loss=lambda update: fit.perception_loss(model, batch), budget=budget, device="cuda")
        final = measure(model.eval(), batch)
        budget.check()
        if sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()) > plan["new_artifact_bytes"]:
            budget.value["stopped"] = True
            budget.save()
            raise fit.FitBudgetExceeded("learnability probe exceeded 256 MiB")
    immutable(target, dict(plan_identity=plan["identity"], initial=initial, final=final))


def report(plan):
    result = fit.files.read(ROOT / "result.json")
    checkpoint = torch.load(ROOT / "checkpoint.pt", map_location="cpu", weights_only=False)
    if (result["plan_identity"] != plan["identity"] or checkpoint["plan_identity"] != plan["identity"]
            or not checkpoint["complete"] or checkpoint["updates_completed"] != plan["updates"]):
        raise ValueError("learnability result or completed checkpoint binding differs")
    return dict(schema="issue_76_parser_learnability_report_v1", plan=plan, result=result,
        costs=fit.require_finished_budget(ROOT, "learnability"),
        minimal_batch_fit_succeeded=succeeded(result["final"], plan),
        generalization_tested=False, model_deployable=False, advancement_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "prepare", "run", "publish", "validate"):
        modes.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = make_plan()
    if args.dry_run:
        fit.log(f"no-write learnability dry run: {[e['member_identity'] for e in plan['entries']]}; 1000 updates, no captures")
        return
    if args.prepare:
        immutable(ROOT / "plan.json", plan)
        fit.log("five-image learnability plan frozen; no fitting started")
        return
    if fit.files.read(ROOT / "plan.json") != plan:
        raise ValueError("learnability source or recipe changed")
    if args.run:
        run(plan)
    value = report(plan)
    if args.run or args.publish:
        immutable(ROOT / "report.json", value)
    if args.publish:
        immutable(OUTPUT / "report.json", value)
    if args.validate and fit.files.read(OUTPUT / "report.json") != value:
        raise ValueError("learnability publication differs")
    fit.log(f"minimal batch fit succeeded={value['minimal_batch_fit_succeeded']}; not a deployment or advancement result")


if __name__ == "__main__":
    main()
