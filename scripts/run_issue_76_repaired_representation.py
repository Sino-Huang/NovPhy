"""Paired downstream refit with the published mixed/balanced shared parser."""
import argparse
import gc
import gzip
from pathlib import Path

import torch

from scripts import run_issue_76_balanced_pig as parser_fit
from scripts.publish_issue_76_native_refit import compact


fit, previous = parser_fit.fit, parser_fit.previous
ROOT = fit.files.ROOT / ".local-artifacts/issue-76-repaired-representation-v1"
OUTPUT = fit.files.ROOT / "data/issue-76-repaired-representation"
SOURCES = ("scripts/run_issue_76_repaired_representation.py",
           "docs/issue-76-repaired-representation-protocol.md",
           "scripts/publish_issue_76_native_refit.py")
PHYSICAL_METRICS = ("presence_mse_to_engine", "pig_count_absolute_error", "block_count_absolute_error")
CONTRAST_POLICIES = ("adaptive", "fixed-750-continuous")


def make_plan():
    native = fit.load_plan()
    parser_plan = parser_fit.make_plan()
    report = fit.files.read(parser_fit.OUTPUT / "report.json")
    if report["plan"] != parser_plan or not report["balance_hypothesis_supported"]:
        raise ValueError("requires the published three-seed balanced parser result")
    original = fit.files.read(fit.files.ROOT / "data/issue-76-native-refit/report.json")
    if original["plan_identity"] != native["identity"]:
        raise ValueError("original diagnostic source differs")
    references = [{**{k: row[k] for k in ("seed", "pure", "member_identity", "role", "family", "available")},
        "policies": {p: {k: value for k, value in row["policies"][p].items() if k in (*PHYSICAL_METRICS, "failure")}
                     for p in CONTRAST_POLICIES} if row["available"] else {}} for row in original["rows"]]
    return {**native, "identity": ROOT.name, "parent_plan": native,
        "source_root": str(fit.ROOT), "parser_plan": parser_plan,
        "original_reference": references, "original_reference_summaries": original["summaries"],
        "parser_checkpoints": {str(seed): str(parser_fit.ROOT / "checkpoints" / f"visual-balanced-{seed}.pt")
                               for seed in native["seeds"]},
        "parser_reference_summaries": report["summaries"],
        "updates": {**native["updates"], "visual": 0},
        "source_text": {**native["source_text"], **{s: (fit.files.ROOT / s).read_text() for s in SOURCES}},
        "new_artifact_bytes": 8 * 2**30, "preparation_seconds": 300,
        "parser_reused_without_updates": True, "fresh_access": False,
        "advancement_authorized": False}


def metadata_shard(shard, plan):
    """Reuse exact prepared targets; omit RGB already owned by the parent shard."""
    return {**shard, "source_plan_identity": shard["plan_identity"],
        "plan_identity": plan["identity"],
        "tensors": {k: v for k, v in shard["tensors"].items() if k != "images"}}


def check_storage(plan):
    if sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()) > plan["new_artifact_bytes"]:
        raise fit.FitBudgetExceeded("repaired representation exceeded 8 GiB of new artifacts")


def prepare_data(plan):
    target = ROOT / "data-index.json"
    if target.exists():
        fit.require_finished_budget(ROOT, "data-preparation")
        return fit.data_index(ROOT, plan)
    source = Path(plan["source_root"])
    parent = fit.data_index(source, plan["parent_plan"])
    with fit.fitting_budget(ROOT, "data-preparation", plan["preparation_seconds"], "cpu") as budget:
        for ordinal, entry in enumerate(parent["entries"], 1):
            budget.check()
            if entry["usable"]:
                path = ROOT / entry["path"]
                if not path.exists():
                    shard = fit.load_shard(source, entry, plan["parent_plan"])
                    fit.atomic_torch(path, metadata_shard(shard, plan))
            if ordinal % 25 == 0:
                fit.log(f"repaired metadata={ordinal}/{len(parent['entries'])}")
        budget.check()
        check_storage(plan)
    value = {**parent, "plan_identity": plan["identity"], "source_plan_identity": parent["plan_identity"]}
    previous.immutable(target, value)
    return value


def load_parser(plan, seed, device):
    checkpoint = torch.load(plan["parser_checkpoints"][str(seed)], map_location=device, weights_only=False)
    if (checkpoint["plan_identity"] != plan["parser_plan"]["identity"] or not checkpoint["complete"]
            or checkpoint["updates_requested"] != 1500 or checkpoint["updates_applied"] != 1488
            or checkpoint["updates_skipped"] != 12):
        raise ValueError("shared parser is not the completed matched checkpoint")
    model = fit.NativeVisualParser().to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval().requires_grad_(False)


def train_common(plan, device):
    index = fit.data_index(ROOT, plan)
    training = [e for e in index["entries"] if e["exposure_role"] == "training"]
    for seed in plan["seeds"]:
        final = fit.common_path(ROOT, seed)
        if final.exists():
            value = torch.load(final, map_location="cpu", weights_only=False)
            if value["plan_identity"] != plan["identity"] or value["seed"] != seed:
                raise ValueError("common checkpoint source differs")
            fit.require_finished_budget(ROOT, f"common-{seed}")
            continue
        torch.manual_seed(seed)
        with fit.fitting_budget(ROOT, f"common-{seed}", plan["limits"]["common_seconds_per_seed"], device) as budget:
            parser = load_parser(plan, seed, device)
            for ordinal, entry in enumerate(index["entries"], 1):
                budget.check()
                if not entry["usable"]:
                    continue
                target = fit.visual_path(ROOT, seed, entry)
                if not target.exists():
                    shard = fit.load_shard(Path(plan["source_root"]), entry, plan["parent_plan"])
                    visual = fit.encode_visual(parser, shard["tensors"])
                    fit.atomic_torch(target, dict(plan_identity=plan["identity"], seed=seed,
                        member_identity=entry["member_identity"], visual=visual))
                    del shard
                if ordinal % 25 == 0:
                    budget.save()
                    fit.log(f"repaired common seed={seed} visual-cache={ordinal}/{len(index['entries'])}")
            common = fit.CommonHistoryFit().to(device)

            def history_loss(update):
                entry = training[update % len(training)]
                if not entry["usable"]:
                    return None
                shard = fit.load_shard(ROOT, entry, plan, fitting=True)
                cached = torch.load(fit.visual_path(ROOT, seed, entry), map_location="cpu", weights_only=False)
                return common.loss(cached["visual"].to(device), shard["tensors"]["timestamps"].to(device), shard["segment_ranges"])

            fit.fit_updates(ROOT / "checkpoints" / f"history-{seed}.pt", common,
                plan_identity=plan["identity"], updates=plan["updates"]["history"], lr=.001,
                make_loss=history_loss, budget=budget, device=device)
            common.eval().requires_grad_(False)
            for ordinal, entry in enumerate(index["entries"], 1):
                budget.check()
                if not entry["usable"]:
                    continue
                target = fit.carrier_path(ROOT, seed, entry)
                if not target.exists():
                    shard = fit.load_shard(ROOT, entry, plan)
                    cached = torch.load(fit.visual_path(ROOT, seed, entry), map_location="cpu", weights_only=False)
                    with torch.no_grad():
                        carrier = fit.encode_history(common.history, cached["visual"].to(device),
                            shard["tensors"]["timestamps"].to(device), shard["segment_ranges"]).cpu()
                    fit.atomic_torch(target, dict(plan_identity=plan["identity"], seed=seed,
                        member_identity=entry["member_identity"], carrier=carrier))
                if ordinal % 25 == 0:
                    budget.save()
                    fit.log(f"repaired common seed={seed} history-cache={ordinal}/{len(index['entries'])}")
            budget.check()
            check_storage(plan)
            fit.atomic_torch(final, dict(schema="issue_76_native_common_checkpoint_v1", plan_identity=plan["identity"],
                seed=seed, parser=parser.state_dict(), history=common.history.state_dict(),
                parser_source_checkpoint=plan["parser_checkpoints"][str(seed)],
                synthetic=plan["synthetic"], auxiliary_deployed=False))
        del parser, common
        gc.collect()
        torch.cuda.empty_cache()


def physical_contrasts(plan, rows):
    keyed = {(r["seed"], r["pure"], r["member_identity"]): r for r in rows}
    records = []
    for original in plan["original_reference"]:
        row = keyed[original["seed"], original["pure"], original["member_identity"]]
        if (row["role"], row["family"], row["available"]) != (original["role"], original["family"], original["available"]):
            raise ValueError("physical comparison membership or endpoint availability changed")
        changes = {}
        if row["available"]:
            for policy in CONTRAST_POLICIES:
                old, new = original["policies"][policy], row["policies"][policy]
                changes[policy] = dict(original_failure=old["failure"], repaired_failure=new["failure"],
                    deltas=None if old["failure"] or new["failure"] else {k: new[k] - old[k] for k in PHYSICAL_METRICS})
        records.append({**{k: original[k] for k in ("seed", "pure", "member_identity", "role", "family", "available")},
                        "repaired_minus_original": changes})
    return records


def publish(plan, validate=False):
    original = fit.publish_diagnostic(ROOT, plan, validate=True)
    value = compact(original)
    value.update(schema="issue_76_repaired_representation_report_v1", plan=plan,
        paired_physical_contrasts=physical_contrasts(plan, original["rows"]),
        complete_step_records="diagnostic-report.json.gz",
        source_preprocessing_cost=fit.require_finished_budget(Path(plan["source_root"]), "data-preparation"),
        parser_reuse_costs=[s["costs"] for s in plan["parser_reference_summaries"]],
        interpretation="Development refit with repaired shared perception, not advancement; carrier targets differ from the old refit.")
    raw = (ROOT / "diagnostic-report.json").read_bytes()
    target, archive = OUTPUT / "report.json", OUTPUT / "diagnostic-report.json.gz"
    if not validate:
        previous.immutable(target, value)
        if not archive.exists():
            archive.write_bytes(gzip.compress(raw, mtime=0))
    if fit.files.read(target) != value or gzip.decompress(archive.read_bytes()) != raw:
        raise ValueError("repaired representation publication or full archive differs")
    fit.log("repaired representation: all 600 assigned diagnostic records published and validated; no fresh access")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "prepare", "run", "publish", "validate"):
        modes.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = make_plan()
    if args.dry_run:
        fit.log("no-write paired refit: reuse three balanced parsers; 500 history / 6000 predictor / 2000 controller scheduled updates")
        return
    if args.prepare:
        previous.immutable(ROOT / "plan.json", plan)
        fit.log("repaired representation protocol frozen; no new data/weights yet")
        return
    if fit.files.read(ROOT / "plan.json") != plan:
        raise ValueError("repaired representation source/recipe changed")
    if args.run:
        prepare_data(plan)
        train_common(plan, "cuda")
        fit.train_predictors(ROOT, plan, "cuda")
        fit.train_controllers(ROOT, plan, "cuda")
        fit.diagnose(ROOT, plan, "cuda")
        check_storage(plan)
    publish(plan, validate=args.validate)


if __name__ == "__main__":
    main()
