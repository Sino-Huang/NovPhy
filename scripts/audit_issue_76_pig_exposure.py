"""Count pig classes in the exact existing mixed-batch training exposure; no fit."""
import argparse
from pathlib import Path

import torch

from scripts import run_issue_76_mixed_batches as mixed


fit = mixed.fit
ROOT = fit.files.ROOT / ".local-artifacts/issue-76-pig-exposure-v1"
OUTPUT = fit.files.ROOT / "data/issue-76-pig-exposure/report.json"


def count_samples(presence, seed, updates):
    """Keep duplicated draws: class weights describe exposures, not unique frames."""
    positive = negative = 0
    for update in updates:
        rows = torch.randint(len(presence), (32,), generator=fit.generator(seed, update))
        labels = presence[rows, 10].bool()
        positive += int(labels.sum())
        negative += int((~labels).sum())
    return dict(positive=positive, negative=negative)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args()
    torch.set_num_threads(1)
    if OUTPUT.exists():
        raise ValueError("retain the completed exposure audit; do not repeat source reads")
    plan = mixed.make_plan()
    if fit.files.read(mixed.ROOT / "plan.json") != plan:
        raise ValueError("mixed source/recipe changed")
    parent = fit.load_plan()
    records = []
    with fit.fitting_budget(ROOT, "training-label-counts", 300, "cpu") as budget:
        for index, entry in enumerate(plan["training"]):
            budget.check()
            row = {k: entry[k] for k in ("member_identity", "exposure_role", "family", "usable")}
            row["counts"] = {}
            if entry["usable"]:
                shard = fit.load_shard(fit.ROOT, entry, parent, fitting=True)
                for seed in plan["seeds"]:
                    row["counts"][str(seed)] = count_samples(shard["tensors"]["presence"], seed,
                        range(index, plan["updates"], len(plan["training"])))
                del shard
            records.append(row)
        budget.check()
    summaries = []
    for seed in plan["seeds"]:
        counts = {k: sum(r["counts"][str(seed)][k] for r in records if r["usable"])
                  for k in ("positive", "negative")}
        total = sum(counts.values())
        assert total == 47616
        summaries.append(dict(seed=seed, exposures=total, **counts,
            balanced_weights={k: total / (2 * n) for k, n in counts.items()},
            negative_lineages=[r["member_identity"] for r in records
                               if r["usable"] and r["counts"][str(seed)]["negative"]]))
    value = dict(schema="issue_76_pig_exposure_audit_v1", parent_plan_identity=plan["identity"],
        source_text=Path(__file__).read_text(), rows=records, summaries=summaries,
        costs=fit.require_finished_budget(ROOT, "training-label-counts"),
        fitting_performed=False, fresh_access=False, advancement_authorized=False)
    mixed.previous.immutable(OUTPUT, value)
    for row in summaries:
        fit.log(str(row))


if __name__ == "__main__":
    main()
