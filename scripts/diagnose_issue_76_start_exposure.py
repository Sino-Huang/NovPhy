"""Count exact existing predictor exposure to shot-start training states."""
from collections import Counter, defaultdict
from pathlib import Path
import time

from scripts import run_issue_76_matched_dynamics as run

OUTPUT = run.fit.files.ROOT / "data/issue-76-start-exposure-diagnostic/report.json"


def main():
    began = time.monotonic()
    plan = run.fit.files.read(run.ROOT / "plan.json")
    starts = []
    for entry in plan["training"]:
        if entry["usable"]:
            shard = run.fit.load_shard(run.parent.ROOT, entry, plan["parent_plan"], fitting=True)
            starts.append({s["start"] for s in shard["segment_ranges"]})
        else:
            starts.append(set())
    lengths = [e["frames"] if e["usable"] else 0 for e in plan["training"]]
    rows = []
    for seed in plan["seeds"]:
        schedule = run.sampled_schedule(lengths, seed, False, 6000)
        for pure in (False, True):
            totals = Counter()
            boundaries, first = defaultdict(Counter), defaultdict(Counter)
            for update, samples in schedule.items():
                pair = run.fit.policy_name(run.fit.pair_for_update(pure, update))
                for entry, start in samples.tolist():
                    totals[pair] += 1
                    if start in starts[entry]:
                        boundaries[pair][entry] += 1
                    if start == 0:
                        first[pair][entry] += 1
            for pair in sorted(totals):
                rows.append(dict(seed=seed, pure=pure, pair=pair, samples=totals[pair],
                    shot_start_samples=sum(boundaries[pair].values()),
                    first_shot_start_samples=sum(first[pair].values()),
                    lineages_with_shot_start_exposure=len(boundaries[pair]),
                    lineages_with_first_shot_start_exposure=len(first[pair]),
                    per_lineage=[dict(member_identity=e["member_identity"], usable=e["usable"],
                                     shot_start_samples=boundaries[pair][i], first_shot_start_samples=first[pair][i])
                                 for i, e in enumerate(plan["training"])]))
        if time.monotonic() - began > 180:
            raise RuntimeError("start-exposure diagnostic exceeded 180-second CPU ceiling")
    report = dict(source_plan_identity=plan["identity"], rows=rows,
                  usable_training_lineages=sum(bool(s) for s in starts),
                  training_shot_starts=sum(len(s) for s in starts),
                  active_seconds=time.monotonic() - began, source_text=Path(__file__).read_text(),
                  optimization_performed=False, fresh_access=False,
                  interpretation="Exact original and mixed schedules have identical per-pair start multiplicities")
    run.parent.previous.immutable(OUTPUT, report)
    print({k: v for k, v in report.items() if k not in ("rows", "source_text")}, flush=True)
    for row in rows:
        print({k: v for k, v in row.items() if k != "per_lineage"}, flush=True)


if __name__ == "__main__":
    main()
