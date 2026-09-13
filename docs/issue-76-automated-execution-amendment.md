# #76 automated execution amendment, 2026-09-13

The operator explicitly authorized fully automated work toward a genuine #76
advancement pass, including changes to workflow and resource budgets. This
supersedes the operator-only long-run handoff and the old active-time ceilings;
it does not authorize inventing a positive result, weakening an advancement
criterion, selecting favorable fresh outcomes, or discarding previous evidence.

## Execution changes before research fitting

The completed development collection retains all 350 assignments, including
the three display-start failures. Its final active time is 40,399.406984 seconds.
The operator cleanly paused CPU preparation after 1,343.024253 active seconds;
there are no new research model checkpoints. The old shared 12-hour allowance
left only 2,800.593016 seconds for preparation, against an observed multi-hour
requirement. This is an execution feasibility problem, not a model comparison.

Keep the frozen native-refit-v2 numerical experiment unchanged: three paired
seeds, all data roles, 22 slots, 352 carrier values, physical endpoints, losses,
optimizer update counts, model sizes, and failure/coverage rules. Extend each
preparation/common/predictor/controller-and-diagnostic job's active-time ceiling
to 24 hours. Preparation now has its own allowance. These are operational
ceilings, not extra optimizer updates or claims of matched compute. Retain all
previous active time and engineering charges; both predictor arms and every
seed get identical corresponding limits. Report actual time and compute.
Memory (12 GiB CPU, 8 GiB allocated CUDA) and working data (256 GiB) limits remain.
Further amendments, if needed, must also be explicit rather than resetting a
counter or concealing a stopped job.

`scripts/run_issue_76_automated_refit.py` is a separately source-bound execution
entry point. It leaves the old fitting modules, plan, collection and raw data
unchanged. Its amendment receipt preserves all pre-amendment budget records,
records the new limits and sources, and marks each changed budget. The entry
point supplies these declared time limits to the original budget context; all
training and prediction functions remain those in the frozen base plan. The
old command intentionally cannot silently adopt changed allowances.

The collection is finished and immutable. Its saved attempt-byte total plus
the size of its non-attempt files is the collection size. Add the current refit
directory size for the working-data check. This gives the same logical byte
count as the old whole-corpus traversal without rescanning captured frames at
every shard. A fixture checks exact equivalence. No new image hashes, capture
replays, label changes or whole-corpus integrity sweep are introduced by this
execution amendment. Existing per-record scientific validation is unchanged.

## Scientific progression

The native refit is a development prerequisite, not the fresh experiment or a
live gameplay policy. After fitting, assess perception, equal-time dynamics,
actual action ranking, legal live planning/re-observation, strongest independent
continuous/fixed/no-model comparators, useful mode use, validity and complete
compute. Development changes must retain their provenance and comparable arm
exposure. Preserve the earlier #72/#74/#75 conclusions.

Before once-only fresh evaluation, archive the exact eligible assets and freeze
numerical practical margins, paired clustered uncertainty and multiplicity,
sample-size/precision, system identities, legal search and failure rules. No
fresh outcome may tune the system, select a subset or reset the test. Only a
validated `supported_for_issue_64_non_final_pilot` result authorizes #64; neither
successful preparation, lower prediction error nor this amendment passes it.
Negative or infeasible results remain possible and must be reported honestly.

## Commands

Initialize `novphy` and source `env.sh`, then use the new entry point:

```bash
python -m scripts.run_issue_76_automated_refit --prepare-amendment
python -m scripts.run_issue_76_automated_refit --dry-run
python -u -m scripts.run_issue_76_automated_refit --run-refit --device cuda
```

Individual `--prepare-data`, `--fit-common`, `--fit-predictors`,
`--fit-controllers`, `--diagnose` and `--validate` modes use the same amendment.
The automation owns execution; do not start another copy concurrently. Clean
pauses retain the original optimizer/shard resume behavior. A failed or unclean
job still requires an audit, not an implicit retry.
