# Run the #75 controller-coverage pilot

Protocol: [prospective decision](issue-75-controller-coverage.md).
The new script leaves all #74 predictors/controllers and their results untouched.
It fits only six new controllers with broader existing window coverage.

In the initialized `novphy` environment (`conda activate novphy`, then
`source env.sh`), run sequentially:

```bash
python -u -m scripts.run_issue_75_controller_coverage \
  --prepare \
  2>&1 | tee -a data/issue-75-prepare.log

python -u -m scripts.run_issue_75_controller_coverage \
  --train --device cuda \
  2>&1 | tee -a data/issue-75-training.log

python -u -m scripts.run_issue_75_controller_coverage \
  --score-development --device cuda \
  2>&1 | tee -a data/issue-75-development.log

python -u -m scripts.run_issue_75_controller_coverage \
  --publish --device cuda \
  2>&1 | tee -a data/issue-75-publish.log

python -u -m scripts.run_issue_75_controller_coverage \
  --validate --device cuda \
  2>&1 | tee -a data/issue-75-validate.log
```

`--prepare` records the source-bound settings and exits successfully; it does
not run training. Logs show each seed/arm, label-lineage progress, controller
updates and elapsed/ETA, followed by every candidate score and publication
validation progress. No display or new game collection is needed. No new
hashing or full-image integrity scans are performed.

Budget: at most3600 active-command seconds for labels/training and3600 for
scoring, persisted across resumes;3GiB peak CPU RSS,2GiB peak CUDA allocated,
6GiB new artifacts. Time/memory checks happen at saved work boundaries; disk
checks happen after training cells and every50 scored states. Validation time
is separate and must not be interpreted as training/inference cost.

After an ordinary interruption, repeat the same command. Do not run concurrent
writers or delete the budget counters. Completed labels/state scores are reused;
controller optimizer and sampler checkpoints resume every60 updates. The counter
cannot account exactly for work lost between a hard kill and the previous save.

If a resource limit stops training/scoring, **do not increase it or continue the
next expensive stage automatically**. Run `--publish` and `--validate` to retain
the explicit `budget_or_readiness_insufficient` report, then send the logs for
review. A partial set is never interpreted as a favorable result. A negative
scientific gate is `not_supported_by_this_pilot`; it is not a software failure.

Outputs: `.local-artifacts/issue-75-controller-coverage-v1/` contains `plan.json`,
`budget-training.json`, `budget-scoring.json`, `seed-*/{continuous,hybrid}/`
controller/label artifacts, `seed-*/states/`, and `result.json`.
Original fixed-policy traces are read from #74 rather than duplicated or
rerun; the two new controllers are scored on all200 calibration states and
all3 seeds. Validation checks saved trace costs, original-control regrets,
label/checkpoint bindings, coverage and the published disposition; it does not
remeasure all neural inference or historical wall times.

Bounded developer checks (already run before handoff; optional to repeat):

```bash
python -u -m scripts.run_issue_75_controller_coverage --dry-run
python -u -m scripts.run_issue_75_controller_coverage --smoke-test --device cuda
python -m unittest tests.test_issue_75_controller_coverage -q
```

Smoke uses real existing predictors, two-family training lineages5/10, all their
existing windows, both controller rounds, resume, and state4 candidate scoring.
Its temporary controller weights are removed; `smoke.json` is diagnostic only.
It does not constitute the pilot or authorize fresh evaluation.

The ticket remains open until the complete run or explicit resource-stop report
is inspected. Even a positive development pilot only proposes a candidate for
#76; it does not authorize #64/#65. Source is local/uncommitted/unpushed and large
artifacts are not an archived release. Preserve the source snapshots and #74
dependencies for the later archive handoff.

## Publication JSON repair

The initial publisher could return a NumPy boolean for
`checks.recursive_mse_not_worse`; Python JSON rejects it even though its displayed
type name is `bool`. The repair converts the mean used in that comparison to a
Python float, producing a normal Python boolean. Values, thresholds and the
decision are unchanged. The regression now exercises the final strict JSON write,
not only in-memory report equality.

For an existing frozen run, use this once before rerunning publication:

```bash
python -u -m scripts.run_issue_75_controller_coverage --repair-publication
```

It records `publication-json-repair.json` with the exact original and repaired
runner sources. The original plan, checkpoints, labels, scores and resource
counters are not rewritten. Other source/settings changes remain rejected.
Then run the existing `--publish` and `--validate` commands above. Do not retrain,
rescore or delete/refreeze the plan for this serialization-only repair.
