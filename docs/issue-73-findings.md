# #73 findings: useful actions are missed, and controller coverage is pre-launch

## Completed evidence

The read-only diagnostic ran in 49.4 seconds with peak process RSS about
1,399 MiB. It used all 200 existing calibration states / 2,400 replay candidates,
96 raw calibration traces, a frozen 12-lineage training sample, and four-policy
inference on eight calibration states. No new rollout, model training, fresh
holdout or final access was performed. Main publication validation passed.

The initial uniform training sample (indices 5, 255, ..., 2755) aliases the
alternating generator order and contains only type010101. It is retained, not
rewritten. A separately frozen, metadata-selected supplement uses indices
10, 260, ..., 2760 for type010102 and audits its 32 shots. That supplement's
aggregate validation also passed. Combined training coverage is 24 lineages,
44 first-shot-window instances and 132 later-window instances; these are NOT
44 or 132 independent lineages.

## 1. There is missed headroom within the current grid

| Selection | States with selected replay pig removal, out of 200 |
| --- | ---: |
| Best actually observed grid candidate | 21 |
| Frozen fixed-action prior | 9 |
| Teacher-forced model | 8 |
| Adaptive hybrid | 6 |
| Corrected fixed-h15 | 6 |
| Uniform random expectation | 1.83 |

The adaptive method misses 15 of the 21 available pig-removal opportunities.
This is evidence of selection loss within the tested grid. The best-of-grid
choice is a retrospective diagnostic, not an executable policy or multi-shot
oracle. These are reused single-shot replay outcomes, not new gameplay trials.

There are 81 count-informative states and 119 all-count-tied states. Among all
200 states, at least one candidate has pig contact in 64, block contact in 126,
support change in 120, pig displacement in 50, and block displacement in 102.
42 states have none of the declared progress signals or count discrimination.
Support changes/movement are not automatically collapse; absence of single-shot
pig removal does not establish that a big pig cannot be solved over several shots.

Remaining-count costs are reported for every candidate. Direct block-destruction
events were audited only in the bounded raw sample (two events across 96
captures); health/damage remained unavailable rather than inferred from score
or appearance. Do not generalize those sample counts to all 2,400 captures.

## 2. The controller's sampled training windows precede the shot

All 96 audited calibration captures launch at offset 150 fixed steps, about
3 seconds of recorded simulation time. The controller's source windows are
60 steps long. All 12 first windows in the original training sample and all
32 first windows in the second-family supplement are entirely pre-launch.

Their available macro labels are all steady-state positives and zero
structure-unstable positives: 2,596 positive steady labels in total. The later
windows already stored in the same shards contain both motion and unstable
examples (135 positive unstable labels across the sampled later windows).

This follows the actual #71 controller loader: it chooses only the first window
of each shot for its DP labels and aggregation. The PREDICTOR was trained on
first and later windows; do not incorrectly say the whole dynamics model saw
only static data. The controller's utility also targets duration-weighted carrier
MSE plus compute, not settled pig/block cost.

This establishes a coverage mismatch and a plausible intervention. It does not
prove that expanding coverage will improve gameplay, or that it is the only
limitation. Do not convert nominal 600 ms action hold time into simulated time
without an explicit execution-clock contract.

## 3. Timing, recursive error and action ranking are different measurements

At the current offset-225 prediction target there are only 75 simulated steps
after launch in the 96 audited calibration captures. Two of those captures still
have different active pig/block counts at 225 versus final settling. Keep the
bounded prediction endpoint and eventual settled task outcome distinct.

On eight prospectively selected calibration states, fixed h1/h5/h15 and adaptive
use the same corrected HYBRID-trained checkpoint and identical candidates:

| Policy | Mean endpoint carrier MSE | Mean ranking regret |
| --- | ---: | ---: |
| Fixed h1 | 1542.409 | 0.125 |
| Fixed h5 | 0.1405 | 0.125 |
| Fixed h15 | 0.01822 | 0.125 |
| Adaptive | 0.5996 | 0.125 |

Short h1 transitions accumulate substantial recursive error despite small local
errors. Yet this small sample does not distinguish policies by ranking regret.
Neither lower local MSE nor lower recursive MSE alone establishes better shots.
These are diagnostics, not newly selected winning configurations or a genuine
pure-continuous training baseline.

## 4. Visual review

Open `data/issue-73-review/index.html` in the local workspace.
It contains 15 source-linked existing WebMs, with event times, candidate IDs,
frame timing and explicit outcome-based diagnostic-selection labels. All 15
pass the available frame-count / recorded-dt / 50-FPS timing checks. Existing
video files are linked rather than copied. No video is represented as a new run.

Start with state 0037:

- Candidate 11: launch at 3.00 s; bird/pig collision at 4.42 s; level clear at
  6.46 s. The inspected pre-collision frame shows the bird reaching the pig.
- Candidate 12 (adaptive selection): launch at 3.00 s; ground collision at
  5.10 s. The inspected frame shows its path passing below the pig.

Both are in the high-y action stratum; that name alone does not guarantee an arc
that reaches the pig. Inspect actual flight/body volume and engine contacts,
not merely the displayed line. Lines are visible in the images, but structured
prediction-overlay ground truth is not available for a quantitative overlay
error claim. Other gallery cases show missed count reductions, contact without
removal and no recorded progress.

## Recommended next step

Continue #74's genuinely independent pure-continuous dynamics baseline and
matched-system contract. Record the controller timing/coverage defect explicitly.
Under #75, test ONE controller-only data-coverage intervention: use existing first
and later windows, including launch/post-launch contexts, for DP labels and the
same one aggregation round. Retain the original controller as the control; keep
the predictor, task objective, runtime input and action grid fixed. Apply
equivalent coverage to the pure-continuous controller for fair comparison.

Do not initialize action scoring from a candidate-specific observed post-drag
or post-launch frame: that would leak action-dependent future information.
Do not add MPPI/MCTS/gradient search merely because current results are negative.

Planning allowance for that ONE controller pilot: no new captures/images,
reuse existing shards, under roughly 1 GiB additional label/checkpoint storage,
allow about 2 GiB CPU RAM and 1 GiB GPU RAM pending a measured smoke, and at most
one GPU-hour before reassessing feasibility. These are planning estimates, not
measured production costs; #74 baseline fitting is separate. Freeze numeric
ranking/compute criteria and a stop rule before running the pilot. A negative
outcome must remain a negative outcome, not trigger another automatic search.

## Reproduction / artifacts

```bash
python -u -m scripts.run_issue_73_headroom_diagnostic --dry-run
python -u -m scripts.run_issue_73_headroom_diagnostic --smoke-test --device cuda
python -u -m scripts.run_issue_73_headroom_diagnostic --run --device cuda \
  2>&1 | tee -a data/issue-73-diagnostic.log
python -u -m scripts.run_issue_73_headroom_diagnostic --validate --device cuda \
  2>&1 | tee -a data/issue-73-validate.log
python -u -m scripts.run_issue_73_family_supplement --device cuda \
  2>&1 | tee -a data/issue-73-family-supplement.log
python -u -m scripts.run_issue_73_family_supplement --validate
```

Initialize `novphy` and source `env.sh` as usual. These commands have already
completed here; no rerun is necessary to proceed. The main runner resumes
completed trace/controller shards and recomputes cheap headroom/source checks.

Evidence root: `.local-artifacts/issue-73-headroom-v1/`, including the original
plan/result, per-state headroom, timing, training and controller records, and the
separate `family-supplement/plan.json` and `result.json`. Validation recomputes
headroom from source scores/outcomes and all 96 timing summaries from raw data,
then checks aggregate publication/gallery references. It does not rerun every
cached neural diagnostic; the supplement separately checks its frozen source,
family membership and aggregates.

140 focused tests pass. No old #70–#72 source/results were changed. New code and
the prior implementation remain local/uncommitted; snapshots identify them but
are not an archived Git release. No hashes, model training, new collection,
fresh/final access or #64 authorization were introduced by this diagnosis.
