# Issue 69 corrected h15 experiment

This experiment uses the completed #67 checkpoints and #68 v2 release. No new
training, Unity capture, expert demonstrations, or final benchmark access is
needed. The original #63 result and checkpoints remain immutable references.

## Run in order

Activate the `novphy` conda environment and source `env.sh` first. Commands resume
completed immutable scoring cells. An interrupted cell repeats on the same
frozen states. Every model has a separate worker process; legacy checkpoint
metadata is released at worker exit. CPU threads are bounded to one per worker;
CUDA performs inference when requested. There are no new hashes or corpus scans.

```bash
python -u -m scripts.run_issue_69_corrected_experiment --dry-run

python -u -m scripts.run_issue_69_corrected_experiment \
  --prepare 2>&1 | tee -a data/issue-69-prepare.log

python -u -m scripts.run_issue_69_corrected_experiment \
  --score-calibration --device cuda 2>&1 | tee -a data/issue-69-calibration.log

python -u -m scripts.run_issue_69_corrected_experiment \
  --freeze-decision 2>&1 | tee -a data/issue-69-freeze.log

python -u -m scripts.run_issue_69_corrected_experiment \
  --score-model-selection --device cuda 2>&1 | tee -a data/issue-69-model-selection.log

python -u -m scripts.run_issue_69_corrected_experiment \
  --publish 2>&1 | tee -a data/issue-69-publish.log

python -u -m scripts.run_issue_69_corrected_experiment \
  --validate 2>&1 | tee -a data/issue-69-validate.log
```

`--prepare` binds the existing sources and freezes thresholds without loading
model-selection bundles. `--freeze-decision` chooses exactly one system using
calibration only. Model-selection scoring requires that immutable freeze and
records access before loading its bundles. It cannot add new calibration cells
after access. Do not change checkpoints, seeds, thresholds, states, or outputs
after this step to seek a different result.

`--smoke-test --device cuda` is an optional read-only real-checkpoint test after
prepare: all 14 scoring cells on the first calibration state. It writes no score
artifacts and never loads model selection. `--dry-run` needs neither data nor
checkpoints, writes no artifacts, and exercises every system plus selection and
decision logic on synthetic data. Neither substitutes for the full experiment.

## Frozen comparisons and decisions

The 14 scoring cells are the original three #63 seeds, nine #67 single models
(teacher-forced, u2, u4), and two recursive ensemble means (u2, u4). Reusing the
member candidate scores supplies 16 comparison systems, including the ensemble
mean and disagreement-penalized version of each corrected family. Each system
uses all 200 states and all 12 actions per role. The deterministic no-model
diagnostic rotates the selected action ordinal across states, independently of
outcomes. The three seeded models are averaged within each state for baseline
contrasts; seeds are not additional independent observations.

The matched #67 comparisons use 8 million sequence examples per cell and the
same architecture, seeds, data, optimizer settings, and carrier. Unroll length
necessarily changes the number of model evaluations per sequence example; that
compute is reported. The #63 original reference used 4 million examples, so it
is explicitly an unmatched external comparison. No retraining is necessary.

Calibration selects minimum mean normalized ranking regret, then h15 AUC, then
system name, with coefficient ties resolved toward the smaller coefficient.
Coefficients are 0, 0.25, 0.5, 1, and 2. Recursive ensemble accuracy means feeding
back the average predicted carrier at every step. Ensemble ranking means
averaging the independently rolled-out member endpoint costs, then adding the
coefficient times their population standard deviation. These are distinct,
explicit definitions from the deterministic seed-ensemble design, not full PETS.

For each state, regret is `(selected - minimum) / (maximum - minimum)` using the
full candidate inventory. All-tied states receive zero regret. A model prediction
failure receives regret one even on a tied state and vetoes advancement. The
retained #68 replay failure keeps its frozen cost of 1e9, participates in the
denominator, and is never replaced or filtered. This normalization emphasizes
within-state ranking and is not a pig-removal or gameplay-success estimand.

Advancement requires all seven one-sided simultaneous bootstrap contrasts:

- Relative h15 AUC improvement above 20% against both original and matched
  teacher-forced seed averages.
- Normalized regret improvement above 0.05 against both reference averages and
  the no-model action prior.
- No increase in the local squared carrier-bound proxy relative to the matched
  baseline, and no regret increase relative to the calibration-selected single
  member (the latter is an identity contrast for a selected single model).

Intervals use 20,000 paired independent-state bootstrap draws, fixed seed
20260906, and one-sided Bonferroni alpha `0.05 / 7`. These are approximate
percentile-bootstrap intervals, not a finite-sample distribution-free guarantee.
Only the calibration-selected system can advance; model-selection results never
select a replacement configuration.

Additional frozen gates require zero h15/local/ranking prediction failures;
absolute mean carrier-bound excess at most 0.01 in every state; every state's
recursive step MSE at most 1; each state's late-half/early-half MSE ratio at most
4 (early denominator floor 1e-8); and selected-action disagreement exceeding its
calibration 95th percentile in at most 10% of states. H1 recursive behavior is
diagnostic only. Carrier-bound diagnostics are representation-validity proxies,
not proof of physically correct engine behavior.

Planning must take at most 30 seconds and 540 model evaluations per 12-candidate
state, counting all ensemble members. These are prospectively declared #69
eligibility ceilings for the #64 non-final pilot; #64 has not yet frozen its
final gameplay-matrix budget. Per-cell training and inference compute and peak
RSS are retained. Times exclude checkpoint loading; measured score timings
include synchronized endpoint-cost reads.

## Outputs and completion

Private plan, access receipt, full per-state/per-candidate scores, per-step h1/h15
curves, local errors, physical proxies, failures, and compute live under
`.local-artifacts/issue-69-corrected-experiment-v1/`. Compact JSON and a Markdown
comparison table are published under `data/runtime_evidence/issue-69/`.
Validation recomputes selection, paired bounds, all gates, and the published
JSON/table exactly from the complete saved scores, and checks frozen inventory
and role disjointness. It does not repeat neural inference or hash checkpoints.

Close #69 only after full scoring, publication, and validation succeed. A
`not_supported_by_this_experiment` result still completes #69 but creates no
#64 authorization. A `supported` result binds one exact checkpoint list,
coefficient, candidate/cost definition, and budget; it permits considering the
#64 non-final pilot, not claiming gameplay success or opening final evaluation.
