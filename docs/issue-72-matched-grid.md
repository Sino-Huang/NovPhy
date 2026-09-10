# #72: matched-grid development screen

The completed #71 supplies trained executable models, not a useful-adaptation
result. Its 24 training-role diagnostics favored fixed h15 over adaptation in
every case. Do not switch optimizers or start a large fresh gameplay experiment
on that evidence alone.

This implementation covers **phase 1's calibration comparison and screening**.
It does not implement or authorize phase 2's fresh gameplay cohort. A passing
screen leaves the ticket open for a prospective fresh-cohort protocol, numerical
power/precision and advancement rules, collection, and evaluation. A failing
screen publishes `readiness_or_precision_insufficient` without unnecessary new
collection. It is not a fresh confirmatory negative result or a general rejection
of hybrid modeling. Closing the ticket requires inspection of the completed
screen and the disposition's scope, not merely successful software execution.

## Frozen comparison

Reuse all 200 calibration states / 2,400 recorded candidates from the #68/#70
CNN endpoint artifacts. Do not open model-selection, fresh evaluation, or final
data. Preserve the original source evidence unchanged.

Every model arm sees the same single-image CNN carrier and the same original
12-action grid. All predict to fixed-step offset 225; shorter stable source
captures are treated as absorbing for the observed diagnostic target.

1. #71 teacher-forced predictor, fixed continuous h15.
2. #71 corrected predictor, fixed continuous h15.
3. The **same corrected checkpoint** with its trained adaptive controller.
4. A no-model prior selected among 12 fixed action ordinals and the expected
   uniform-random action policy by calibration mean regret. The selected prior
   is optimistic on this calibration data; it is not an independent test result.

The two new model recipes match data, seed, capacity, local supervision and
optimizer exposure; recursive training work is an explicitly differing factor.
Legacy #70 models are not substituted as falsely matched controls. Optional
fixed-h1/axis/factorized controls are not included in this initial screen, so it
cannot establish a uniquely joint/nonseparable mechanism advantage.

Scoring inputs are only carrier, interface action, selected pair/controller and
the vocabulary-defined task objective: `1000 * expected pigs + expected blocks`.
Realized candidate outcomes and future carriers never enter this API.

The ranking diagnostic uses **settled replay engine count cost**. It is not
claimed to be an engine count measurement at step 225. Recursive MSE and parsed
count MAE separately compare against the cached CNN carrier at that common
horizon. This distinction avoids conflating a bounded-lookahead prediction with
the eventual settled outcome or claiming the CNN target is engine truth.

## Prospective development screen

The following numeric rules were implemented before the full calibration run:

- At least 20 outcome-informative calibration states.
- Zero prediction-failure states across the model arms.
- Adaptive mean normalized count regret at least 0.02 below both corrected
  fixed h15 and the strongest declared baseline (teacher-forced h15, corrected
  h15, or selected no-model prior).
- The second-most-used mode and horizon must each account for at least 5% of
  adaptive prediction decisions. These are descriptive nondegeneracy screens,
  not proof that varying the pair is useful.
- Each arm's common perception plus complete 12-action planning must stay within
  30 seconds per state on the declared execution device.

No state or candidate is replaced or selected based on outcomes. Rank ties use
the original ordinal. Normalize regret over accepted realized candidate costs;
a failed selected replay costs regret 1, rather than changing the scale of all
successful candidates through a 1e9 sentinel. All-failed states also cost 1.
Any failed model prediction makes that arm's state regret 1. All candidates and
failures remain in the records. OOM is an infrastructure interruption, not an
ordinary prediction failure used for scientific comparison.

Report training and adaptivity contrasts separately, using 10,000 paired
state-bootstrap draws (seed 7201). Intervals are **descriptive calibration
intervals only**: prior selection and reuse of opened development data preclude
fresh confirmation. Training gains cannot rescue a failing adaptivity screen.

## Compute, reproducibility, and validation

Per-state records retain every action, predicted endpoint/cost, chosen pair,
effective horizon, controller/decoder/transition call count, linear MAC count,
and measured planning time. Shared agent PNG read/CNN/carrier construction is
measured and charged once per arm. Startup and manifest I/O are excluded; full
gameplay cost and matched-FLOP claims are not made. Failed prediction traces may
be incomplete and are explicitly marked; their time is retained and the gate
fails rather than pretending their compute was zero.

The initial smoke found a 0.00043118 probability difference between B=1
deployment parsing and historical B=3 endpoint batching on CUDA. The initial
1e-3 tolerance was insufficient: the operator's run stopped at state 4 with
a 0.00382608 difference. Copies of the same initial image in a B=3 batch reproduce
that cached carrier exactly. This is a batch-dependent numerical difference,
not a different image, changed checkpoint, or capture failure.

The repaired boundary keeps the same fresh B=1 carrier for every model arm.
A separate B=3 audit uses three copies of **only the current image** to validate
the historical cache with tight 1e-5 tolerance. No future image is inspected for
this check. All 200 anchors passed; maximum audit/cache difference was
1.7881393432617188e-7. The additional offline validation time is reported
separately, not disguised as deployment planning work. Weights, deployment
inputs, old caches and scientific thresholds were not changed.

`plan.json` binds source inventories, model/controller identities, the #71
carrier contract, numeric screening rules, Git revision, and exact relevant
source text. No new hashes or image-corpus integrity passes. Local uncommitted
source is disclosed, not advertised as an archived release.

`--validate` checks every state/candidate/source binding, cost against stored
predicted endpoint, exact horizon and charged pair traces, and recomputes the
entire statistical publication. It does **not** rerun all neural predictions or
re-measure timings; the bounded real smoke exercises that inference path.

## Commands

After initializing `novphy` and sourcing `env.sh`:

```bash
python -u -m scripts.run_issue_72_matched_grid --dry-run

python -u -m scripts.run_issue_72_matched_grid \
  --smoke-test --device cuda

python -u -m scripts.run_issue_72_matched_grid \
  --run-development --device cuda \
  2>&1 | tee -a data/issue-72-development.log

python -u -m scripts.run_issue_72_matched_grid \
  --validate --device cuda \
  2>&1 | tee -a data/issue-72-validate.log
```

`--run-development` performs preparation, scoring and publication. Separate
`--prepare`, `--score-development` and `--publish` stages are also available.
Resume with the same command; completed states are validated and reused. A
partially scored state is recomputed, without recollecting or replacing it.

Output: `.local-artifacts/issue-72-matched-grid-v1/`, including `plan.json`,
`states/state-*.json`, and `result.json`. `smoke.json` is explicitly diagnostic
and is never promoted into production scores.

No Unity/display or retraining is needed. The one-state real CUDA smoke measured
about 0.20 s teacher-forced, 0.13 s corrected-fixed, 0.99 s adaptive, plus 0.17 s
shared perception. Allow roughly **5–15 minutes** for the full run, with I/O and
varying adaptive path lengths adding uncertainty. Logs identify each state, arm,
candidate, completion, elapsed time and ETA. Memory is bounded to the small
loaded models plus one state's source/score records.

On completion:

- `stopped_at_development_screen`: inspect/validate the readiness-insufficient
  result. No fresh gameplay was performed, and #64/#65 remain blocked.
- `fresh_protocol_required`: do not close #72 or start #64. Phase 2 still needs
  implementation and a prospective disjoint sample/power/budget/advancement
  freeze before access. A development pass is never a supported disposition.

Do not automatically retrain, change the screening thresholds after outcomes,
or launch successive corrective experiments to obtain a favorable result.

## Verification and handoff status

128 focused tests passed (13 #72 tests plus #71/predictor/carrier/CNN/#70
regressions). The no-write dry run passed against real source plans. The public
CUDA smoke exercised all 36 model/candidate combinations on the first real
calibration state, with no prediction failures; all scored endpoints and pair
traces validated. The initial anchor tolerance test failed on the actual
batch-rounding difference before the numerical-boundary fix, then passed, and
the original real-RGB smoke was rerun successfully. `git diff --check` passed.

Only diagnostic `smoke.json` exists under the new output root. The full
200-state run and its production decision/publication are left to the operator.
#72 remains open. The implementation and prerequisite #70/#71 changes remain
local/uncommitted; no commit or push was performed. Source snapshots are not an
archived release. No previous frozen source or experiment result was modified.

## Resume after the state-4 anchor-validation failure

The validation-only correction is recorded in
`anchor-validation-repair.json`. It binds the original frozen script and its
replacement exactly; `plan.json` and the three completed state records are left
untouched. Other code changes still fail source validation. The report references
this repair and discloses separate audit timing for newly scored states.

The agent has prepared this receipt in the shared workspace. Rerun the same
`--run-development --device cuda` command, then `--validate --device cuda`.
There is no need to delete data, use a new output directory, or retrain anything.
To reproduce receipt preparation on another copy of the same interrupted run:

```bash
python -u -m scripts.run_issue_72_matched_grid \
  --repair-anchor-validation --device cuda
```

This operation refuses changes to the frozen experimental settings/membership
or any other source component. Regression tests cover the actual B=1/B=3
mismatch, rejection of a genuinely different cache, and exact-source repair
binding. The focused suite now passes 130 tests.
