# #75 completed: controller-coverage pilot not supported

Verified 2026-09-10. Disposition: `not_supported_by_this_pilot`.
This is a completed negative experiment under its frozen scope, not an unfinished
implementation or resource stop. No second correction or task expansion was run.

## Verification and evidence

The actual `--validate --device cuda` command completed with exit0 and exact
saved-evidence equality. It checked the source-repair binding, all completed
controllers/labels, symmetric window coverage, all200 states across three seeds,
original #74 control regrets, prediction costs/work traces, content interventions
and the frozen decision. It did not rerun every neural prediction or remeasure
historical wall times.

Separately replayed state4/candidate1 on the actual trained continuous and hybrid
controllers for all three seeds (six live CUDA predictions): traces matched
exactly, carriers matched at atol/rtol1e-6, and costs within1e-4. The publication
JSON repair and its15 passing regressions are documented in
[operator notes](issue-75-operator.md). Original plan/checkpoints/scores were not
replaced; the repair changed a NumPy scalar type, not numerical gates.

Artifact root: `.local-artifacts/issue-75-controller-coverage-v1/`:

- `plan.json`, `publication-json-repair.json`, `result.json`;
- `seed-*/{continuous,hybrid}/` controllers and two rounds of labels;
- `seed-*/states/` new candidate scores;
- `budget-training.json`, `budget-scoring.json`, diagnostic `smoke.json`.

Logs: `data/issue-75-{training,development,publish,validate}.log`.
Source and experiment protocol: [prospective plan](issue-75-controller-coverage.md).

## What completed

Six controllers: continuous and hybrid for seeds20260908/20260909/20260910,
two1800-update rounds each, with predictors unchanged. Each arm/seed/round used
1092 first windows plus3276 later windows across the same600 controller-training
lineages, supplying262080 label rows. Broader coverage was applied symmetrically;
optimizer exposure and the original60-step DP utility stayed fixed.

All200 calibration states × three seeds × two new policies ×12 actions completed:
14400 new candidate predictions. The600 state records are repeats of the SAME200
independent states, not600 independent lineages. There were zero prediction-failure
states in either new policy. All six fixed symbolic-content interventions passed.
No new game collection occurred; task outcomes come from existing single-shot
candidate replays, not a new multi-shot gameplay evaluation.

## Results

Mean normalized action-ranking regret across all three seeds; lower is better:

| Policy | Regret |
| --- | ---: |
| Frozen no-model ordinal09 prior | 0.304995 |
| Independent continuous h5 | 0.329997 |
| Same hybrid checkpoint, fixed continuous h15 | 0.335000 |
| Original hybrid adaptive | 0.349998 |
| Broader-coverage hybrid adaptive | 0.361663 |
| Original continuous adaptive | 0.366665 |
| Broader-coverage continuous adaptive | 0.395000 |

Hybrid action regret worsened in each seed:
0.329995→0.349995, 0.390000→0.394995, 0.330000→0.340000.
Mean225-step recursive carrier MSE increased from8.285743 to12.637019, although
individual seeds differed strongly. It is not a local one-step prediction score.

Covered-minus-original improvement convention (positive would favor coverage):
hybrid regret contrast -0.011665, descriptive paired-state95% interval
[-0.035000,0.011668]; continuous adaptive -0.028335, interval
[-0.045002,-0.013333]. Covered hybrid versus continuous h5 is -0.031667,
interval[-0.056668,-0.006667]. These reused-development intervals do not establish
fresh confirmation or seed-robust superiority; no favorable-seed selection.

Existing grid headroom:81 informative states;21 states have a pig-removing/
level-clearing candidate. Covered hybrid selected pig-removing shots in5/2/2
states across seeds, versus original hybrid7/1/6 and the fixed prior9 in each
seed. Do not count these seed repeats as independent gameplay trials.

Failed gates: improvement over original hybrid, improvement over strongest
comparator, improvement in at least two seeds, and no increase in recursive MSE.
Passed gates: complete inventory, informative-state count, zero prediction
failures, mode/horizon use, real symbolic-content effects and planning wall limit.
Useful variation is not established simply by mode/horizon usage.

Recorded active work: training/labels1818.1s, scoring1581.8s (~56.7 minutes total),
within separate3600s allowances. Recorded output size3,693,816,559 bytes; peak
CPU RSS1510.8MiB and CUDA allocation37.7MiB. Validation/smoke and work lost since
an interrupted unsaved checkpoint are outside those counters, as disclosed.

## Disposition and handoff

Close #75 with its validated negative disposition; do not automatically launch
another correction or relax the frozen gate. This pilot tested controller-window
coverage, NOT the broader NovPhy scenario/novelty concern. Population remains
normal single-force/multiple-forces; the shared CNN is semantically supervised.
No general impossibility claim about hybrid world models follows.

Next is amended #76 Phase A: bounded fixed-horizon/error-growth diagnostics and
metadata-first task/novelty compatibility and feasibility work. Fresh collection
requires its separate protocol and readiness decision. #64/#65 remain unauthorized;
earlier #15/#72/#74 evidence is unchanged.

No code changes, further fitting or collection were needed for this closure.
Implementation is still local/uncommitted/unpushed; `archived_release=false`.
This completion does not claim a pushed source or archived large-artifact release.
