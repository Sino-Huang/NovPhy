# #74: genuine continuous dynamics and matched hybrid controls

Implementation and operator handoff; full fits and development results are pending.
Dependencies #71/#72/#73 are closed. #74 stays open until the operator run is
validated. No new game collection, CNN fit, #75 intervention, or fresh/final
evaluation is included. Existing local work and historical evidence are preserved.

## What is being compared

Both arms share the accepted #70 CNN and its 236-value/18-slot carrier. The CNN
already has semantic/engine-derived supervision. “Pure continuous” describes the
new DYNAMICS model, not a symbol-free end-to-end pixel baseline. #15's original
central result remains unchanged; no uniquely joint/nonseparable mechanism is claimed.

The continuous model is independently initialized: horizon-only sinusoidal FiLM
conditioning, a continuous residual trunk, no mode embedding, symbolic readout,
adapter, symbolic loss or hybrid checkpoint initialization. Its controller has
only three trained horizon logits. The new hybrid uses #71's actual learned
selected-mode conditioning; its fixed continuous-h15 control shares that exact
new hybrid checkpoint, not a separately trained model. No old #70/#71 model is
silently substituted into the matched fits.

The frozen executable policies are:

- continuous fixed h1, h5 and h15;
- continuous learned horizon control;
- hybrid learned pair control;
- the same hybrid model fixed to continuous h15;
- no-model fixed candidate ordinal09, previously selected on #72 development.

Three paired seeds: 20260908, 20260909, 20260910. No favorable-seed selection.
Select ONE continuous policy by mean calibration regret across all three seeds
and all 200 states; exact ties follow the listed policy order. Any prediction
failure disqualifies a policy from selection and incurs state regret1. If none
is eligible, report insufficient readiness. Preserve every attempted result.

## Prospective matching contract

Reuse #71 shards, without copying/reparsing RGB. Same 2,400 dynamics-fitting
lineages; the other 600 training-role lineages supervise controllers. The frozen
CNN saw all 3,000 training lineages, so controller diagnostics are not independent
perception tests. Calibration outcomes never fit the predictors/controllers.

Each model gets 9,000 AdamW updates, batch64, lr0.0001, weight decay0.0001,
gradient clipping1; last-update checkpoint only. The identical three-lineage
groups and sampler seeds yield identical minibatches and horizon exposure.
Hybrid cycles nine pairs; continuous uses the same horizon on every update,
including updates where the hybrid uses micro/macro. Thus continuous receives
3,000 updates per horizon, NOT only one third of the hybrid's fitting exposure.
Both use four local transitions and the same short-recursive/bound penalty.
Only the hybrid adds symbolic supervision; extra loss and compute are disclosed.

Capacity is determined before fitting from architecture counts, not outcomes:
choose the closest continuous trunk width among 384..512 in steps of8, smaller
width breaking ties; total-parameter tolerance2%. This selects width456:
1,862,396 continuous versus 1,837,690 hybrid parameters (1.344% difference).
All modules must receive task gradients and update; no padding/dead heads.
Continuous controller width131 versus hybrid128 matches total controller
parameters within0.2%, with only eligible output logits. Reports separately
measure parameters in executed operators per pair, controller capacity and
linear MACs. Total-capacity matching is NOT active-path or FLOP equality.

Controllers retain the ORIGINAL first window per shot in BOTH arms. #73 showed
this coverage is pre-launch; it is deliberately the common reference for #75,
not an unnoticed fix applied to only one arm. Both have 1,800 updates per round,
batch128, lr0.001, and one aggregation round. DP utility is duration times
continuous carrier MSE plus0.0001 times executed linear MACs normalized by that
model's continuous-h15 cost, plus continuation. No symbolic labels enter the
controller loss/utility. Future continuous targets are TRAINING-only supervision;
runtime accepts only current carrier, interface action and remaining time.

## Development and compute boundary

Use the same 200 calibration states and 12 replay actions as #72. All policies
start from the same current B=1 agent observation and predict225 fixed steps.
The documented B=3 cache check uses three copies of ONLY that same initial image;
its offline audit time is recorded separately. No counterfactual rollout starts
from a candidate-specific observed post-launch frame. No new videos are needed.

The report includes all per-seed regret/top1/top3, ties/failures, pair usage,
common-endpoint carrier/count errors, executed MACs and synchronized planning
wall time including shared perception. No infilling occurs. Mean paired-state
contrasts have descriptive bootstrap intervals; the seed is not a new independent
state and three seeds do not establish seed-robust superiority. Collapse to one
mode is visible in usage, not masked or fixed by forced exploration.

This is a common-candidate-budget quality/work frontier, NOT an equal-FLOP
superiority claim. All continuous policies are eligible; there is no forced h15
handicap or adaptive-only extra action budget. #76 must prospectively freeze its
numerical work/frontier and feasibility gate before fresh access. #74 readiness
never authorizes #64 and does not overwrite #72's negative screen.

## Commands

Activate `novphy` and source `env.sh`, as usual. Run sequentially; do not launch
multiple copies against the same output directory.

```bash
python -u -m scripts.run_issue_74_matched_dynamics --prepare \
  2>&1 | tee -a data/issue-74-prepare.log

python -u -m scripts.run_issue_74_matched_dynamics --train --device cuda \
  2>&1 | tee -a data/issue-74-training.log

python -u -m scripts.run_issue_74_matched_dynamics --score-development --device cuda \
  2>&1 | tee -a data/issue-74-development.log

python -u -m scripts.run_issue_74_matched_dynamics --validate --device cuda \
  2>&1 | tee -a data/issue-74-validate.log
```

Train logs identify seed, arm, stage, update/lineage counts, elapsed/ETA and peak
memory. Development logs each of72 candidate-policy scores per state per seed.
Predictors checkpoint every90 updates; controllers every60. Both restore optimizer
and sampler state. Labels and state scores resume from completed, bound artifacts.
Use the same command after interruption, not a new freeze. Work since the most
recent atomic checkpoint can be repeated and is not exactly recoverable in cost
totals; reports disclose this limitation. Source changes require explicit versioning.

Full job planning allowance: up to four GPU-hours for fitting/labels, plus
development scoring (ETA is measured during the run); this is not a measured
duration or automatic wall-time cap. Plan for3GiB CPU RAM,2GiB CUDA memory and
2GiB new artifacts. No additional captures or corpus-sized RAM allocation.
The real smoke peaked around2GiB CPU/0.1GiB CUDA; a small smoke cannot measure
the complete workload. Stop and report a larger unexpected memory requirement.

The implementation's bounded checks are:

```bash
python -u -m scripts.run_issue_74_matched_dynamics --dry-run
python -u -m scripts.run_issue_74_matched_dynamics --smoke-test --device cuda
python -m unittest tests.test_issue_74_matched_dynamics -q
```

Dry-run reads contracts only and writes nothing. Smoke uses real shards and
full-sized models,18 updates per arm with optimizer resume, both controller
rounds, the previously troublesome state4 anchor and all72 candidate-policy
predictions, then validates common endpoints/costs/work traces. Its temporary
weights are removed and it is NOT production evidence. Smoke report is retained
at `.local-artifacts/issue-74-matched-dynamics-v1/smoke.json`.

Full outputs live under `.local-artifacts/issue-74-matched-dynamics-v1/`:
`plan.json`, three `seed-*/{continuous,hybrid}/` checkpoint/label directories,
`seed-*/development/`, and `readiness.json`. Validation recomputes the publication
from saved outcomes/trace costs and strict source-bound checkpoints; it does not
rerun all neural inference or remeasure historical wall times. No hashes or
full-image integrity scans are added.

The plan records HEAD and exact relevant source text. Code is currently local,
uncommitted and unpushed; `archived_release=false`. Reproducible archive handoff
requires these new source files plus the referenced #70/#71 implementation,
frozen CNN/shards/source release and the #74 output directory. Commit/push and
large-artifact archival need their own authority; a local source snapshot is not
a pushed release. Inspect the completed report before closing #74 or starting #75.
