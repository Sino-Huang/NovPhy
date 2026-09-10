# #75 prospective decision: implement one controller-coverage pilot

This document is the pre-implementation protocol. #73 found pre-launch-only
controller windows in its bounded two-family audit; #74 supplied three paired,
independently trained continuous/hybrid dynamics fits. This is evidence to test
coverage, not proof of the cause or a promise of positive results. Preserve the
#72 negative development screen and #74 results unchanged.

## One intervention

Use ALL existing #71 windows (first and later, up to four per shot) for controller
DP labels and its one aggregation round, instead of first-window-only sampling.
All 600 controller-training lineages (inventory indices5,10,...,3000) remain;
no outcome filtering. Both continuous and hybrid controllers receive the same
coverage rule. Reinitialize each from its original #74 seed+7400, retain the
same architecture, optimizer, sample schedule, batch128, 1800 updates per round
and two rounds total. Same optimizer exposure, broader label coverage; additional
label-generation work is measured, not described as identical compute.

Reuse all six #74 predictors unchanged, the frozen #70 CNN/carrier, existing
12-action grid, task objective and225-fixed-step rollout endpoint. No dynamics
fit, new images, capture, high-arc correction, CEM change or other optimizer.
The existing60-step DP objective and225-versus-settled target mismatch remain
limitations; this pilot does NOT change them alongside coverage. Engine-derived
symbols never enter the continuous controller labels or either runtime policy.
Later observed frames are allowed training targets/contexts only, never a
candidate-specific post-launch starting state at deployment.

## Membership, controls and endpoint

Use seeds20260908/20260909/20260910 and exactly the same200 calibration states,
all12 candidates per state. Score both new adaptive controllers. Retain all six
original #74 policies and their saved outcomes/traces as controls; fixed-model
results need not be recomputed because their complete runtime is unchanged.
The genuine continuous baseline remains distinct from the SAME hybrid checkpoint
fixed to continuous h15. The no-model prior remains frozen ordinal09.

Report selected-action settled engine count cost and normalized ranking regret,
top1/top3, tied/informative states, best-observed headroom and selected pig-removal/
level-clear/contact/support outcomes from existing replay records. These are
single-shot replay outcomes, not newly executed multi-shot gameplay success.
Separately report recursive carrier and parsed pig/block count errors at225,
pair usage, prediction failures, symbolic-content interventions, measured wall
time, linear MACs and peak memory. All rollouts are B=1 and feed predictions
into the next step without truth resets. Keep the B=3 SAME-initial-image cache
audit separate from deployment perception time. No future-image leakage.

## Numeric development gate, frozen before new outcomes

All conditions must pass for `correction_supported_for_prospective_test`:

1. Complete all3 seeds and200 states, with at least20 informative states and
   zero prediction failures in either new policy.
2. New hybrid mean normalized regret improves over the original hybrid adaptive
   by at least0.02, averaged equally over the3 seeds and200 states.
3. It improves by at least0.02 over the strongest declared comparator: the best
   across-seed original continuous policy OR the new continuous controller,
   the original hybrid fixed-continuous-h15 control, and the ordinal09 prior.
   Eligible continuous policies have zero failures in every seed. No favorable
   seed selection; include all attempts.
4. Hybrid improvement over its original controller is strictly positive in at
   least2 of3 seeds. Mean common225-step carrier MSE is no worse than the original
   hybrid adaptive. Neither local loss nor carrier MSE alone can pass the pilot.
5. Aggregate hybrid second-most-used description mode AND second-most-used horizon
   each account for at least5% of decisions. Do not force variation. All six
   frozen hybrid micro/macro content-intervention checks (two per seed) must
   change the actual transition by more than1e-7 on the designated training input.
6. Every new policy uses at most30 seconds perception+planning per state.

Descriptive paired-state bootstrap:10000 draws, seed7201. Average seed contrasts
within each state before bootstrapping; do not count seed repeats/candidates as
independent lineages. Intervals and all per-seed results are reported, but this
is an exploratory point-estimate feasibility screen, NOT fresh confirmation,
not a passed #72 rerun, and not a seed-robust or equal-FLOP superiority claim.

Failure of any scientific condition gives `not_supported_by_this_pilot`, not
another automatic repair. Incomplete evidence or exhausted resources gives
`budget_or_readiness_insufficient`. Neither disposition authorizes #64/#65.
#76 must freeze a fresh protocol and actual numerical compute/precision gate
before any fresh data access even if this pilot passes.

## Resource bound and operator handoff

Planning estimate from #74 measured costs: approximately25–40 minutes for wider
label generation/controllers, plus roughly20–40 minutes for scoring; actual ETA
will be logged. Hard cumulative active-command wall allowance:3600s for training/
labels,3600s for scoring, checked after each saved unit. Stop without automatically
increasing either allowance. Persist counters across resumes. Work since an
unsaved checkpoint after a hard process kill is not exactly recoverable and must
be disclosed. Smoke/validation costs are reported separately, not hidden fits.

Memory planning caps:3GiB CPU RSS and2GiB CUDA allocated, checked at saved units;
new artifact allowance6GiB (about four times #74 label storage plus new traces).
One process, one lineage/minibatch at a time; no corpus-sized RAM preload.
Preserve partial evidence at a budget stop and allow publication/validation of
an explicit insufficient-readiness disposition. No new hashes or image scans.

Before the operator job: no-write dry-run; real bounded smoke covering both
families, all window positions, both controller rounds, checkpoint resume,
same-input225-step predictions and source-bound publication validation; focused
regressions for coverage symmetry, no future/symbolic runtime inputs, no predictor
updates, exact resume, unchanged original controls and the frozen gate.

New output root: `.local-artifacts/issue-75-controller-coverage-v1/`.
No #74 source or artifact is rewritten. Source bindings are exact source text,
existing identities and checkpoint contracts, not new checksums. Code remains
local/uncommitted unless separately authorized; no pushed/archived release claim.
