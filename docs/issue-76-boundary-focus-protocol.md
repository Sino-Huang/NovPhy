# Paired boundary-focused continuation

The completed lower-rate stage passes all endpoint comparisons but fails
initial-accuracy retention for two pure seeds. The prospectively paired
[intervention probe](issue-76-local-interventions-findings.md) supports
boundary-focused sampling as the next single-factor candidate. It does not
prove a full-fit repair. Preserve every earlier negative result and checkpoint.

## Frozen training recipe

Continue all six completed lower-rate predictors: paired seeds 760930001,
760930002 and 760930003, each with hybrid and independently trained pure
arms. Keep the common 352-value carrier, capacity matching, all 250 training
assignments (including the two unusable assignments), all three horizons and
all hybrid modes. No fitting on calibration/model-selection/fresh/final data.

Run exactly 1,800 scheduled positions per arm/seed, absolute indices
13,800..15,599, comprising 1,785 applied and 15 skipped updates. This is a
fixed continuation of about seven passes through the assigned lineages, not
an early-stopped search. Use only the last scheduled checkpoint. This length
is a prospective engineering choice, not an effect-size guarantee from the
one-step probe. No extension or checkpoint selection within this stage.

Pre-freeze preparation corrected an initial count assumption: the unusable
training indices are 56 and 132, and the partial final pass repeats index 56.
Thus there are 15 skips, not the preceding stage's 14. The regression fixture
uses these actual indices. No fitting occurred during the rejected preflight;
its 4.403 CPU seconds remain in the cumulative preflight budget. The schedule
and membership were not changed to meet an expected count.

The sole training-recipe change is 100% observed shot-boundary starts instead
of 50% boundary / 50% uniform starts. Batch size remains 32. For each assigned
lineage/update, retain the prior even-row boundary RNG (seed, 1,000,000 plus
absolute update); odd rows now independently sample its observed shot starts
with (seed, 2,000,000 plus absolute update). Mix within the same exact pair
using the existing schedule. Audit exact paired horizon-level lineage/start
multiplicities and verify every sampled start is an observed shot boundary.
Use all observed shot boundaries, not just the five diagnostic first shots.

Retain all source parameters, optimizer moments and per-parameter counters.
AdamW learning rate remains 0.00003, weight decay 0.0001 and gradient clip 1.
Keep the unchanged local-first-four plus full-recursive loss and hybrid
symbolic loss. Recursive supervision extends without resets or truncated
backpropagation to 11,250 native steps wherever observations are available.
Do not create targets beyond censored observations.

## Qualification and retained controls

Recompute every one of the existing 30 assigned training diagnostic records
and 180 policy curves. Each of all 36 policy/seed endpoint cells must retain
the previous mean-MSE ratio <=0.8 to unchanged carrier and at least four of
five individual wins. Retain the existing 1.10 boundary initial-step and
endpoint retention criteria, the stronger pure boundary/mixed endpoint
reference, and the additional full-duration pure endpoint check.

Additionally, all six horizon-750 anchors must retain both initial and endpoint
accuracy within 1.10 times their lower-rate source. This prevents discarding
stronger current references; it only adds requirements. Qualification requires
all six complete arm/seed qualifications. Retain all prior curves and controls.
No threshold, mode, horizon, seed or lineage exclusion follows outcomes.

These are training-readiness checks, not #76 advancement. Subsequent permitted
development readiness, controllers, real adaptive gameplay, strongest-pure,
same-hybrid fixed and strong-prior comparisons, practical margins/uncertainty,
useful nondegenerate adaptation and matched-compute gates remain required.

## Execution, cost and validation

Prepare/audit source checkpoints and actual schedules before fitting. Archive
source, this protocol and the public plan before `--run`. Initialize novphy
and source env.sh. Use sequential CUDA fits with 1,800 active seconds per
arm/seed, 300 CPU seconds for preflight and for each seed's preparation, 600
CPU seconds for the diagnostic and 180 CPU seconds for independent validation.
Retain the shared 12 GiB RSS / 8 GiB allocated-CUDA limits and 1 GiB new local
artifact cap. Prior training, representation, validation and probe costs
remain in the source-bound plan and are not free.

Independent validation must replay all curves/decisions exactly, inspect all
six checkpoint counters and optimizer settings/step increments, compare the
source-bound frozen/public plans, and verify the complete recorded budget
inventory. Deterministic resume retains completed fits and never restarts
solely because observation timed out. No new capture or fresh access occurs.
