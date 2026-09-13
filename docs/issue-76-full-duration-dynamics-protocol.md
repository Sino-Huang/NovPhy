# Prospective paired full-duration recursive continuation

The boundary-balanced stage repaired first-transition prediction but failed
overall qualification and retained severe horizon-50 drift. Preserve that
negative result. This separately identified stage changes the recursive
training duration, not architecture, sampler, local/symbolic objectives, or
controller incentives. It is not a retroactive extension of the old experiment.

## Initialization and exact exposure

Continue ALL six completed `issue-76-boundary-dynamics-v1` predictors, seeds
760930001–3, retaining each model and its AdamW state. No seed/checkpoint choice,
fresh initialization, or optimizer reset. Source checkpoints have 6,000
completed scheduled updates, 5,952 applied and 48 skipped. New progress files
have their own plan identity and stage-local counters starting at zero while
retaining all per-parameter AdamW moments and step counters. Record source
checkpoint paths and prior-stage costs; previously paid work is not free.

Use absolute update positions 6000–7799, preserving the original pair schedule
and update-local generators. Reconstruct the boundary-balanced starts before
mixing: 16 uniform observed shot starts from the assigned lineage and 16 retained
uniform starts, with exactly the same absolute generator indices as the source
sampler. Mix only within each exact pair, with groups ordered by first appearance
in this continuation interval. Do not renumber absolute sampler/pair indices
from zero. All 250 assigned training lineages remain; the two unusable
assignments are skipped, not replaced.

Each arm/seed receives 1,800 scheduled positions, 1,786 applied updates,
14 skipped positions, and 57,152 sampled starts (28,576 deliberate boundary
draws plus 28,576 retained uniform draws). Paired horizon-level lineage/start
multiplicities remain identical across arms. Hybrid examples are divided among
its three modes. Keep batch 32, AdamW lr .0003, weight decay .0001 and gradient
clip 1. Resume clean interruptions from the saved new-stage counter and
optimizer state; failed or unclean jobs require audit, not silent replay.

## Objective and data contract

Use the tested `issue_76_full_duration_loss` implementation. Recursive targets
reach the same 11,250-native-step duration as evaluation: 225 / 45 / 15
transitions at horizons 50 / 250 / 750. Predictions feed predictions without
truth resets or gradient truncation. Average the recursive MSE and existing
.01 carrier-bound penalty over the available recursive steps. The local loss
still averages at most four observed-context transitions, with equal weight
between local and recursive averages. Direct symbolic supervision remains
only at observed offsets 0–4 with its original coefficient. Pure fitting
receives no symbolic target dictionary.

Targets must be exact observed endpoints within the same shot segment. Keep
the original adjacent-availability mask and stop when no batch row has an
available adjacent target pair. Admitted intact windows provide contiguous
prefixes; no interpolation, invented tail, cross-shot target, or true-state
reset is introduced. Reuse the same repaired common representation without
updates, recording its standalone cost separately. No new captures, oracle
runtime inputs, held-out fitting data, or fresh access.

## Frozen training qualification

After ALL six continuations, evaluate the same first-assigned training lineage
per family and its first-shot anchor. Retain all nine hybrid and three pure
fixed-policy local/recursive curves at the original diagnostic checkpoints.
Keep both the boundary and older mixed references in every paired record.
These five training lineages diagnose readiness; they do not establish
held-out generalization or actual gameplay competence.

Every fixed policy in each arm/seed must have mean 11,250-step recursive MSE
<=80% of the unchanged-carrier mean and beat that reference on at least 4/5
lineages. Missing required endpoints fail qualification. Do not hide a failed
short horizon behind horizon-750 performance.

Additionally, each seed must retain its first-transition horizon-750 mean MSE
within 110% of the boundary checkpoint (micro for hybrid, continuous for pure).
Hybrid horizon-750 micro endpoint mean must be within 110% of its boundary
reference. Pure horizon-750 endpoint mean must be within 110% of BOTH its
boundary and its older mixed reference, equivalently the stronger of their
means. This prevents a weaker pure recipe from manufacturing a hybrid win.
No choice of a deployable pure policy or per-seed fresh control follows from
this conservative training-reference check.

All policies and all six seed/arm cells must satisfy their requirements.
Report policy-level stability and retained-accuracy decisions separately.
Publish failures without extending this stage or changing criteria. Passing
only supports the next separately specified development-readiness work;
actual gameplay, useful joint switching, strong fixed/prior comparisons,
uncertainty, matched deployment compute, and once-only fresh advancement
remain unproven. No controller fitting is included here.

## Resources and publication

Root `.local-artifacts/issue-76-full-duration-dynamics-v1`, publication
`data/issue-76-full-duration-dynamics`. The completed no-update resource probe
used batch 32 for every pair, finite full gradients, peak allocated GPU
418.7 MiB, and at most one second per forward/backward pass. Its record and
source are included in the plan. Approximate per-fit estimates are 676 seconds
hybrid and 487 seconds pure before production overhead, roughly one hour total.

Freeze ceilings of 1,800 active seconds per arm/seed for continuation,
300 CPU seconds per seed for preparation, and 600 CPU seconds for diagnostics.
CPU RSS <=12 GiB, allocated GPU <=8 GiB, new artifacts <=1 GiB. No transfer of
unused budgets and no concurrent GPU research job. Retain complete parent and
new-stage cost records; distinguish optimizer continuation from diagnostic
comparison work and reused common training.

Initialize `novphy` and source `env.sh`. Run focused tests, real-schedule
exposure checks and the no-write module invocation, then `--prepare`.
Commit/push and record the freeze in #76 BEFORE `--run --device cuda`.
Archive all six new-stage checkpoint counts, all 30 assigned diagnostic
records/180 policy curves, complete source bindings, decisions and resource
records; exactly validate the completed publication. #64/#65 remain unauthorized.
