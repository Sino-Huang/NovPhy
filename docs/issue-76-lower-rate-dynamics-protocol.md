# Paired lower-rate full-duration continuation

The finite-step diagnostic demonstrated two total-objective overshoots and
three horizon-750 anchor overshoots eliminated by one-tenth parameter steps.
One-tenth steps improved the full objective in all 36 probed cells, but four
direction-predicted anchor regressions remained. This supports testing a
lower rate, not a claim that optimization or readiness is solved. Preserve
the full-duration qualification failure and both diagnostic reports.

## Fixed intervention and exposure

Continue all six completed full-duration checkpoints, seeds 760930001–3,
with their model weights, AdamW moments and per-parameter step counters.
The sole optimizer change is learning rate .0003 to .00003 in every loaded
parameter group. Constructor arguments alone do not override saved AdamW
rates. Keep weight decay .0001, betas, epsilon, gradient clipping at norm1,
architecture, repaired shared perception/history, batch32 and all objectives.
No optimizer reset, checkpoint choice, seed removal, new capture or fresh
data access. Shared perception already has semantic supervision; pure refers
to independently trained continuous dynamics, not an end-to-end symbol-free
perception system.

Use a separately identified stage of 6,000 scheduled positions with absolute
indices7800–13799. Source checkpoints completed 1,800 positions (1,786 applied,
14 skipped) in their own stage. New counters begin at zero; prior costs and
optimizer history remain charged. Reuse the existing continuation schedule:
original absolute update-local generators, 16 uniformly sampled observed
shot starts plus16 retained uniform starts, then mixing only within the exact
mode/horizon pair. Do not reset absolute sampler indices or cross-horizon
sample membership. Audit all three seeds before fitting.

All250 assigned training lineages remain, including the two unusable ones.
Expected per-arm/seed exposure is5,952 applied updates,48 skipped positions,
190,464 starts (95,232 deliberate boundary draws plus95,232 uniform draws).
Verify exact paired horizon-level lineage/start multiplicities, not just
counts. No outcome-conditioned sampling or replacement. Six thousand
positions give more optimization time at the smaller rate; this is one
prospectively fixed continuation, not a rate sweep or early-stop selection.

Recursive predictions still reach11,250 native steps through225/45/15
transitions at horizons50/250/750, without truth resets or gradient truncation.
Only exact observed within-shot targets and existing availability masks are
used; no invented tail. Keep the original four-step local average and
observed-offset0–4 symbolic terms. Pure fitting receives no symbolic labels.

## Qualification and limits

After all six fits, retain all30 assigned training records and180 new policy
curves alongside boundary, mixed and full-duration references. Use the same
first-assigned lineage per family and first-shot anchor. Every policy in every
arm/seed must have mean endpoint MSE <=80% of unchanged carrier and win on
at least4/5 lineages. Missing endpoints fail. Do not omit horizon50.

Keep first-step horizon750 MSE within110% of the boundary reference (hybrid
micro, pure continuous); hybrid micro endpoint within110% of boundary; pure
continuous endpoint within110% of both boundary and older mixed references.
Additionally retain pure endpoint accuracy within110% of the completed
full-duration reference, preserving its stronger first/third-seed results.
This extra safeguard cannot relax any prior criterion. No deployable pure
policy or fresh control is selected by this training-only check.

All six arm/seed cells must qualify. Report any failure without extending
this stage or changing its criteria. Passing would support separately planned
development readiness, not actual gameplay, meaningful adaptive switching,
strongest-control superiority, uncertainty or matched-compute advancement.
No controller fitting, fresh/final access or #64/#65 authorization here.

Root `.local-artifacts/issue-76-lower-rate-dynamics-v1`; publication
`data/issue-76-lower-rate-dynamics`. Record300 CPU seconds for the exposure
audit,180 CPU seconds for source-checkpoint preflight,300 CPU seconds preparation per seed,3,600 active seconds fitting per
arm/seed and600 CPU seconds for diagnostics. No allowance transfers. Peak
RSS <=12GiB, allocated CUDA <=8GiB, new local artifacts <=1GiB. Based on the
completed full-duration run, expected total fitting time is about3¼ hours.
Only one GPU fit at a time; retain progress, resource records and deterministic
resume for clean interruptions. Audit failed/unclean jobs before replay.

Initialize novphy and source env.sh. Run focused tests, no-write invocation
and `--audit`, then `--prepare`. Archive source/plan and record the freeze in
#76 before `--run --device cuda`. After completion exactly validate all six
checkpoints, optimizer history/rate, every diagnostic curve, decisions and
costs; publish the full outcome. No new hashes or full-corpus integrity passes.
