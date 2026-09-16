# Balanced-perception native dynamics refit v2

This is a new engineering candidate after the retained event-ranking failure.
It does not alter or extend that completed stage.  The failure showed that the
original shared parser produced nearly invariant decision carriers, while the
already completed balanced-pig parser diagnostic passed its predeclared
outcome-independent semantic criteria on all three seeds.  Reuse exactly those
three frozen parser checkpoints as the common perception surface for both
dynamics arms.  No event outcome enters this refit.

The frozen v1 execution completed four predictor cells, then its seed-3 hybrid
cell failed after update 9,114.  Its forward loss was finite; the audit is
consistent with float32 backward overflow through the 226-step delta-50
recursion, followed by unchecked non-finite gradient clipping.  The optimizer
step made all 40 tensors on the active macro/continuous path and their Adam
moments non-finite, while the five inactive micro-head tensors remained finite.
Update 9,115 detected the damage.  The exact input batch was finite and
bounded, completed seed-1 and seed-2 models differentiated it finitely, CUDA memory was stable,
and the failed checkpoint is retained.  V2 is an audited new execution, not an
automatic retry: refit all six predictors from scratch and retain the v1
failure binding in the frozen plan.

Retain the original native-refit training membership, action encoding,
22-slot vocabulary, observed-history contract, horizons 50/250/750, three
paired seeds 760930001--3, and deterministic capacity-matching rule.  Only the
250 original training assignments enter an optimizer; unavailable assignments
remain scheduled skips.  Calibration, model-selection, event-replay, and fresh
outcomes are excluded.

For each seed, encode every assigned trajectory with the corresponding frozen
balanced parser.  Fit a new shared observed-history encoder from scratch for
1,000 scheduled AdamW updates (lr 0.001), then freeze it for both arms.  Fit
independent hybrid and pure-continuous dynamics from scratch for 12,000
scheduled AdamW updates per arm (lr 0.0003, batch 32).  The first 6,000 use the
original exact four-transition local/recursive objective.  The final 6,000 use
the pre-existing full-duration recursive objective through the available
11,250-native-step window.  Both arms receive the identical update and sampled
start schedules; the hybrid retains its declared micro/macro auxiliary labels,
and the pure arm has no symbolic modules or mode embedding.

For predictor backward passes, multiply the scalar loss by `2^-16` before
automatic differentiation, compute the norm on those scaled gradients, apply
the equivalent unit-norm clipping coefficient, and divide the clipped
gradients by `2^-16` before AdamW.  This preserves the intended unit-clipped
gradient while keeping long-recursion intermediate gradients inside float32.
Any still-non-finite scaled gradient is a hard retained failure before the
optimizer step.  Common-history and controller fitting retain their original
unscaled backward passes.

Fit the existing parameter-matched controllers for 4,000 scheduled updates
per arm/seed (lr 0.001, batch 32).  Their dynamic-programming teacher remains
training-only.  Set the predeclared linear-MAC tradeoff coefficient to 0.0001,
instead of the older 0.01 whose cost term dominated measured local-error gaps
and collapsed every hybrid decision to continuous mode.  This coefficient is
fixed from the scale of the already published outcome-independent controller
diagnostic; no event action outcome is consulted.

Each common job has 3,600 active seconds, each predictor 14,400 seconds, and
each controller 3,600 seconds, with 12 GiB RSS and 8 GiB allocated CUDA limits.
Retain every checkpoint, skipped assignment, teacher, work counter, and
failure.  No automatic retry, seed deletion, or threshold adjustment follows
a resource or numerical stop.

The completed refit is still an engineering candidate.  Existing event
training/calibration/model-selection outcomes are already exposed and may be
used only for clearly labeled engineering diagnosis.  Before any readiness
claim, freeze and collect a new development cohort disjoint from every prior
lineage.  Fresh non-final evaluation remains closed.
