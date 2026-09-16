# Balanced-native v2 gradient-norm recovery

V2's seed-3 hybrid fit stopped before optimizer update 9,114 because the
gradient norm was non-finite. All saved model tensors and Adam states remained
finite; the failed step was not applied. Retain its exact checkpoint and budget
as immutable recovery evidence.

The read-only replay of that checkpoint and assigned batch is deterministic.
Its loss is 1.6729963 and every gradient is finite. The float32 norm reduction
overflows, while the float64 norm is finite: 7.2955313e35 without downscaling,
or 1.1132097e31 at the frozen 2^-16 backward scale. The repaired path produces
a finite clipped gradient with norm 0.999999999. The minimal two-element
finite-large-gradient regression reproduces the same reduction failure.

Change only the norm reduction to float64. Keep the loss, backward scale,
unit-norm clipping formula, AdamW recipe, every assigned sample/horizon/mode,
all three seeds, capacity matching, and controller recipe unchanged. This is
an arithmetic implementation correction, not a model-selection intervention.
No action outcome or fresh evidence informed it.

Freeze a separate numerical-recovery plan binding the original v2 plan, exact
original/repaired fitter and regression-test source, replay evidence, archived
failure checkpoint, and charged budget. Permit no other parent-source or
numeric/data change. Archive the failure before clearing only its failure
flag; compare all model and optimizer tensors against that archive and retain
the update counter 9,114, 9,041 applied assignments, 73 skips, and 1,791.4734
charged active seconds. Continue the same allowance, not a reset allowance.

Retain the four completed predictor cells and all completed common surfaces.
Resume seed-3 hybrid at its unapplied assignment 9,114, then fit seed-3 pure,
all six controllers, and the original engineering diagnostic. Publish the
separate source-repair receipt alongside the parent report. The normal v2
entrypoint remains source-frozen and cannot silently accept the changed
fitter; only this explicitly frozen recovery entrypoint authorizes it.

A new numerical/resource stop is retained and is not cleared by this receipt.
All claims remain engineering-only; disjoint development readiness and fresh
non-final evaluation remain closed.
