# Paired finite-step overshoot diagnosis

The gradient probe contradicted widespread direct gradient conflict, while
leaving finite-step overshoot unresolved. Before any perturbation outcomes,
assign all36 pair/seed cells at the completed full-duration checkpoints,
using the same five assigned training lineages and first-shot anchors.
All three seeds, nine hybrid pairs and three pure pairs remain included.

For each cell, compute the full production-objective gradient and retained
clipped AdamW parameter delta using the tested no-mutation calculation.
Measure the local, recursive and total objective plus the horizon750
first-step anchor loss (hybrid micro / pure continuous) at the original
weights, and at0.1x and1x that delta in temporary model copies. Restore
exact weights after every pair; never change saved checkpoints or optimizer
moments. These72 perturbations are diagnostic counterfactuals, not fitted
candidates or a checkpoint-selection procedure.

Overshoot prediction: total or anchor loss rises at1x despite its negative
directional derivative, while0.1x improves it. Report all cases, including
failures of both scales, both improvements, and worsening predicted by the
derivative. Never infer that a lower learning rate will pass a full fit or
gameplay gate from this one-batch check alone.

Record one cumulative300-active-second CUDA budget, <=12GiB RSS, <=8GiB
allocated CUDA and <=10MiB publication. No captures, fresh/final access,
controller fitting, checkpoint writes or advancement authorization. Archive
this source and protocol before `--run`; use initialized novphy and env.sh.
