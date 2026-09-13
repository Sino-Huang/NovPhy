# Frozen-weight gradient tradeoff diagnosis

The full-duration stage failed local retention for all six fits and endpoint
qualification for all horizon-50 policies. Keep that completed result unchanged.
The read-only replay `python -m scripts.reproduce_issue_76_local_retention`
reproduces all six local failures from actual checkpoints. The minimized
`--seed 760930001 --pure-only` replay reproduces exactly 0.001677635586 versus
0.006031142641 first-step MSE twice; it requires only the first pure seed,
the five assigned training anchors and one horizon-750 transition per anchor.
The five lineages are retained because the frozen criterion is their mean.
This isolates a readiness regression, not the full gameplay advancement gate.

Before testing, rank these predictions:

1. Cross-horizon interference: the horizon-50/250 total-loss gradient is
   negatively aligned with the horizon-750 anchor-loss gradient, more often
   than horizon-750's own gradient.
2. Within-horizon loss conflict: recursive and local gradients are negatively
   aligned and the recursive norm dominates. This can explain local damage
   without cross-horizon interference.
3. Optimizer history: the retained AdamW direction increases anchor loss even
   when the current negative-gradient direction decreases it.

Compute all 72 pair/checkpoint probes: every three-seed hybrid (nine pairs)
and pure (three pairs), at both the boundary and completed full-duration
checkpoints. Use exactly the same first-assigned training lineage per family
and first-shot anchors as the completed diagnostic. No outcome-selected
additional samples. Batch five anchors; exact observed within-segment targets,
availability masks and full recursive durations remain unchanged.

Separately differentiate local, recursive and complete objectives, including
the existing symbolic loss when applicable. Check their summed loss against
the production implementation. Report per-offset adjacent availability,
losses, gradient norms and cosines, including each total gradient's alignment
with that checkpoint's horizon-750 first-step anchor loss (micro for hybrid,
continuous for pure). Compute the next clipped AdamW parameter delta from
the saved moments without applying it. Positive anchor/delta cosine predicts
local worsening to first order; negative anchor/gradient cosine predicts the
same for ordinary gradient descent. Report both, without treating derivatives
as measured outcomes of a fit or proof of long-run causality.

No model or optimizer mutation, captures, controller fitting, fresh/final
access, or advancement claim. Verify exact model tensors unchanged. Record
one cumulative 600-active-second CUDA probe budget, <=12 GiB RSS, <=8 GiB
allocated CUDA, and <=10 MiB publication. Archive source before `--run`.
Use the existing initialized `novphy` environment and `env.sh`.
No new full-corpus checks or hashes. The result chooses the next discriminating
development experiment; it does not select a fresh checkpoint or relax gates.
