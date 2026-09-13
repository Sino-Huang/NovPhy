# Residual local-retention diagnosis

The lower-rate stage's independent replay established that endpoint criteria
pass but pure seeds 760930001 and 760930002 fail initial accuracy retention.
This is a training tradeoff to investigate, not yet an identified code defect.

## Fast red-capable replay

After initializing the novphy environment and sourcing env.sh:

```sh
python -m scripts.diagnose_issue_76_local_retention
```

Two executions returned exit status 1 with identical MSEs and reference ratios:

| Seed | Actual local MSE | Boundary reference | Ratio | Retained |
| --- | ---: | ---: | ---: | --- |
| 760930001 | 0.0022831326350569725 | 0.0016776355856563895 | 1.3609228694106874 | false |
| 760930002 | 0.005399298295378685 | 0.0036573459627106788 | 1.4762886394747685 | false |

Elapsed in-process times were 0.419 and 0.408 seconds. The minimized replay
uses only each failing pure checkpoint, the original five assigned training
lineages, and one actual horizon-750 prediction at each first shot boundary.
It exactly matches the independently validated full diagnostic means. It
omits later recursive predictions, other horizons and non-failing arms without
changing the symptom. All five lineages remain necessary for the prescribed
aggregate comparison. No model is fitted or saved; no held-out data are read.
The cumulative replay budget is recorded under
`.local-artifacts/issue-76-local-retention-diagnosis-v1/budgets/replay.json`.

## Ranked hypotheses, before intervention probes

1. **Local objective underweighted relative to recursive accuracy.** If so,
   increasing only the local-loss coefficient on identical training batches
   should improve the finite-step change in the failed local anchor. Measure
   the recursive endpoint too; local improvement alone is not a repair.
2. **Shot-boundary contribution diluted by other sampled starts.** If so,
   using only the boundary subset of the same batch, with loss coefficients
   unchanged, should improve local-anchor change relative to the mixed batch.
3. **Retained optimizer moments direct the update away from the anchor.** If
   so, replacing only those moments in a temporary optimizer copy should
   improve local-anchor change at the same learning rate and objective.

These are untested hypotheses, not causal findings. Probes must retain all
seeds/arms for paired reporting, restore temporary parameters, record costs,
and preserve the existing checkpoints and qualification criteria. A subsequent
training repair requires a separate prospective specification; this replay
does not authorize opening fresh data or claiming advancement.
