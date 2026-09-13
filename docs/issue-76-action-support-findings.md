# Native action support before counterfactual replay

The native training source does not provide alternative-action outcomes from
the same initial decision state. Every one of its 250 assigned training
lineages starts with drag `(-80, 10)`, release 600 ms, tap 0 ms. Both independently
trained dynamics arms share this limitation. The completed fixed-development
evaluation therefore tests recorded-action prediction, not selection among the
12 alternative launch actions.

The [source-bound audit](../data/issue-76-action-support/report.json) inspects
the existing collection plan and action metadata in all 250 assigned training
shards. It retains unusable assignments and distinguishes planned shots from
actually available observed segments. It does not inspect outcome fields,
compute predictor scores, fit weights, collect data or open fresh evidence.

| Shot | Drag | Assigned training segments | Available observed training segments |
| --- | --- | ---: | ---: |
| First | (-80, 10) | 250 | 248 |
| Second | (-60, 45) | 50 | 20 |
| Third | (-10, 80) | 50 | 16 |

Second and third shots occur only in family `type010102`. They occur after
earlier interventions and therefore cannot be reused as alternative first-shot
outcomes. Counts above describe available segments, not optimizer sample counts
or successful gameplay. The two unusable training assignments remain recorded.

## The action grid must be explicit

The historical shared broad-action design uses x offsets
`{-10, -60, -110, -160}`, y offsets `{-80, 0, 80}`, tap 0 ms and release 600 ms.
None of these twelve exact actions appears at the first shot of the native
training source. Only `(-10, 80)` appears at any shot, in sixteen observed
third-shot segments of the multiple-force family.

The native live loader accepts caller-supplied bounds. Its unit-test fixture
uses a different x range, `[-160, -40]`, producing a different twelve-action
grid. That fixture is not a frozen learned-gameplay protocol and must not
silently replace the historical design. This audit does not freeze new
deployment bounds or alter either source.

Exact-action absence does not prove that neural interpolation/generalization
fails, and this audit does not establish a cause of the hybrid/pure near-tie.
It does establish that the native source contains no direct paired launch-state
evidence for ranking the historical grid. Nor does it rule out compatible
evidence elsewhere: the audited scope is the native 350-lineage source.

## Next evidence, not yet launched

Use a separately frozen training-only action-replay protocol to measure actual
task utility and available ranking headroom. Keep original scenario lineages,
roles and engine-seed pairing; compare all predeclared alternative actions from
equivalent initial states; retain failures, ties and physically early terminals.
Freeze the shared action bounds, original-action reference, exact replay
inventory, scoring definitions and resource limits before rendering. A small
training diagnostic cannot establish strongest gameplay controls or authorize
fresh evaluation.

Do not retrain a controller merely to produce symbolic choices, infer
counterfactual ranking from recorded-action carrier MSE, or launch a learned
gameplay policy using controllers fitted to older predictors. The full #76
advancement requirements and all prior negative findings remain unchanged.

Two focused regression tests pass. The actual training-action metadata scan
took 4.350 seconds wall time, with 652.7 MiB peak RSS, under a 60-second CPU
wall allowance. Reproduce after initializing the `novphy` environment and
sourcing `env.sh`:

```bash
python -m unittest tests.test_issue_76_action_support
python -m scripts.audit_issue_76_action_support --audit
```

The audit retains its completed report on rerun without reopening the shards.
