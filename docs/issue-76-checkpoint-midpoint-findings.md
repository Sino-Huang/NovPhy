# Fixed checkpoint midpoint: five qualifications, one retained failure

The one predeclared 50/50 parameter midpoint completed all 30 training records
and 180 policy curves. Independent validation reconstructed all six midpoint
models and exactly replayed every midpoint and parent-error comparison.
Five of six arm/seed combinations qualify; pure seed 760930001 still fails
the stronger boundary-focused endpoint-retention criterion. Aggregate
qualification remains false. No predictor checkpoint was saved or changed,
no optimization occurred, and no fresh/final data were accessed.

| Horizon-750 anchor | Midpoint initial MSE | Midpoint endpoint MSE | Qualification |
| --- | ---: | ---: | --- |
| Hybrid 760930001 | 0.001513 | 0.003888 | pass |
| Pure 760930001 | 0.001770 | 0.003703 | fail: stronger endpoint retention |
| Hybrid 760930002 | 0.004248 | 0.002278 | pass |
| Pure 760930002 | 0.003958 | 0.002644 | pass |
| Hybrid 760930003 | 0.003729 | 0.002451 | pass |
| Pure 760930003 | 0.001814 | 0.001888 | pass |

Hybrid anchors execute micro mode and pure anchors continuous mode. Endpoints
are 11,250 native steps on the same five assigned training lineages. All
36 policy/seed endpoint comparisons against unchanged carrier pass. Every
older stronger reference remains in the report, including the endpoint
continuation's new local and endpoint improvements.

The midpoint repairs hybrid seed 1's initial-retention failure and pure
seed 3's endpoint-retention failure. Pure seed 1 also passes its initial
checks, but endpoint MSE 0.003702689753845334 remains above the retained
limit 1.10 times 0.003193629835732281, approximately 0.003513.

## What the prediction comparison establishes

For pure seed 1's endpoint, parent MSEs are 0.0038189456332474946 and
0.0038617510115727784. Their error cross-product mean is
0.00355088550131768; the post-hoc prediction midpoint MSE is
0.0036956170340999963. Averaging therefore reduces error below both adjacent
parents, but still cannot meet the stronger boundary-focused limit. Parameter
and prediction midpoint errors are close in this remaining failed comparison.
Nonlinear parameter interpolation is not its principal observed obstacle.

The hypothesis of useful complementary variation receives partial support:
one common midpoint recovers several retention checks without extra fitting.
The stronger claim that averaging repairs all failures is rejected. Neither
these correlated adjacent errors nor a single midpoint establishes the full
cause of the training trajectory. The output midpoint is diagnostic only,
not a recurrent ensemble policy. No alternative averaging weights were tried.

## Scope and next step

Do not relabel the remaining failure, select weights by seed or silently
adopt this as a qualified deployment checkpoint. Preserve the failed
training qualification and all parent controls. Further work must distinguish
this training-proxy limitation from the actual development/gameplay readiness
required by #76; favorable carrier curves alone cannot support advancement.
Any bounded development evaluation or further repair needs its own prospective
specification, and fresh access remains closed until readiness is established.

## Reproducibility and costs

Frozen diagnostic source/protocol: `729e020`. The diagnostic consumed 25.771
CPU seconds with peak RSS 735.582 MiB. Independent reconstruction/replay
consumed 26.064 CPU seconds with peak RSS 743.477 MiB. Both completed within
their separate cumulative 180-second limits. Earlier reproduction costs are
retained in the report. All parent training and diagnostic work remains
chargeable; averaging does not make those costs free.

Artifacts: [report](../data/issue-76-checkpoint-midpoint/report.json),
[validation](../data/issue-76-checkpoint-midpoint/validation.json),
[validation cost](../data/issue-76-checkpoint-midpoint/validation-budget.json), and
[protocol](issue-76-checkpoint-midpoint-protocol.md).
Actual adaptive gameplay, strongest eligible pure/fixed/prior controls,
practical margins with uncertainty, useful adaptation and matched-compute
advancement remain unproven.
