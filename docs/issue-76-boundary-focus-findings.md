# Boundary-focused continuation: validated qualification failure

All six frozen fits completed 1,800 scheduled positions each: 1,785 applied
updates and 15 retained skips. The independent validator exactly recomputed
all 30 assigned training records and 180 policy curves, checked checkpoint
identity, optimizer settings and per-parameter step continuity, and verified
the full 11-job resource inventory. Aggregate qualification remains false:
only hybrid seeds 760930002 and 760930003 qualify. No fresh/final data were
accessed and advancement remains unproven.

## What improved and what failed

| Anchor | Lower-rate initial MSE | Boundary-focused initial MSE | Boundary-focused endpoint MSE |
| --- | ---: | ---: | ---: |
| Hybrid 760930001 | 0.001690 | 0.001646 | 0.004090 |
| Pure 760930001 | 0.002283 | 0.002048 | 0.003194 |
| Hybrid 760930002 | 0.005253 | 0.004704 | 0.002600 |
| Pure 760930002 | 0.005399 | 0.004642 | 0.002823 |
| Hybrid 760930003 | 0.004286 | 0.004289 | 0.002562 |
| Pure 760930003 | 0.002480 | 0.002897 | 0.002096 |

Anchors are horizon-750 micro for hybrid and continuous for pure; endpoints
are 11,250 native steps on the same five assigned training lineages.

The two original pure local failures improved, but not enough: seed 760930001
has initial MSE 0.002048 versus the boundary reference 0.001678, and seed
760930002 has 0.004642 versus 0.003657. Both exceed the retained 1.10 ratio.
Pure seed 760930003 now fails the additional lower-rate local-retention check:
0.002897 exceeds 1.10 times 0.002480. Its older boundary check still passes;
that does not excuse losing the stronger source accuracy.

Hybrid seed 760930001 also fails the horizon-50 micro endpoint win-count
criterion: only three of five individual wins, below the required four.
Its mean endpoint MSE 0.035346 passes the separate 0.8-times-unchanged mean
criterion (unchanged MSE 0.047606). Both criteria are required. The other
35 policy/seed endpoint comparisons pass. All six old anchor endpoint checks
and all three additional full-duration pure endpoint checks pass; all six
also retain lower-rate anchor endpoint accuracy.

The one-step diagnostic correctly predicted useful local improvements in the
two failing seeds, but did not establish a qualified full-fit repair. Removing
uniform starts introduced other regressions. Neither the sampling change nor
a longer training budget is sufficient evidence for further deployment.

## Next action and reproducibility

Do not extend this frozen stage, drop a failing seed/policy or relax a gate.
Retain both lower-rate and boundary-focused checkpoints and every stronger
pure reference. The next development-only repair should address local
accuracy while preserving mixed-start exposure and recursive accuracy; the
earlier local-loss-weight probe provides evidence to investigate, not a
guarantee of success. A separate prospective recipe is required before fitting.

Frozen fit, protocol and validator source: `911e9cf`.
Independent validation used 9.451 CPU seconds and peak RSS 684.305 MiB.
All training, preparation, preflight (including its rejected count assumption),
diagnostic, validation and parent-stage costs are retained. No resource-limit
stop occurred. Training qualification does not substitute for development
readiness, trained controllers or the full actual adaptive-gameplay advancement
gate with strongest controls, practical margins, uncertainty and matched compute.

Artifacts: [report](../data/issue-76-boundary-focus/report.json),
[validation](../data/issue-76-boundary-focus/validation.json),
[validation cost](../data/issue-76-boundary-focus/validation-budget.json), and
[frozen protocol](issue-76-boundary-focus-protocol.md).
