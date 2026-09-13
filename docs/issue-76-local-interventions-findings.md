# Local-retention probes favor boundary-focused continuation

The [frozen protocol](issue-76-local-interventions-protocol.md), archived at
`dd58f3a`, completed all 36 paired policy/seed cells and 144 temporary
perturbations. All measurements are finite. Each original parameter tensor
was restored exactly after every perturbation, and no predictor checkpoint
was written. No fresh/final data were accessed.

## One-factor results

Counts below compare each one-step variant with the retained-state control
update, not with the original checkpoint. Lower error is better.

| Variant | Better local anchor | Better anchor endpoint | Better updated-policy endpoint | Better unchanged mixed objective |
| --- | ---: | ---: | ---: | ---: |
| Local weight 10 | 29/36 | 23/36 | 23/36 | 34/36 |
| Boundary only | 32/36 | 25/36 | 27/36 | 10/36 |
| Reset optimizer history | 30/36 | 11/36 | 6/36 | 14/36 |

Control itself improves the original local anchor in 29/36 cells and the
updated-policy endpoint in 22/36. Boundary-only improves these in 29/36 and
27/36, respectively. Thus beating control does not always mean improving the
original checkpoint, and none of these counts establishes qualified training.

For the two failing pure seeds at horizon 750, boundary-only changes local
anchor error by -6.478% and -2.326% from the original checkpoint, compared
with control's -3.170% and -1.427%. Their endpoint changes are -2.507% and
-5.672%, versus control's +1.134% and -4.736%. Both predictions for boundary
dilution are supported in these particular one-step comparisons.

Local weight 10 produces stronger horizon-750 local improvements (-13.568%
and -5.017%) with endpoint improvements in both failing seeds, but produces
an 8.581% local-anchor regression for pure seed 760930003 at horizon 250.
It is not a universally beneficial local repair. Resetting optimizer history
improves many local anchors but increases pure horizon-50 endpoint errors by
493%, 1,024% and 479% across the three seeds. Retained moments should not be
discarded on this evidence.

## Interpretation and next step

The boundary-only result supports a separately specified, paired
boundary-focused continuation as the next candidate, retaining the current
loss, learning rate and optimizer history. This changes one training factor
and has the broadest observed endpoint improvement. It does **not** establish
that boundary sampling caused the entire residual error or will pass a full
fit. The diagnostic uses first-shot boundaries on five assigned training
lineages; production training uses all assigned lineages and shot boundaries.
The mixed-objective tradeoff remains visible and must not be discarded.

A new fit must preserve all three seeds, both arms, every horizon/mode,
shared exposure accounting, source checkpoints and stronger pure references.
The existing lower-rate qualification remains false. Thresholds must not be
relaxed, failing seeds excluded, or these temporary parameter states deployed.
All development-readiness and actual adaptive-gameplay advancement requirements
remain to be demonstrated after any qualified training repair.

## Checks and costs

The no-mutation AdamW calculation and the new history-reset copy each passed
their focused unit test, including agreement with an actual clipped AdamW
step. The full output inventory contains exactly the expected 36 distinct
cells and four variants per cell. Every unperturbed anchor and policy endpoint
was checked against the independently validated lower-rate report; maximum
absolute CPU/GPU difference was 1.407e-8. Perturbation outcomes are measured
diagnostics, not an independently rerun full training experiment.

The probe used 80.684 active seconds against its 600-second CUDA cap, peak RSS
1,523.063 MiB and peak allocated CUDA memory 212.100 MiB. The report also
retains the earlier replay's cumulative 0.650 CPU seconds. All recorded jobs
finished without resource-limit stops. Source text, assignments, individual
measurements and costs are in the [report](../data/issue-76-local-interventions/report.json).
