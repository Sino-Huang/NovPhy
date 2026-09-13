# Full-duration continuation: validated qualification failure

The frozen [protocol](issue-76-full-duration-dynamics-protocol.md) completed
all six fits and all 30 assigned training diagnostic records. Independent
validation exactly recomputed all 180 policy curves, checked the source-bound
plans, all checkpoint counters, each parameter's AdamW step increment, and
recorded costs. Qualification is **false** for every arm/seed. This is a
training-readiness failure, not an executed fresh negative experiment and not
support for #64/#65. No fresh or final data were accessed.

## Results

The table reports mean recursive carrier MSE at the common 11,250-native-step
endpoint. Each mean uses the same five first-assigned training lineages.
H750 is hybrid micro; P750 is independently trained pure continuous.

| Seed | H50 continuous | P50 continuous | H750 micro | P750 continuous | Unchanged |
| --- | ---: | ---: | ---: | ---: | ---: |
| 760930001 | 0.068948 | 0.089263 | 0.011294 | 0.009455 | 0.047606 |
| 760930002 | 0.228870 | 0.357303 | 0.008564 | 0.026885 | 0.050641 |
| 760930003 | 0.252706 | 0.048834 | 0.023641 | 0.007483 | 0.051597 |

All 12 horizon-50 policy/seed cells fail the frozen endpoint qualification.
Only 3/12 horizon-250 cells pass; all 12 horizon-750 cells pass their
unchanged-carrier endpoint comparison. Thus 15/36 policy/seed cells pass that
comparison, but none of the six complete arm/seed qualifications pass.
The publication retains every mode, reference and individual curve.

All six horizon-750 anchors fail initial-step accuracy retention: new/source
MSE ratios are 1.282, 1.136 and 3.202 for hybrid, and 3.595, 9.993 and 1.438
for pure, against the frozen maximum 1.10. Endpoint accuracy retention also
fails for hybrid seeds 760930002/3 and pure seed 760930002. Pure endpoint
retention uses the stronger of the boundary and older mixed references;
neither older result is discarded.

Full-duration training substantially reduced the preceding horizon-50
explosion, but did not make short-horizon recursion useful or retain local
accuracy. These measurements do not identify the remaining cause by
themselves. They do not justify a controller fit, held-out success claim,
post-hoc horizon exclusion, threshold change, or extension of this frozen
stage. Next work must diagnose the remaining local/recursive tradeoff on
permitted development data and specify any repair separately before fitting.

## Reproducibility and cost

Frozen fit source: commit `a897fca`; independent validator: `05fae7b`.
Every fit completed 1,800 scheduled positions: 1,786 applied, 14 skipped,
57,152 sampled starts, with source model and AdamW state retained.
Six-fit active time totals 3,534.558 seconds (58.91 minutes); the separate
diagnostic used 8.907 CPU seconds and exact validation 9.315 CPU seconds.
Peak fitting/diagnostic RSS was 2,666.934 MiB and allocated CUDA memory
443.309 MiB. All recorded jobs finished without resource-limit stops.
Parent-stage and common-representation costs remain in the source-bound
plan and are not treated as free.

Artifacts: [report](../data/issue-76-full-duration-dynamics/report.json),
[validation](../data/issue-76-full-duration-dynamics/validation.json), and
[additional validation cost](../data/issue-76-full-duration-dynamics/validation-budget.json).
The unchanged qualification and advancement flags remain false. Actual
gameplay, useful adaptation, strongest-control comparisons, uncertainty,
and matched-compute advancement remain unproven.
