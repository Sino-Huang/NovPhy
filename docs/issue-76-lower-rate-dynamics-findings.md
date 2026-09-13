# Lower-rate continuation: endpoint recovery, qualification still false

All six fits in the frozen [protocol](issue-76-lower-rate-dynamics-protocol.md)
completed. Independent validation exactly recomputed 30 records and 180 policy
curves and checked checkpoint identity, optimizer continuity and recorded costs.
Four of six arm/seed qualifications pass; the aggregate qualification is false.
No fresh or final data were accessed. This is training-readiness evidence, not
an advancement result or support for #64/#65.

## Results

Every one of the 36 policy/seed endpoint comparisons passes the unchanged
criterion: mean carrier MSE at 11,250 native steps is at most 0.8 times the
unchanged-carrier baseline, with five of five assigned training lineages won.
All six endpoint-retention checks pass, including all three additional pure
checks against the stronger full-duration reference.

| Seed | H50 continuous endpoint | P50 endpoint | H750 micro endpoint | P750 endpoint | Unchanged endpoint |
| --- | ---: | ---: | ---: | ---: | ---: |
| 760930001 | 0.014486 | 0.029891 | 0.004136 | 0.003841 | 0.047606 |
| 760930002 | 0.023936 | 0.023010 | 0.002523 | 0.003035 | 0.050641 |
| 760930003 | 0.026228 | 0.020013 | 0.003077 | 0.002259 | 0.051597 |

The remaining failures are initial-step accuracy retention for pure seeds
760930001 and 760930002. Their horizon-750 initial MSEs are 0.002283 and
0.005399, versus boundary references 0.001678 and 0.003657: ratios 1.361 and
1.476 exceed the frozen 1.10 maximum. All three hybrid anchors and pure seed
760930003 retain initial accuracy. Neither failing seed is excluded and the
threshold is not relaxed.

The lower-rate stage substantially improves long recursive predictions and
recovers every short-horizon endpoint comparison. It does not establish why
two pure local anchors remain worse, nor justify another unplanned extension.
The next step is a development-only diagnosis of that residual local error,
followed by a separately frozen paired repair if the evidence supports one.

## Reproducibility and costs

Frozen fit source: `5bcfac7`; independent validator: `49386f4`.
Each arm/seed completed 6,000 scheduled positions, with 5,952 applied and 48
skipped updates. Source model and AdamW moments were retained; the frozen
learning rate was 0.00003. Six-fit active time totals 11,866.064 seconds
(197.77 minutes). Diagnostic CPU time was 8.965 seconds and independent
validation 9.363 seconds. Maximum fitting/diagnostic RSS was 2,708.742 MiB;
maximum allocated CUDA memory was 443.645 MiB. All jobs finished without
resource-limit stops. Preparation, audit, preflight and parent-stage costs
remain recorded in the source-bound report and plan.

Artifacts: [report](../data/issue-76-lower-rate-dynamics/report.json),
[validation](../data/issue-76-lower-rate-dynamics/validation.json), and
[validation cost](../data/issue-76-lower-rate-dynamics/validation-budget.json).
Actual adaptive gameplay, strongest-pure and fixed-policy controls, practical
margin with uncertainty, useful mode/horizon adaptation and matched-compute
advancement remain unproven.
