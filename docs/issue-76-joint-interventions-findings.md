# Joint-retention probes support targeted endpoint supervision

The [protocol](issue-76-joint-interventions-protocol.md) was frozen at
`0896a1b` before execution. All 36 policy/seed cells and 144 temporary
perturbations completed with finite measurements and exact parameter
restoration. No saved predictor checkpoint was changed and no fresh/final
data were accessed.

## Results

Counts compare each variant with the current-objective control update.
The local-weight-1 variant is identical to control at horizons 50 and 250.

| Variant | Better local anchor | Better anchor endpoint | Better updated-policy endpoint |
| --- | ---: | ---: | ---: |
| Add endpoint term | 4/36 | 21/36 | 19/36 |
| Local weight 1 | 0/36 | 0/36 | 0/36 |
| Boundary only | 30/36 | 23/36 | 27/36 |

The endpoint term has a particularly consistent horizon-750 result: all 12
hybrid/pure policy/seed cells improve both local and anchor endpoint errors
relative to the original checkpoint, and all 12 improve anchor endpoint
error relative to control. Only three improve local error relative to control;
an endpoint improvement can reduce the magnitude of the local improvement.

| Pure seed, horizon 750 | Local change from original | Endpoint change from original |
| --- | ---: | ---: |
| 760930001, add endpoint | -5.269% | -7.893% |
| 760930002, add endpoint | -2.242% | -8.884% |
| 760930003, add endpoint | -11.470% | -13.923% |

For comparison, the current-objective control changes these endpoints by
-6.314%, +0.405% and -4.755%. The endpoint term therefore repairs the adverse
one-step endpoint direction for pure seed 760930002 while retaining a local
improvement. It also improves the failing pure-seed-1 endpoint more than
control. These are one-step changes, not full qualification decisions.

Reducing horizon-750 local weight from 10 to 1 beats control on neither
anchor metric in any of the 12 horizon-750 cells. The predicted improvement
from reducing local weight is not supported by these measurements.

Boundary-only sampling remains useful in many cells, but its pure-seed-2
horizon-750 endpoint still increases by 0.071%, and the earlier full
boundary-only training introduced other failures. Added endpoint loss at
shorter horizons is inconsistent: for pure seed 1, its horizon-250 update
worsens the local anchor and improves the endpoint less than control; for
pure seed 3, its horizon-50 policy endpoint worsens by 1.712%. The evidence
does not justify applying an extra endpoint term indiscriminately.

## Next candidate and limitations

The best-supported next candidate adds a unit-weight endpoint MSE term only
at horizon 750, retaining local weight 10 there, mixed-start exposure, saved
optimizer history and every other horizon/mode. This follows all 12 paired
horizon-750 cells, not selection of a favorable seed. Freeze a separate
recipe before fitting; do not extend or alter the completed targeted-local
stage. Preserve every stronger reference and all existing qualification gates.

This is a controlled ten-start training diagnostic. It does not establish
the cause of the whole training trajectory, generalization to other lineages,
or actual adaptive gameplay competence. The two original residual checks
remain failed at the saved checkpoints, and #76 advancement remains unproven.

## Checks and cost

The output inventory was checked against the complete expected 36-cell
matrix and four variants per cell. Every unperturbed anchor and policy
endpoint was compared with the independently validated targeted-local
report; maximum absolute CPU/GPU difference was 2.292e-8. The endpoint-loss
test verifies full recursive gradients and exclusion of unavailable targets;
four focused tests passed before the probe.

The probe used 89.594 active seconds against a 600-second CUDA budget, peak
RSS 1,549.242 MiB and allocated CUDA memory 304.601 MiB. Earlier replay
costs are retained in the [complete report](../data/issue-76-joint-interventions/report.json),
along with assignments, source text, all individual measurements and
unchanged-objective measurements. No resource-limit stop occurred.
