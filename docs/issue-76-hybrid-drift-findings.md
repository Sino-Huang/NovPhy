# Frozen-weight drift and training-start exposure diagnosis

The failed matched-dynamics qualification motivated three frozen-weight probes:
symbolic-head gradient conflict, shared dynamics gradients across modes, and
local versus recursive error. No optimizer was constructed and no checkpoints
were updated. All probes use existing training data only, not fresh outcomes.

## Gradient evidence

On five assigned training lineages, three seeds, and three horizons, compare
the same 32 observed-start examples across the hybrid's three modes. All 135
pairwise shared-trunk gradient comparisons have positive cosine similarity.
This does not support conflicting same-batch mode gradients at these final
checkpoints. It does not exclude conflict earlier in training or between
different minibatches.

The dynamics and symbolic-supervision gradients on the macro head disagree in
34/45 comparisons; the micro head disagrees in 18/45. Symbolic-supervision head
gradient norms are often much larger than dynamics head gradients. This makes
macro-head conflict a candidate, not a demonstrated cause of the micro-policy
failure. Symbolic supervision is applied to observed carriers and directly
trains the heads, not the dynamics trunk; do not describe these numbers as
direct symbolic-supervision gradients on shared trunk parameters.

## Two distinct prediction problems

Mean first-transition error across all 15 seed/lineage combinations is already
large. At horizon 750 it is .048659 for hybrid micro versus .010781 for pure
continuous. After four recursive transitions, the errors are .044302 versus
.007947. Thus the hybrid deficit is present within its training unroll length,
not exclusively after long extrapolation.

Both arms additionally exhibit severe short-horizon recursive drift. At the
11,250-step endpoint, pure horizon-50 mean recursive MSE by seed is
2,757.75 / 3,201.99 / 2,812.24, while its observed-context local MSE at that same
endpoint is approximately .000093 / .000067 / .000139. Hybrid horizon-50 micro
has corresponding recursive MSE 612.54 / 4.44 / 6.31. The earlier favorable pure
result at horizon 750 is not evidence that its other horizons are ready.

All curves feed predictions forward without truth resets. Local predictions
are separately labeled observed-context diagnostics. Exact missing endpoints
remain unavailable. The final endpoints reproduce all 180 corresponding
fixed-policy values in the previously published 30-row qualification report
exactly.

## Deployment-start states are rarely sampled during fitting

The large first-transition deficit motivated counting exact shot-start exposure
in the original sampling schedule. The mixed schedule preserves those same
per-pair sampled-start multiplicities. Among 248 usable training lineages there
are 284 observed shot starts. Failed assignments remain in the inventory.

For hybrid fixed-750-micro, the exact counts are:

| Seed | All sampled starts | First-shot-start examples | Lineages whose first-shot start was sampled |
| --- | ---: | ---: | ---: |
| 760930001 | 21,152 | 42 | 38/248 |
| 760930002 | 21,152 | 37 | 36/248 |
| 760930003 | 21,152 | 52 | 48/248 |

Pure horizon-750 receives 63,488 examples per seed, including 139 / 139 / 172
first-shot starts from 109 / 105 / 119 lineages respectively. Across other
hybrid pairs only 31–53 lineages supply their first-shot start. The remainder
can still generalize from other states, but the counts establish a substantial
sampling mismatch with the first-shot anchors used for deployment/readiness.
This is not proof that unseen individual starts necessarily cause bad errors.

## Next controlled change

Prioritize a separately frozen shot-start-balanced sampling experiment before
changing model capacity, losses, or controller incentives. A concrete candidate
replaces half of the original per-update starts with uniformly drawn observed
shot starts from the same assigned lineage, then applies the existing within-
pair batch mixing. Doing the replacement before mixing preserves paired
horizon-level sample exposure across arms. The other half remain original
uniform starts. This proposal changes the start distribution and must not be
described as preserving all previous sampled examples.

Preserve failures, training membership, architecture, initialization, losses,
and optimizer budget; add no oracle runtime inputs or fabricated endpoints.
Do not launch it until its exact sampler, tests, resource limits, qualification
criteria, and source-bound protocol are frozen. Short-horizon recursive drift
remains a separate issue to assess even if first-transition learning improves.
Neither this diagnosis nor the proposed change passes the advancement gate.

## Artifacts and cost

- `scripts/diagnose_issue_76_hybrid_drift.py` produced all 30 gradient/curve
  records in `data/issue-76-hybrid-drift-diagnostic/report.json` in 14.889 CPU
  wall seconds with one thread.
- `scripts/diagnose_issue_76_start_exposure.py` produced all 36 per-pair/seed
  exposure summaries and their complete 250-lineage count inventories in
  `data/issue-76-start-exposure-diagnostic/report.json` in 2.059 CPU wall seconds.
- Both reports contain their executable source, source-plan identity, and
  explicit no-optimization/no-fresh-access status. Each had a 180-second ceiling.
- Two new tests verify that gradient probes leave weights and gradient buffers
  unchanged, and that curves preserve first-step equality and missing endpoints.

Activate `novphy` and source `env.sh` before running either module. Published
reports are immutable; run-specific timings are not overwritten on repetition.
