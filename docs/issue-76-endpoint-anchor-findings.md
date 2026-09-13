# Targeted endpoint continuation: validated retention failures

All six frozen fits completed 900 scheduled positions each, with 893 applied
updates and seven retained skips. Independent validation exactly replayed
all 30 diagnostic records and 180 policy curves, checked checkpoint identity,
optimizer settings and per-parameter step continuity, and verified the
complete 11-job resource inventory. Aggregate qualification is false: hybrid
seeds 760930002/3 and pure seed 760930002 qualify; the other three do not.
No fresh/final data were accessed and advancement remains unproven.

## Results

All 36 policy/seed endpoint comparisons against unchanged carrier pass both
the mean-MSE and individual-win criteria. This does not override failures
against stronger previously trained references.

| Horizon-750 anchor | Initial MSE | Endpoint MSE | Qualification |
| --- | ---: | ---: | --- |
| Hybrid 760930001 | 0.001687 | 0.004010 | fail: targeted-local initial retention |
| Pure 760930001 | 0.001973 | 0.003862 | fail: local and endpoint retention |
| Hybrid 760930002 | 0.004335 | 0.002492 | pass |
| Pure 760930002 | 0.003884 | 0.002676 | pass |
| Hybrid 760930003 | 0.003439 | 0.002313 | pass |
| Pure 760930003 | 0.001896 | 0.002329 | fail: stronger endpoint retention |

Hybrid anchors execute micro mode; pure anchors execute continuous mode.
Endpoints are 11,250 native steps on the same five assigned training lineages.

Pure seed 760930002 repairs its remaining local failure: initial MSE decreases
from 0.004128 to 0.003884, below the unchanged limit of approximately 0.004023.
Its endpoint also improves from 0.002975 to 0.002676. Hybrid seed 760930003
improves both anchor metrics as well.

However, hybrid seed 760930001 loses initial accuracy relative to the
targeted-local reference (0.001687 versus 0.001409). Pure seed 760930001
relapses on the original boundary initial check (0.001973 versus 0.001678),
also loses targeted-local initial accuracy, and still fails endpoint retention
against the boundary-focused reference (0.003862 versus 0.003194).
Pure seed 760930003 improves local accuracy but loses endpoint accuracy
against both the boundary-focused and targeted-local references (0.002329
versus 0.002096 and 0.001906). The retained maximum ratio is 1.10 throughout.

## Interpretation and next step

The favorable one-step endpoint probe did not translate into simultaneous
full-fit retention across all seeds. The saved results show a local/endpoint
tradeoff and sensitivity to the continued training trajectory; they do not
by themselves establish its cause. Preserve all prior stronger checkpoints
and references, including pure seed 3's targeted-local endpoint and pure
seed 2's new endpoint. No failing cell may be dropped or rounded into a pass.

Do not extend this frozen stage. Next diagnosis should distinguish systematic
objective tradeoffs from variation along the training trajectory before
another recipe is specified. A favorable transient perturbation is not a
sufficient predictor of a long continuation's outcome. Training qualification
still cannot replace held-out development readiness or actual adaptive
gameplay against the strongest required controls with practical margins,
uncertainty, useful adaptation and matched compute.

## Reproducibility and costs

Frozen fit, loss, protocol and validator source: `1fe499f`.
Six-fit active time totaled 1,861.902 seconds (31.032 minutes). Diagnostics
used 9.009 CPU seconds and validation 9.345 CPU seconds. Diagnostic peak RSS
was 2,701.676 MiB; validation peak RSS was 720.902 MiB. All jobs completed
without resource-limit stops. Preflight, preparation, all earlier branch and
representation costs, and the paired probe remain recorded, not free.

Artifacts: [report](../data/issue-76-endpoint-anchor/report.json),
[validation](../data/issue-76-endpoint-anchor/validation.json),
[validation cost](../data/issue-76-endpoint-anchor/validation-budget.json), and
[protocol](issue-76-endpoint-anchor-protocol.md).
