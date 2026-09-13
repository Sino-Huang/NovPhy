# Targeted local weighting: repaired local error, qualification still false

All six fits completed the frozen 1,800 scheduled positions, with 1,785
applied updates and 15 retained skips each. Independent validation exactly
recomputed all 30 training diagnostic records and 180 policy curves, checked
all checkpoint identities, optimizer settings and per-parameter step history,
and verified the complete 11-job resource inventory. Four of six arm/seed
qualifications pass; aggregate qualification remains false. No fresh/final
data were accessed and no advancement claim is supported.

## Results

All 36 policy/seed endpoint comparisons pass the unchanged mean-MSE and
individual-win criteria. All three hybrid arms qualify. Pure seed 760930003
also qualifies. The earlier hybrid horizon-50 micro win-count regression is
absent in this mixed-start branch.

| Horizon-750 anchor | Initial MSE | Endpoint MSE | Qualification |
| --- | ---: | ---: | --- |
| Hybrid 760930001 | 0.001409 | 0.003969 | pass |
| Pure 760930001 | 0.001647 | 0.003819 | fail: stronger endpoint retention |
| Hybrid 760930002 | 0.004293 | 0.002386 | pass |
| Pure 760930002 | 0.004128 | 0.002975 | fail: initial accuracy |
| Hybrid 760930003 | 0.004110 | 0.002794 | pass |
| Pure 760930003 | 0.002098 | 0.001906 | pass |

Hybrid anchors execute micro mode; pure anchors execute continuous mode.
Endpoint MSE uses 11,250 native steps on the same five assigned training
lineages. Every older curve remains in the report.

The original pure-seed-1 local failure is repaired: initial MSE 0.001647 is
below its original boundary reference 0.001678. However, its endpoint MSE
0.003819 exceeds 1.10 times the stronger boundary-focused reference 0.003194.
Improving local accuracy does not excuse this endpoint-retention failure.

Pure seed 760930002 improves initial MSE from the lower-rate source's
0.005399 to 0.004128, but still exceeds 1.10 times the boundary reference
0.003657 (limit approximately 0.004023). That remaining difference must not
be rounded into a pass. Its endpoint retention checks pass.

Pure seed 760930003 improves both initial and endpoint accuracy relative to
both recent branches. Its new endpoint MSE 0.001906 is a stronger reference
that subsequent work must retain. Both local and endpoint quality matter;
neither the older weaker pure controls nor a same-hybrid fixed control can
replace the strongest eligible independent pure baseline.

## Interpretation and next step

The targeted local-weight change repairs the first pure local failure and
substantially reduces the second while preserving all endpoint comparisons
against unchanged carrier. It has not simultaneously met every retained
local and endpoint requirement. This is evidence of progress, not proof that
the candidate is ready for controllers or fresh evaluation.

Do not extend this frozen stage or weaken any threshold. Preserve all seeds,
policies, lineages, checkpoints, branch costs and stronger references. The
next development-only diagnosis should measure updates at these completed
checkpoints against both remaining symptoms: pure-seed-1 endpoint retention
and pure-seed-2 local retention, with all other cells retained as paired
checks. Any subsequent repair requires a separate prospective recipe.

## Reproducibility and cost

Frozen fit/loss/protocol/validator source: `69ed09e`. Six-fit active time was
3,628.897 seconds (60.482 minutes); diagnostics used 9.004 CPU seconds and
independent validation 9.357 CPU seconds. Diagnostic peak RSS was 2,679.848
MiB; validation peak RSS was 699.758 MiB. All jobs finished without a
resource-limit stop. Preparation, preflight and all parent/branch costs
remain recorded and must not be treated as free or counted twice.

Artifacts: [report](../data/issue-76-local-anchor/report.json),
[validation](../data/issue-76-local-anchor/validation.json),
[validation cost](../data/issue-76-local-anchor/validation-budget.json), and
[protocol](issue-76-local-anchor-protocol.md).
Actual adaptive gameplay, strongest controls, useful adaptation, practical
margins with uncertainty and matched-compute advancement remain unproven.
