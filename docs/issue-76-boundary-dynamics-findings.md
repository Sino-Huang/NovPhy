# Shot-start-balanced dynamics: initial-state repair, overall qualification failed

The experiment frozen at `f39effd` and recorded
[before fitting in #76](https://github.com/Sino-Huang/NovPhy/issues/76#issuecomment-5652911624)
completed all six independent predictors. Each used 6,000 scheduled updates,
5,952 applied and 48 skipped, with 95,232 deliberate boundary draws and 95,232
retained uniform draws. Architecture, initialization, objectives, and update
budgets stayed fixed. There was no controller fitting, new capture, or fresh
access.

## Frozen decisions

All hybrid seeds pass their first-transition, endpoint-reduction, and
unchanged-carrier criteria. Two pure seeds fail the allowed endpoint-regression
limit, so the overall qualification is **false**, without changing thresholds.

| Seed suffix | Arm | First-step MSE: reference → boundary | Endpoint MSE: reference → boundary | Qualified |
| --- | --- | --- | --- | --- |
| 001 | Hybrid, 750-micro | .042422 → .002191 | .127440 → .022786 | Yes |
| 002 | Hybrid, 750-micro | .054350 → .005154 | .055028 → .006018 | Yes |
| 003 | Hybrid, 750-micro | .049204 → .004142 | .042298 → .010346 | Yes |
| 001 | Pure, 750-continuous | .008147 → .001678 | .010239 → .012334 | No |
| 002 | Pure, 750-continuous | .013667 → .003657 | .013220 → .011150 | Yes |
| 003 | Pure, 750-continuous | .010530 → .002913 | .012185 → .015270 | No |

Seeds are 760930001–3. Values are means over the five preassigned training
lineages at the first transition and exact 11,250-fixed-step endpoint.
All six listed boundary policies beat the unchanged carrier on 5/5 lineages.
That does not erase the pure-arm relative-regression failures or constitute
gameplay success. References are the completed matched-dynamics checkpoints,
not different carrier targets.

Hybrid first-step error fell approximately 91–95%; its endpoint error fell
76–89%. This supports sparse shot-start sampling as a contributor to the
previous initial-state deficit under this specific controlled training change.
It does not establish that all earlier hybrid failures share that cause.
Pure first-step error also improved, but endpoint error rose about 20.5% and
25.3% in seeds 001 and 003, beyond the frozen 10% limit. Seed 002 improved.

## The remaining short-horizon failure

Every horizon-50 policy, in both arms and every seed, fails to beat the unchanged
carrier at the endpoint on any of the five lineages. Mean horizon-50 micro
recursive errors for hybrid seeds 001/002/003 are 97.53 / 5.73 / 179.09; pure
continuous errors are 1809.06 / 1260.90 / 1603.51. Corresponding local endpoint
errors remain small. Thus learning shot starts does not solve the separate
problem of feeding predictions back over 225 transitions.

At horizon 250, only hybrid seed 002's three modes beat the unchanged carrier
on all five lineages. Other seed/arm horizon-250 policies beat it on none.
All nine hybrid and three pure policy curves remain in the publication;
the favorable horizon-750 results are not presented as all-horizon readiness.

The next targeted repair should address the mismatch between four training
offsets and the 15/45/225 recursive transitions used at the evaluation endpoint.
A separately frozen, paired full-duration recursive objective is warranted
before controller retraining. Its implementation, exposure, checkpoint
initialization, and runtime/memory costs must be tested and declared before
optimizer work. Do not silently extend this failed fit, relax its pure-arm
criterion, select only favorable seeds, or treat a first-step repair as an
advancement result.

The earlier pure checkpoints remain stronger eligible controls on the
prespecified endpoint contrast in two seeds. Their results must stay visible
in subsequent development comparisons. A future hybrid cannot claim an
advantage solely by weakening the pure training recipe.

## Complete validation and costs

The prepared validator at `258138f` recomputed all 30 paired records and all
180 policy curves exactly. It also checked the frozen source plan, six
checkpoint update counts, all qualification decisions, original cost records,
memory limits, and explicit no-fresh/no-advancement status.

Artifacts are under `data/issue-76-boundary-dynamics/`: `plan.json`,
`report.json`, `validation.json`, and the additional `validation-budget.json`.
The final checkpoint files are retained under
`.local-artifacts/issue-76-boundary-dynamics-v1/checkpoints/`.

Predictor active seconds were approximately 313.4 / 316.7 / 315.3 for hybrid
and 261.0 / 262.4 / 262.3 for pure. Preparation took about 15.1 seconds total,
diagnostics 9.0 seconds, and separate exact validation 9.1 seconds. Peak fitting
CPU RSS was 2,653.7 MiB and allocated GPU memory about 61 MiB. Reused common
representation training remains an additional standalone cost, recorded in
the source plan rather than reported as free.

Activate `novphy`, source `env.sh`, and run
`python -m scripts.validate_issue_76_boundary_dynamics` to recompute validation.
Repeated validation accumulates its separate 180-second CPU allowance; its
fixed validation artifact is unchanged. The published validation-budget file
is the snapshot from this first complete validation.

No actual gameplay, useful joint switching, strongest-baseline uncertainty,
matched-deployment-compute, or fresh advancement requirement is established.
#76 remains unpassed; #64/#65 remain unauthorized.
