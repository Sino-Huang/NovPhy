# `cohort_v2_production_plans_v2`

> Historical failed authority: the first stable-only production determination
> consumed all 24 v2 attempts in pre-shot action normalization. It emitted no
> shot or physical outcome. Plan v3 preserves this bundle unchanged and issues
> fresh attempt identities after the executor anchor-order correction.

Issue #53 publishes the stable-only correction at
`data/runtime_evidence/issue-53-plan-v2`. Its immutable identities are
`cohort-v2-production-collection-plan-v2:issue-53:stable-only` and
`cohort-v2-production-parameter-plan-v2:issue-53:stable-only`. The issue-52
plans are superseded for execution, not rewritten.

The correction retains the same six interventions, ordering, actions, central
coverage strata, four exposure roles, one-attempt policy, and total quota of
24. Every intervention now requires `stable_entered`. The production
termination quotas are 0 `level_clear`, 0 `level_fail`, and 24
`stable_entered`. Clear and fail remain in the closed vocabulary and remain
accepted pilot capabilities, but they are non-quota-bearing for this
production cohort. Any non-stable production termination leaves the run
incomplete.

The correction evidence projects only the 18 public non-final issue-53
outcomes, the three public XML authorities' bird/pig counts, and their 18
validated camera-aligned v2 observation diagnostics. Human-review bundles and
reviewed final-evaluation outcomes are not plan-derivation sources.

Training seed 4401, calibration seed 4501, and model-selection seed 4402 retain
their existing lineages. Final evaluation uses a fresh sealed `type010102`
instance at seed 4503. The public plan contains only its sealed projection,
v2 inventory, v2 partition manifest, and pending v2 access workflow; the XML,
scenario manifest, and parameter realization remain in the separate sealed
authority.

Publish or exactly revalidate the plan with:

```sh
python -u -m scripts.build_issue_53_plan_v2
python -u -m scripts.build_issue_53_plan_v2 --validate
```

The executor dry run does not open or materialize the final authority:

```sh
python -u -m scripts.capture_issue_53_evidence \
  --plan-root data/runtime_evidence/issue-53-plan-v2 \
  --dry-run
```
