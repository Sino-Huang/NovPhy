# `cohort_v2_production_plans_v3`

Issue #53 plan v3 is the immutable successor to the failed stable-only v2
determination. Its identities are
`cohort-v2-production-collection-plan-v3:issue-53:stable-only:anchor-order-correction`
and
`cohort-v2-production-parameter-plan-v3:issue-53:stable-only:anchor-order-correction`.

Plan v3 does not change the scientific plan. It retains all four lineages,
seed 4503, the six ordered intervention offsets, central coverage strata,
stable-only termination quotas, one attempt per slot, zero retries, and all
release gates. It creates fresh attempt identities because the v2 attempts are
immutable failure evidence and cannot be rerun within their consumed plan.

The v2 failure occurred before every shot: `collect_rollouts` normalized an
unanchored slingshot-relative action before readiness could attach the live
screen anchor. The corrected executor lets `prepare_screen_shot` establish the
stable camera/slingshot projection and anchor the action before deriving the
socket command. The regression test exercises the exact plan action shape,
which intentionally has no frozen `drag_start`.

Validate and dry-run the successor without final-data access:

```sh
python -u -m scripts.build_issue_53_plan_v3 --validate
python -u -m scripts.capture_issue_53_evidence \
  --plan-root data/runtime_evidence/issue-53-plan-v3 \
  --dry-run
```
