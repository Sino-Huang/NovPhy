# `cohort_v2_release_v4`

Release v4 uses assignment-specific mixed termination expectations and fresh
v4 attempt, replay, accounting, release, publication, public-bundle, and sealed
bundle identities. All existing readiness, frozen-byte, atomic validation,
coverage, derivation, replay, and exposure-boundary gates remain unchanged.

The operator command uses new immutable destinations:

```sh
python -u -m scripts.capture_issue_53_evidence \
  --plan-root data/runtime_evidence/issue-53-plan-v4 \
  --runtime-root .local-artifacts/issue-53-mixed-termination-production-run-v4 \
  --output data/runtime_evidence/issue-53-mixed-termination-v4 \
  --sealed-output .local-artifacts/issue-53-mixed-termination-final-release-v4 \
  --authorization-identity github-issue-authorization-v4:53:mixed-termination-production
```

A complete disposition requires 24 accepted rollouts, exact 4 `level_fail` / 20
`stable_entered` assignment outcomes, all central role/stratum quotas, four
passing exact replays, complete source-bound derivations, and valid public and
sealed boundaries. V1 through v3 artifacts remain immutable history.
