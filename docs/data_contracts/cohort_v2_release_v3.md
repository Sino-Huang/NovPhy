# `cohort_v2_release_v3`

Release v3 uses the corrected readiness-first executor and the immutable
plan-v3 successor. Execution, attempt, replay, accounting, quality, inventory,
derivation-index, publication, public-bundle, sealed-bundle, and validation
identities are all version 3 and derive from the plan-v3 collection identity.

The operator command uses fresh destinations:

```sh
python -u -m scripts.capture_issue_53_evidence \
  --plan-root data/runtime_evidence/issue-53-plan-v3 \
  --runtime-root .local-artifacts/issue-53-stable-only-production-run-v3 \
  --output data/runtime_evidence/issue-53-stable-only-v3 \
  --sealed-output .local-artifacts/issue-53-stable-only-final-release-v3 \
  --authorization-identity github-issue-authorization-v3:53:stable-only-production-after-anchor-fix
```

The v2 runtime and incomplete release must remain untouched. After execution,
run the same command with `--validate` in place of the authorization identity.
A complete result still requires 24 accepted `stable_entered` rollouts, all
role/stratum quotas, four exact replays, complete derivations, and valid
public/sealed boundaries.
