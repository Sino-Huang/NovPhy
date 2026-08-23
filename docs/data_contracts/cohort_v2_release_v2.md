# `cohort_v2_release_v2`

> Historical failed determination: all 24 v2 attempts were quarantined before
> a shot because live-relative actions were normalized before readiness
> anchoring. The incomplete public/sealed release remains immutable.

Issue #53's v2 executor selects its immutable authority with `--plan-root`.
Plan identities, quotas, assignments, attempt/replay identities, partition and
access authorities, and copied plan bytes are resolved from that directory.
The stable-only plan produces v2 execution, replay, quality, inventory,
accounting, sealed-boundary, derivation-index, release, publication, bundle,
and validation schemas and identities.

All v1 validation, atomic quarantine, source-bound derivation, exact replay,
readiness, camera/slingshot alignment, exposure redaction, and release
exclusion rules remain in force. The v2 quality gate additionally requires all
24 accepted rollouts to terminate with `stable_entered`. Clear or fail is a
valid engine termination in the closed vocabulary but a production mismatch,
so it makes the release incomplete.

`--validate` is read-only. It checks the selected plan, the runtime's frozen
plan bytes and execution/replay reports, and the public/sealed release
boundaries without launching Unity.

The operator production command is:

```sh
python -u -m scripts.capture_issue_53_evidence \
  --plan-root data/runtime_evidence/issue-53-plan-v2 \
  --runtime-root .local-artifacts/issue-53-stable-only-production-run-v2 \
  --output data/runtime_evidence/issue-53-stable-only-v2 \
  --sealed-output .local-artifacts/issue-53-stable-only-final-release-v2 \
  --authorization-identity github-issue-authorization-v2:53:stable-only-production
```

This implementation intentionally stops after the dry run. The operator runs
the command above separately. Issue #53 remains open until read-only validation
confirms 24 accepted role/stratum rollouts, 24 `stable_entered` terminations,
four passing exact replays, complete derivations, and valid public/sealed
boundaries.
