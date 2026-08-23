# `cohort_v2_release_v1`

> Historical authority: the failed issue-53 run and both human-review bundles
> are retained unchanged as evidence of the v1 termination-expectation defect.
> New production uses the stable-only v2 contract.

Issue #53 executes the immutable issue-52 plans directly. The executor runs the
24 role/stratum slots once, in frozen exposure-role and intervention order, with
zero retries or outcome-conditioned replacement. It checkpoints an attempt
ledger after every rollout and prints the current attempt, role, intervention,
status, and terminal result immediately.

Every accepted rollout contains one validated `physics_capture_v2` artifact and
one request-72 `observation_trace_manifest_v1` artifact. Contact/support, the
two accepted macro labels, and the two accepted physical-violation labels are
published as three separately versioned source-bound derivations per rollout.
Four additional exact-socket replay proofs—one per exposure role—use the frozen
issue-48 comparison rules and are not cohort rollouts or retries.

Final-evaluation access remains fail closed. The dry run validates only the
public three-role authorities and the sealed projection; it does not materialize
or read the final scenario. Actual execution requires an explicit authorization
identity. The command records an authorized copy of the frozen issue-47
workflow, audits the access, and keeps the final scenario, primary rollouts,
derivations, replay details, and exact access record in the separate sealed
bundle. Public manifests expose only the prospective final attempt membership
and sealed-bundle identity.

The non-launching check is:

```sh
python -u -m scripts.capture_issue_53_evidence --dry-run
```

The progress-reporting production command is:

```sh
python -u -m scripts.capture_issue_53_evidence \
  --runtime-root .local-artifacts/issue-53-production-run \
  --output data/runtime_evidence/issue-53 \
  --sealed-output .local-artifacts/issue-53-final-release \
  --authorization-identity github-issue-authorization-v1:53:production
```

Running the production command is the operator's explicit authorization to use
the frozen final-evaluation workflow under the supplied declared identity. The
runtime and both publication paths must not already exist. Any rollout failure,
quarantine, coverage shortfall, termination mismatch, replay failure, or stale
binding publishes an incomplete disposition and never claims a complete
release.
