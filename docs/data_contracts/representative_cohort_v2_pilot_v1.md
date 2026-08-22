# `representative_cohort_v2_pilot_report_v1`

`representative_cohort_v2_pilot_report_v1` is the immutable acceptance report for
the central cohort-v2 pilot. It does not amend or promote the scoped v3 pilot.
The report binds the exact `cohort-v2-capabilities-v1` declaration, the
prospective pilot plan, all component evidence from issues #44–#50, the
supplementary issue-51 terminal probe, attempt accounting, environment and code
revisions, partition and replay authorities, and accepted derivation identities.

## Prospective plan

After the first supplementary run correctly failed its level-clear quota, the
active plan identity is
`representative-cohort-v2-pilot-plan-v2:cohort-v2-capabilities-v1:issues-44-through-51:determination-2`.
Its quotas come directly from the capability declaration and issue #51 rather
than realized outcomes. It requires all six central strata and all supported
central v2 terminal reasons: `level_clear`, `level_fail`, and `stable_entered`.
The exact component evidence identities and supplementary collection-plan
identity are frozen in the plan before the supplementary Unity attempts begin.

The failed determination-1 plan, attempts, captures, and realized shortfall
remain visible in determination-2 accounting and are not rewritten. The
issue-44 through issue-50 bundles remain immutable. Their actual
non-fixture Unity captures supply the exporter, observation, partition, replay,
macro-semantics, and physical-violation evidence. Issue #51 collects only the
missing `level_clear` terminal evidence under a separately source-bound
two-attempt plan. Determination 2 uses byte-identical `legacy_static` XML so the
target geometry cannot be changed by generator materialization.

## Representative audit

The report can set `representative_audit=true` only when:

- every required central capability has `passed` or the generated-only
  inventory records `explicit_legacy_static` as `passed_not_applicable` without
  relabeling an imported source;
- `contact`, `supports`, `steady-state`, `structure-unstable`,
  `excess_penetration`, and
  `unsupported_stationary_or_floating_body` meet the declaration's positive,
  negative, boundary, identity-span, and unavailable/invalidation floors;
- every central stratum and supported terminal reason has actual Unity evidence;
- synchronized observations, access separation, instance-held-out partitioning,
  deterministic replay, and the sealed final-evaluation boundary revalidate;
- the source-bound invalidation audit rejects and atomically quarantines the
  entire mutated attempt without admitting it as capability evidence; and
- planned, accepted, rejected, failed, quarantined, retried, unavailable, unmet,
  and systematic-defect categories remain explicit even when a category is
  empty.

Secondary, optional, and out-of-scope capabilities are recorded as not required.
They are never fabricated as available or emitted as false central targets.

## Publication and validation

Canonical publication requires the issue-51 implementation to be committed and
the tracked worktree to be clean. The runner refuses an existing runtime or
output directory and publishes through a temporary directory followed by one
atomic rename. Validation exactly re-derives the report and accounting, reruns
the public issue #46–#50 validators, parses every primary v2 capture, and checks
bundle membership.

The non-launching check is:

```sh
python -u -m scripts.capture_issue_51_evidence --dry-run
```

The full progress-reporting command is:

```sh
python -u -m scripts.capture_issue_51_evidence \
  --runtime-root .local-artifacts/issue-51-pilot-run-determination-2 \
  --prior-runtime-root .local-artifacts/issue-51-pilot-run \
  --output data/runtime_evidence/issue-51
```

Both commands print phase changes immediately. The full command also prints the
collector output and a heartbeat every 15 seconds while a child process is
running.
