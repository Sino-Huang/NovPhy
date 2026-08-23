# Cohort-v2 downstream ingestion v1

This contract defines the public model-facing reader and acceptance evidence for
GitHub issue #54. Its only authoritative release input is the immutable
cohort-v2 publication v5 produced by issue #53. The suffix `v5` versions that
production execution and release schema; the scientific cohort remains
`cohort-v2`.

## Public consumer boundary

`world_model.data.CohortV2ReleaseReader` admits exactly one non-final exposure
role and one permission allowed by the real partition manifest. It validates
the publication, release, collection plan, partition, attempt accounting,
primary rollout inventories, physics captures, observation traces, and all
three source-bound derivation kinds before exposing a rollout or derived
example. The released collection and parameter plans must equal the validated
immutable plan-v5 authority, so an action cannot change while retaining its
declared identity. Ordinary readers cannot select `final_evaluation`.

The reader exposes every fixed-step engine record, event, terminal record,
intervention representation, source identity, and exact central label payload.
The separately synchronized agent observation retains its own pre-intervention
fixed step; it is not relabeled as a physics-trace frame record. Canonical observation
bytes remain available only through the existing diagnostic access policy and
are rejected as model input.

`CohortV2OracleWindowDataset` derives one observation-backed oracle-symbol
window per admitted rollout without writing to or changing the primary
release. `build_cohort_v2_oracle_window_loader` passes those windows through
the public training loader, and `score_cohort_v2_endpoints` consumes all six
parts of the complete endpoint tuple through the public evaluation interface.
Unavailable values are excluded from denominators. A missing or malformed
prediction is rejected before metrics are returned.

## Final-evaluation probe

`probe_cohort_v2_final_access` validates the frozen authorized workflow, its
audit record, the sealed-bundle boundary, and the read-only release validation.
It returns only an access receipt. It never exposes sealed rollouts, examples,
labels, observations, or metrics to an ordinary reader.

## Fail-closed behavior

The reader rejects malformed envelopes, missing or promoted capabilities,
stale or cross-release bindings, role or permission crossings, incomplete
inventories, temporal or identity mismatches, unavailable-to-value conversion,
unauthorized canonical access, and ordinary final-evaluation access before an
example or metric is returned. The six accepted central labels are exactly
`contact`, `supports`, `steady-state`, `structure-unstable`,
`excess_penetration`, and `unsupported_stationary_or_floating_body`.

## Evidence command

The operator acceptance command is:

```bash
python -u -m scripts.build_issue_54_evidence \
  --release-root data/runtime_evidence/issue-53-mixed-termination-v5 \
  --sealed-root .local-artifacts/issue-53-mixed-termination-final-release-v5 \
  --output data/runtime_evidence/issue-54
```

Exact read-only revalidation uses the same arguments plus `--validate`.
