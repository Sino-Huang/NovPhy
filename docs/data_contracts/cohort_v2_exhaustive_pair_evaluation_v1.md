# Cohort-v2 exhaustive pair evaluation v1

This contract is the issue-3 evaluation boundary for the immutable cohort-v2
release. It is separate from the legacy continuous-only temporal experiment.

## Declared grid and eligible states

The central grid explicitly declares requested horizons `(1, 5, 15)` and the
exclusive description modes `continuous`, `micro`, and `macro`, for nine
ordered pairs. The numeric horizons intentionally retain the proposal's
provisional short/medium/long fixed-step points, but the declaration has its own
`cohort-v2-pair-grid-v1` identity and is not imported from the legacy temporal
grid. A future horizon change therefore requires a new cohort grid version.

Every retained nonterminal frame record in the permitted training,
calibration, and model-selection rollouts is an eligible state. For every
requested horizon, its target is the record at
`min(context_position + requested_horizon, terminal_position)`. The requested
horizon remains the pair identity; the effective horizon records terminal
clamping and never replaces it.

`CohortV2OracleWindowDataset` requires callers to provide their declared
horizons and enumerates every eligible state. Its `ingestion_smoke` constructor
retains issue #54's historical one one-step window per rollout path without
changing the immutable issue-54 evidence.

## Capability-gated outcomes

Each pair requires the matching checkpoint transition capability. Micro pairs
also require available `contact` and `supports` context and target labels.
Macro pairs require available `steady-state` and `structure-unstable` context
and target labels. Continuous pairs require no symbolic label. A missing checkpoint
capability or unavailable required symbolic label emits a typed unavailable
outcome with the exact capability reason. The evaluator does not invoke the
scorer and does not emit an objective for that pair.

The grid provenance also names the complete endpoint capability pair,
`excess_penetration` and
`unsupported_stationary_or_floating_body`. These labels remain available or
unavailable exactly as published by the release; physical-plausibility scoring
is outside this evaluator and remains assigned to issue #7.

Available finite objectives are selected with fixed numeric tie tolerances.
Ties are ordered by requested horizon and then `continuous`, `micro`, `macro`.
If no pair is available, both the selected pair and tie set are empty; no score
or selection is fabricated.

## Artifact and provenance

The canonical manifest binds the release, partition, capability declaration,
checkpoint identity and declared transition capabilities, objective, grid,
exposure roles, exact nonterminal state-set identity, and all counts. One
SHA-256 identity covers the complete canonical state-record byte stream and is
part of the evaluation identity.
Every state record contains all nine ordered outcomes, target identity,
requested and effective horizons, unavailability reasons, and deterministic
selection metadata. Read-only validation reloads the three public role readers
and recomputes state membership, targets, effective horizons, capability-based
unavailability reasons, coverage, counts, pair order, and ties. It validates
stored objective values but does not invoke model objectives.

The current issue-3 acceptance run is a capability audit because transition
checkpoints for the three modes are delivered by later model tickets. It
therefore enumerates the complete real public state/pair surface while marking
every missing checkpoint capability unavailable. It is evidence about the
evaluation boundary, not model quality.

Dry-run the real release without writing files:

```bash
python -u -m scripts.run_cohort_v2_pair_evaluation --dry-run
```

Materialize and validate the bounded capability-audit artifact with visible
progress logs:

```bash
python -u -m scripts.run_cohort_v2_pair_evaluation
python -u -m scripts.run_cohort_v2_pair_evaluation --validate
```

After committing an implementation, emit a compact source-bound evidence
summary without committing the raw JSONL artifact:

```bash
python -u -m scripts.run_cohort_v2_pair_evaluation \
  --compact-report data/runtime_evidence/issue-3/cohort-v2-exhaustive-pair-evaluation-summary.json \
  --implementation-commit <commit>
```

The default output is local under
`.local-artifacts/issue-3-capability-audit`; primary cohort artifacts are never
modified.
