# Decisions

## 2026-07-28 Resume at Todo 7
- Execute directly in the current worktree; no PR or ship mode was requested.
- Treat Todo 7 as HEAVY because it introduces a new checkpoint-bound policy/state module and deterministic resume contract.

## 2026-07-29 Todo 7 independent verification
- Canonicalize declared allowed sets for schedule identity, but preserve stage order because ordered half-open intervals are semantic.
- Bind catalog identity to immutable episode, shot, frame, action, provenance, split, and capture-contract facts; do not include mutable filesystem state or rescan the catalog.
- Define one-frame normalized start progress as `0.0`, while retaining the positive-horizon eligibility rule that produces no one-frame candidates.

## 2026-07-29 Todo 8 temporal ablations
- Freeze the legal choice sets as `fixed_short=(1,1)`, `fixed_long=(4,2)`, and temporal uniform/curriculum over `(1,1)`, `(2,1)`, and `(4,2)`; no symbolic or abstraction field exists.
- Group preset, local sampling seed, draw count, and immutable cost rule in `AblationRunConfig`; manifest construction separately requires a policy and its verified state.
- Define selected compute budget as base-window total plus predicted-frame total plus the explicit zero non-learned temporal-controller cost.
- Emit separate `sample_matched`/`sample_unmatched` and `compute_matched`/`compute_unmatched` labels rather than collapsing them into one comparison result.

## 2026-07-29 Todo 9 integration and active-root QA
- Keep Torch-backed package exports lazy so planner/type imports remain usable in the collection environment while established training imports remain unchanged.
- Validate independent episodes concurrently in catalog construction while preserving sorted result order and immutable snapshot semantics.
- Give the inspector a metadata-summary validator mode that retains containment, symlink, type, contiguity, and metadata checks but avoids frame-record allocation and per-frame accessibility syscalls; full catalogs retain the latter.

## 2026-07-30 Todo 9 independent-verification remediation
- Replace the ambiguous summary `EpisodeAccepted` outcome with `EpisodeSummary(canonical_acceptance_available=False)` and rename every episode/shot/rejection count in the inspector schema as summary-only.
- Report both no-plan composition dimensions as `{status: unavailable, counts: {}}`; never parse source provenance from episode names.
- Preserve `EpisodeCatalog` at its original import path while extracting cohesive private modules until every touched production module is at most 250 pure LOC.
