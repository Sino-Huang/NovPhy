# Task Plan: BG-NS-JEPA Milestone 0a Reconciliation

## Goal
Reconcile the BG-NS-JEPA roadmap with current accepted evidence and produce an implementation-ready Milestone 0a specification for deterministic macro-event and outcome-label derivation from frozen `physics_capture_v1` sidecars, without publishing a runtime, collecting a cohort, or implementing the derivation.

## Session Scope
- [x] Verify branch, HEAD, status, archive/smoke provenance, and authority ordering.
- [x] Inventory exact frozen state/event fields and observed sidecar edge cases.
- [x] Resolve deterministic label vocabulary, temporal semantics, artifact versioning, identity/alignment, validation, fixtures, acceptance evidence, dependencies, and non-goals.
- [x] Correct stale high-level roadmap progress without overclaiming Milestone 1.
- [x] Write the Milestone 0a execution specification.
- [x] Run final document/evidence validation and record actual output.
- [x] Inspect Git state/diff/history, commit session-owned files, write and commit the handoff, push, and verify `HEAD == upstream`.
- [x] Obtain an independent critic review of the completed writing.

## Key Questions
1. Which exact `physics_capture_v1` fields deterministically support each proposed label?
2. Which requested semantics are engine-verified, and which require representative authorized enriched shots?
3. Can fixture-based schema implementation begin before runtime publication and cohort authorization, and what remains forbidden?
4. What is the exact next dependency-ready action after this planning session?

## Decisions Made
- The frozen `physics_capture_v1` schema will not be mutated or weakened.
- The consumed full smoke will not be rerun.
- Runtime publication and enriched-cohort collection remain outside this session absent explicit owner authorization.
- Historical evidence is subordinate to the latest handoff and machine-readable accepted evidence.

## Errors Encountered
- Three delegated explorer/verification lanes returned session errors without evidence. Known source paths were read directly instead; no task was retried unchanged.
- Three bounded specification-writer attempts stopped without creating their owned file. Empty/stuck lanes were reconciled or cancelled; the settled specification was then written directly.
- The first independent critic review rejected one authority-wording contradiction. The opening status was narrowed to match the already-settled fixture-only authority boundary; no semantic rule changed.

## Status
**Completed** — roadmap/specification, validation, critic correction and acceptance, Git commits, handoff, and published branch state are complete. The exact next action is the fixture-only implementation session in the handoff.
