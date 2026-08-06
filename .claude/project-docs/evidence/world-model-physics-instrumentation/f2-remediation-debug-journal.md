# Debug Journal - F2 remediation
Started: 2026-08-06T00:00:00+10:00
Goal: Repair bounded-memory parsing, root confinement, idempotent resume, and Unity pending-client capacity.

## Environment Snapshot
- Runtime: Python 3 collector/tests; Unity 2019.4.41f2 EditMode C# tests.
- Entry: focused unittest invocations and Unity `-runTests -testPlatform editmode`.
- Git HEAD: shared dirty worktree; preserve all pre-existing changes.
- References read: debugging methodology 00-setup, 02-investigate, 06-fix, 08-qa, 09-cleanup; runtimes/python.md.
- Evidence root: `.omo/evidence/world-model-physics-instrumentation/` (no active ulw-loop attempt).

## Hypotheses
1. [OPEN] Sidecar readers allocate the complete file before enforcing byte/line limits. Distinguishing evidence: oversized sidecar reaches `read_text`, `splitlines`, or `read_bytes`. If true, fix is: bounded streaming.
2. [OPEN] Artifact validation checks lexical relative paths but follows symlinks without verifying resolved containment. Distinguishing evidence: fixed-name sidecar symlink outside shot root validates. If true, fix is: resolved confinement.
3. [OPEN] Resume recovery recognizes invalid artifacts only and attempts to recreate valid completed shot directories. Distinguishing evidence: second collection of a valid shot raises/overwrites rather than skipping. If true, fix is: preflight reuse.
4. [OPEN] Unity admission accounts only queued clients, while accepted clients moved to `Serve` no longer consume capacity. Distinguishing evidence: silent connections can exceed `MaxPendingClients` before read timeout. If true, fix is: global reservation.
5. [CONFIRMED] The adapter accepts `clock`/`sleeper` but drops them, while persistence loops without a deadline. Distinguishing evidence: the direct 2 fps probe records request times `[0.0, 0.0]` and no sleeps. If true, fix is: deadline pacing.
6. [REFUTED] Bridge latency alone collapses the schedule. Distinguishing evidence: the zero-latency fake still bursts both calls at virtual time zero.
7. [REFUTED] `CaptureBounds` computes an incorrect frame count. Distinguishing evidence: it correctly resolves two frames; only their dispatch times are wrong.

## Failed Hypothesis Round Counter
- Round 1: all four hypotheses confirmed by failing-first regressions.

## Artifacts To Revert
- [x] Focused regression tests in scoped Python/Unity test files - retained as product tests.
- [x] Production edits in scoped Python/Unity files - retained as product fix.
- [x] Temporary runtime roots under `/tmp/novphy-f2-remediation-*` - none remain.
- [x] Unity processes and temporary test project state - no Editor/test processes or core files remain.
- [ ] `tests/test_collect_rollouts.py` - retain the deterministic request timestamp regression.
- [ ] `scripts/physics_rollout_persistence.py` and `scripts/collect_rollouts.py` - retain the minimal pacing fix.
- [ ] `/tmp/novphy-f2-pacing-*` - remove after QA.

## Findings
- RED Python: `f2-remediation-python-red.log` captured whole-file read, sidecar-symlink acceptance, and `ENOTEMPTY` on second collection.
- RED Unity: `f2-remediation-unity-red.xml` captured one failed silent in-flight capacity test.
- Root causes confirmed: unbounded `read_text`/`read_bytes`; missing direct-validator confinement; no valid-final reuse branch; queued-only Unity capacity accounting.
- Additional recovery edge: a regular-file `frames` artifact leaked `NotADirectoryError`; normalized to `PhysicsArtifactError` and locked by a quarantine regression.
- Final Python: `f2-remediation-python-final2-run1.log` and `run2.log` each report 35/35 passed.
- Final Unity: `f2-remediation-unity-final-run1.xml` and `run2.xml` each report 11/11 passed, including request 70 and silent-client timeout/reuse.
- Known environment: Unity writes passing XML, then exits 134 in pre-existing headless CEF shutdown (`PrintJobManager::Shutdown`).
- Full collector: 80/81 effective tests pass; the one dataset-backed test fails because both expected fixture shot directories are absent. The protected active cohort exists and was not used or mutated.
- Pacing RED manual probe: `f2-pacing-direct-probe.log` exits 1 with request times `[0.0, 0.0]`, expected `[0.0, 0.5]`, and no sleeper calls.
- Pacing regression RED: `f2-pacing-regression-red.log` fails because request timestamps are `[0.0, 0.0]` rather than `[0.0, 0.5]`.
- Pacing GREEN: the adapter forwards its injected clock/sleeper and persistence waits for absolute sample deadlines. `f2-pacing-regression-green.log` passes.
- Pacing manual QA: `f2-pacing-direct-green.log` reports request timestamps `[0.0, 0.5]`, sleep durations `[0.5]`, and `paced=true`.
- Affected regression: `f2-pacing-focused-final-1.log` and `-2.log` each report 54/54 passed.

## Final Fix
- Stream and hard-bound JSONL and checksum reads.
- Resolve and confine metadata, state, event, frame directory, and PNG files to the shot root; reject symlinks and malformed directory shapes.
- Reuse validated completed shots before reset/capture/promotion.
- Count Unity queued and in-flight clients with a reservation released from `Serve` cleanup.
- Pace request-70 samples against absolute monotonic deadlines and forward the existing clock/sleeper seam.
