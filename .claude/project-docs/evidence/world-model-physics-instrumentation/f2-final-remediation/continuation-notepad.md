# Ultrawork Notepad - Complete world-model physics instrumentation
Started: 2026-08-06T00:00:00+10:00 (continuation after context handoff)

## Plan (exhaustively detailed)
1. Wait for the four active remediation workers to terminate naturally.
2. Verify each lane produced a nonempty DoneClaim and cleanup receipt.
3. Inspect the combined diff and reconcile the accepted-marker provenance contract and any omitted bounds.
4. Run focused and full relevant verification, capturing exact outputs under `.omo/evidence`.
5. Commit verified source increments atomically using repository history conventions.
6. Build twice at the final source commit and require byte-identical archives.
7. Publish the exact verified archive and run real request 62/38/70 smoke.
8. Refresh canonical smoke/provenance evidence and rerun docs/collection verification.
9. Run F1-F4, five-lane review, and a three-hypothesis runtime debugging audit at the exact final SHA/archive.
10. Synchronize plans, ledger, notepads, and evidence, then request explicit user approval.

## Success criteria + QA scenarios
- HEAVY: cross-module runtime, filesystem transaction, packaging, and external player publication changes require happy, boundary/adversarial, regression, and real-surface evidence.
- Published happy path: exact staged player invocation must decode request 62, execute request 38, and return one request-70 state plus seven events with exactly one `bird_launched`; PASS only if the captured smoke JSON records all observables and protected roots are byte-unchanged.
- Boundary/adversarial: focused verifier, collector/persistence, packaging, bridge, docs, and Unity tests must replay their previously captured RED cases GREEN; PASS only with nonempty logs and zero exit status.
- Determinism/regression: two builds at one final source commit must have identical archive SHA-256 and legacy request 38/62 tests must pass; PASS only with captured hashes and test logs.
- Review: F1-F4 and the mandatory five-lane review plus runtime audit must approve the exact final source/archive; PASS only with unconditional verdict artifacts.
- Failing-first proofs already captured before implementation under `f2-final-remediation/{unity-runtime,verifier,persistence,package}` and `f2-b2-b3-reproduction`; do not replace or infer them.
- STOP: stop when every final scenario passes at one exact source/archive, all resources are cleaned, state is synchronized, and the user gives the explicit approval required by the plan.

## Now
Waiting for active remediation workers 317445-317448 to finish and emit terminal evidence.

## Todo
- Verify four terminal claims and cleanup receipts.
- Integrate combined diff and close accepted-marker/package/confinement gaps.
- Run verification, commit, rebuild/publish, live smoke, final reviews, synchronization, and approval gate.

## Findings
- No active `omo ulw-loop` attempt directory exists; evidence is governed under worktree `.omo/evidence`.
- Existing goal remains active for F2/F3/F4, final reviews, exact evidence, state synchronization, and explicit user approval.
- Active workers at resume: Unity 317445, verifier 317446, persistence 317447, package 317448.
- Bridge/docs claim and cleanup receipt are nonempty; four other claims are not present yet.
- Product files are delegated worker ownership; this orchestrator edits only `.omo` until integration is delegated.
- `scripts/prepare_rollout_dataset.py:150-175` still requires `status=passed`, top-level hashes, capture contract, and version fields, while `tests/test_prepare_rollout_dataset.py:721-798` encodes the documented `status=accepted` plus nested `provenance` shape; this is the bounded post-lane integration RED/GREEN.
- The authoritative plan requires F1-F4 to be rerun at the exact final provenance-repair commit/archive and explicit user approval after surfacing all four verdicts.
- The worktree plan mirror is stale: it still contains Unity 2019.3 scope and a checked F2; do not synchronize it until final source/archive hashes exist.
- Verifier lane complete: `verifier/DoneClaim.json` parses with `status=done`; evidence records 15 intended RED failures, two focused GREEN runs, 12 full-module tests, real staged verification, malformed-path and wrong-SHA rejection, Oracle pass, and `verifier/cleanup-receipt.json`.
- Packaging lane complete: `package/done-claim.json` parses with `status=complete`; initial 7 intended failures became 12 tests GREEN twice plus 6 focused probes, quality gates pass, and `package/cleanup-receipt.json` is nonempty.
- Packaging is intentionally fail-closed until integration tracks `scripts/9001-player-wrapper.sh`; its verified unchanged SHA-256 is `cff7f36268fd29831e0ab0ac6c952379b390c87bb7b1f152962b87322ec801d1`.

## Learnings
- The documented accepted marker uses `status=accepted` with nested `provenance`; collection previously expected `status=passed` and top-level hashes.
- Any new source commit invalidates the currently published archive binding and requires deterministic rebuild plus republication.
