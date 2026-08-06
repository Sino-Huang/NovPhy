# F2 Unity request-70 socket re-audit

Verdict: **needs-fix**. Queue and in-flight admission are capped and silent pre-request clients time out, but response writes are synchronous and unbounded, and teardown does not close in-flight clients.

| ID | Adversary | Exact probe / inspection | Observable | Result |
|---|---|---|---|---|
| UQ-01 | malformed input | `timeout --signal=TERM --kill-after=5s 60s python -m unittest tests.test_webui_bridge.PhysicsCaptureV1Tests tests.test_webui_bridge.PhysicsCaptureV1MalformedEnvelopeTests -v` | 15 tests, `OK`, repeated twice | PASS for Python decoder |
| UQ-02 | cancel/resume | inspect `PhysicsCaptureProtocol.cs:344-390` | `finally` closes a normally-completing client, but `OnDestroy` closes only queued clients; no in-flight registry/cancellation | NEEDS-FIX |
| UQ-03 | stale state | fresh Unity `SilentInFlightClientsHoldCapacityUntilTimeoutThenReleaseIt` | timeout releases four reservations and subsequent request reaches `WaitForEndOfFrame` | PASS for tested pre-request timeout |
| UQ-04 | dirty worktree | `git status --short -- <three scoped Unity files>` | `AIBirdsConnection.cs` modified; protocol source/test untracked in governed worktree | OBSERVED; hashes unchanged by QA |
| UQ-05 | hung/long command | every executable probe wrapped in `timeout`; Unity command used `timeout --signal=TERM --kill-after=20s 180s` | test XML persisted; process exited 134 after test completion | BOUNDED, misleading exit disclosed |
| UQ-06 | flaky repeat | Python 15-test command twice; prior Unity XML twice plus one fresh XML | Python `15/15` twice; Unity silent-client `1/1` in all three XMLs | PASS for covered behavior |
| UQ-07 | misleading success | compare fresh XML with process exit | XML says `Passed`; shell status was 134 (`timeout: the monitored command dumped core`) | DISCLOSED; XML pass is not process success |
| UQ-08 | repeated interruption | inspect teardown ownership and run fresh bounded Unity shutdown | queued clients are closed, but in-flight clients are not tracked by `OnDestroy` | NEEDS-FIX |
| UQ-09 | oversized protocol | Python malformed suite; Unity bounds tests at `PhysicsCaptureProtocolTests.cs:166-199` | declared PNG/JSON/envelope bounds are covered | PASS for envelope construction/decoding |
| UQ-10 | non-reading response client | inspect `PhysicsCaptureProtocol.cs:344-370` and tests `:201-352` | synchronous `NetworkStream.Write` has no send timeout/cancellation; no test sends 70 then withholds reads | NEEDS-FIX |
| UQ-11 | prompt injection | N/A | No natural-language instruction or agent-input surface exists in this socket/protocol scope. | N/A |

Fresh Unity artifacts:

- `.omo/evidence/world-model-physics-instrumentation/f2-unity-socket-reaudit/unity-silent-inflight-fresh.xml`: `result="Passed" total="1" passed="1" failed="0"`, duration 1.2232996 seconds.
- `.omo/evidence/world-model-physics-instrumentation/f2-unity-socket-reaudit/unity-silent-inflight-fresh.log`: test callback completed; command exit was 134 during Unity shutdown.

Direct slop/overfit pass: no deletion-only or prose-pinning tests. The socket tests drive real loopback clients, but invoke private `Serve` and inspect private queues/listeners via reflection (`PhysicsCaptureProtocolTests.cs:217-219, 313-339`), so they mirror implementation and omit response backpressure. `PhysicsCaptureProtocol.cs` is 364 pure LOC and its test is over 250 pure LOC; this is a maintenance note, while the two resource-safety failures are blockers.
