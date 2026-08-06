# F2 Unity Runtime Safety Remediation

- Unity 2019.4 request-response coroutines can avoid main-thread socket blocking with `NetworkStream.BeginWrite`, frame-by-frame `IAsyncResult.IsCompleted` polling, a fixed `Stopwatch` deadline, connection close on expiry, and `EndWrite` only after completion.
- Request-70 envelope limits remain enforced both while building the envelope and immediately before transmission; no request-38/request-62 or envelope layout code needs to change.
- Recorder timeout starts on the first authoritative fixed-time contact sample. `Mathf.Max(0f, fixedTime - shotStartFixedTime)` handles backward/nonmonotonic samples without creating negative elapsed time.
- Finalization must guard private helper mutation as well as public entry points. `TruncatedFinalization` is set before `finalized = true`, while all later event/failure helpers reject mutation.
- ABBird launch capture is testable without constructing `ABGameWorld`: a protected method assigns the `Rigidbody2D` velocity and records the assigned value, and an EditMode fixture invokes that seam with a real `PhysicalSnapshotRuntime` recorder.
- For Unity 2019.4 C#, derived exceptions must precede base exceptions in catch order. In this run, catching `ObjectDisposedException` after `InvalidOperationException` caused CS0160 before NUnit discovery; the catch order was corrected afterward.
- A bounded Unity run that exits before NUnit discovery must not be reported as a test failure or GREEN run. Preserve the compiler log, record that no results XML was produced, and keep the DoneClaim fail-closed until another invocation is explicitly authorized.
