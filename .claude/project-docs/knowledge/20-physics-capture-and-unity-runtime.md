# Physics Capture and Unity Runtime

## Contract Boundary

`physics_capture_v1` is defined by:

- `docs/data_contracts/physics_capture_v1.schema.json`
- `docs/data_contracts/physics_capture_v1.md`
- `scripts.physics_capture_contract.load_physics_capture(...)`
- golden sidecars under `tests/fixtures/physics_capture_v1/`

The canonical episode validator must reject this capture contract until enriched shot artifacts are atomically persisted and validated. Do not remove this reservation to make an incomplete cohort consumable.

Every record retains capture/shot identity, sidecar-local sequence, render and fixed clocks, and exact coordinate/unit declarations. Exact image/state synchronization requires equal Unity `render_frame` in a synchronized-endpoint response; desktop capture supports visual review but cannot prove exact synchronization. Support labels require two retained consecutive fixed-step contact samples and cannot be reconstructed from snapshots alone.

## Snapshot and Collision Invariants

- Support labels require two retained consecutive fixed-step samples for the same canonical pair; both samples must satisfy support predicates, and both contact IDs and fixed-step values must be retained.
- Preserve legacy request 62; enrich a parsed copy through a separate exporter.
- Dynamic IDs are `<unity-instance-id>:<reuse-ordinal>` where the ordinal is scoped to each numeric Unity instance ID and resets per level.
- Static colliders use `world:static:<collider-id>`.
- Advance fixed-step count from real `FixedUpdate` with observed `Time.fixedTime`.
- Read current `Rigidbody2D` velocity, mass, and angular velocity at capture time; do not substitute cached `lastVelocity`.
- Intermediate snapshots make no RGB claim.
- Return production capture through the intended callback without changing legacy dispatch.
- Validate collision samples atomically so malformed evidence cannot partially mutate recorder state.
- Valid collisions require non-empty/sorted/unique `contact_ids` and finite non-negative `relative_speed`.

## Project and Build Safety

Do not open the canonical Unity project for a read-only claim: imports can modify `Library`, logs, and settings. Use the migration worktree project for Unity 2019.4 work.

Never build directly into production. Build and verify in isolated stages. Publication to `sciencebirdsgames/physics-v1` is separately approved work.

Deterministic compilation requires committed `Assets/csc.rsp` with `-deterministic`, its Unity `.meta`, and `BuildOptions.NoUniqueIdentifier | BuildOptions.StrictMode`; the stable tar/gzip packager must preserve deterministic archive bytes. Roslyn determinism stabilizes managed PE checksum/MVID bytes, while `NoUniqueIdentifier` stabilizes Unity's generated player identity.

Any commit can change archive identity because provenance is packaged. After any re-pin candidate change, rebuild and re-smoke; never carry forward an old archive receipt.

Deterministic publication requires two isolated builds from one committed source, static verification of both, equality of archive/player/assembly/provenance digests, one bounded live smoke against an exact candidate, protected-root comparison, and cleanup evidence.

## Unity Test Rules

- Do not combine `-quit` with `-runTests`.
- Partition EditMode runs by test class.
- Treat NUnit XML as authoritative; process exit alone is insufficient.
- If compilation or editor failure occurs before NUnit discovery and no XML exists, classify it as pre-test/inconclusive, preserve the compiler log, and keep the DoneClaim fail-closed.
- In Unity 2019.4 C#, catch derived exceptions such as `ObjectDisposedException` before bases such as `InvalidOperationException` to avoid CS0160 before discovery.
- Distinguish a late shutdown crash after complete XML from failure before result publication.
- Run every required class; partial partitions can hide permanent failures.
- EditMode `AddComponent` does not automatically reproduce runtime `Awake`; tests that require runtime binding must invoke the real initialization path.

## Runtime and Evidence Safety

Use `NetworkStream.BeginWrite`, frame-polled `IAsyncResult.IsCompleted`, a fixed `Stopwatch` deadline, connection close on expiry, and `EndWrite` only after completion; synchronous main-thread writes are not acceptable. Enforce request-70 envelope limits while building and immediately before transmission without changing request-38/request-62 layouts.

Recorder timeout starts at the first authoritative fixed-time contact sample. Clamp backward/nonmonotonic elapsed time to non-negative. Finalization guards private and public mutation; set `TruncatedFinalization` before `finalized = true`, and reject all later event/failure mutation.

Use child-scoped compatibility libraries through `LD_LIBRARY_PATH`; do not install legacy libraries system-wide. `/tmp` compatibility caches are ephemeral and must be rechecked.

Redact Unity logs before retention because they can include licensing identity. Protected-root manifest timeout is inconclusive, never permission to hand-write a success result. Recorder/socket paths must avoid blocking async loops and preserve cleanup/error reporting.

Publication provenance binds archive bytes, receipt, smoke marker, staged hash, DoneClaim, and release decision as one digest chain.

The old Unity 2019.3 licensing constraint is historical. Current editor, license, package, and runtime state must be established from dated worktree evidence rather than copied into stable knowledge.
