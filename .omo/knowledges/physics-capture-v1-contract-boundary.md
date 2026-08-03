# Physics Capture v1 Contract Boundary

Recorded 2026-07-31 for `world-model-physics-instrumentation` Todo 1.

- Canonical schema: `docs/data_contracts/physics_capture_v1.schema.json` (JSON Schema Draft 2020-12).
- Normative temporal/support/event rules: `docs/data_contracts/physics_capture_v1.md`.
- Golden sidecars: `tests/fixtures/physics_capture_v1/physics_state.jsonl` and `physics_events.jsonl`.
- Public strict loader: `scripts.physics_capture_contract.load_physics_capture(state_path, event_path)`.
- `physics_capture_v1` remains unsupported by the canonical episode validator until Todo 5 atomically persists and validates enriched shot artifacts. Do not remove that reservation early.
- Every JSONL record repeats capture/shot identity, sidecar-local sequence, render and fixed clocks, and the exact coordinate/unit declaration.
- The only exact image/state claim is same synchronized-endpoint response plus equal Unity `render_frame`; desktop captures are not exact.
- Support cannot be recomputed from snapshots without retained contacts. It requires two cited consecutive fixed-step contacts and the frozen `support_v1` normal/vertical rules.
- Parser files are deliberately dependency-free. `scripts/physics_capture_parsing.py` is 247 pure lines; split it before adding Todo 5 behavior.

Verification command:

```bash
/home/sukai/miniconda3/bin/python -m unittest tests.test_physics_capture_contract -v
```

## Unity Snapshot Boundary

Recorded 2026-08-01 for `world-model-physics-instrumentation` Todo 2.

- Preserve legacy request 62 by keeping `SymbolicGameState.GetGTJson()` unchanged; enrich a parsed copy in a separate exporter.
- Dynamic IDs are `<unity-instance-id>:<reuse-ordinal>`, with ordinals scoped per numeric Unity instance ID and reset per level. Static colliders retain `world:static:<collider-id>`.
- The fixed-step counter must be incremented by an actual `FixedUpdate` callback and paired with the observed `Time.fixedTime`; do not derive a count from division.
- Read current `Rigidbody2D.velocity`, `mass`, and `angularVelocity` at capture time. `ABGameObject.lastVelocity` is historical damage state and is never a current-state source.
- Todo 2's intermediate snapshot has no RGB claim. Only Todo 4's same-response endpoint may add `rgb_frame.source=synchronized_endpoint`.
- `ABGameWorld.CapturePhysicalSnapshot` is the Unity production handoff for later protocol work. It yields to end-of-render and returns the intermediate model through a callback without changing legacy request dispatch.
- Todo 2 acceptance requires the exact project-pinned Unity editor `2019.3.4f1 (4f139db2fdbd)`; no local binary, Hub install, package/snap install, array-mounted install, or cached Docker image was available as of 2026-08-01.
- Official editor cache: `/tmp/opencode/Unity-2019.3.4f1.tar.xz`, size `1555657836`, ETag `cbfaf57ab22561677cfe35fdc1eb45fd`, SHA-256 `11687f1ada2826c363991c01c6703fe56384657ef3349e1194dee5f941949ca8`; extracted binary: `/tmp/opencode/unity-2019.3.4f1/Editor/Unity`.
- On Ubuntu 24.04 this editor requires legacy `libgconf-2.so.4`. Without that dependency, the loader exits before version logging, licensing, project compilation, or EditMode execution.
- Private legacy runtime cache: `/tmp/opencode/unity-2019.3.4f1-libs/root`. Use child-only `LD_LIBRARY_PATH=/tmp/opencode/unity-2019.3.4f1-libs/root/usr/lib/x86_64-linux-gnu:/tmp/opencode/unity-2019.3.4f1-libs/root/usr/lib` and prepend `/tmp/opencode/unity-2019.3.4f1-libs/root/usr/share` to child-only `XDG_DATA_DIRS`.
- The pinned editor links completely with that private runtime and reaches licensing. This machine has no valid existing Unity license for the batch gate, so Todo 2 remains unverified and uncommitted until user-managed licensing exists.
- Non-licensed static gate: bundled Mono `5.11.0` plus Roslyn `2.9.1.65535` compile the default production assembly and separate Todo 2 Editor tests from `/tmp/opencode/unity-2019.3.4f1-todo2-{production,tests}.rsp`. Unity's matching NUnit is inside `com.unity.ext.nunit-1.0.0.tgz`, not Mono's deprecated NUnit 2.4 assembly.
- Static audit log `/tmp/opencode/unity-2019.3.4f1-todo2-compile.log` records both zero exits as `STATIC_COMPILE_PASS_RUNTIME_UNVERIFIED`; it cannot establish GameObject behavior, end-of-frame semantics, legacy JSON parity, or EditMode acceptance.
