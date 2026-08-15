# Session 3 plan — runtime re-pin gate

Wave `runtime-repin-gate-20260810`, third session, 2026-08-11.
Historical receipts (`task_plan.md`, `notes.md`, `handoff-next-session.md` Parts A/B) are **not** rewritten.

## Locked preconditions (verified before any edit)

| Fact | Required | Observed |
|---|---|---|
| Working tree | `/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4` | same |
| Branch | `physics-unity-2019.4` | same |
| HEAD | `6f25ced` *(docs(runtime-gate): record the still_blocked verdict…)* | `6f25ced4cfc43ca6d6205b916f1a867534f93a1e` |
| Unpushed commits | 4, nothing pushed | 4 |
| Tracked dirty paths | none | none |
| Staged pin | `429cac1d748bed…95f2de` | identical |
| Port 2004 | free | `/proc/net/tcp` 0, `/proc/net/tcp6` 0 |
| `scripts/__pycache__/` | absent | absent |
| `tests.test_smoke_physics_capture` | 75 OK | 75 OK (30.0 s) |
| `mutation_check.py` | 8/8 red, source restored | 8/8 red, restored sha `72f6a121…88b6f6` identical |

## Phase order

1. **Empty-evidence policy** at `PhysicsShotRecorder.cs:531` — decide and pin *before* any gameplay wiring.
2. **Gameplay wiring** — `RecordCollisionCallback` directly (not `base`) in `ABBirdBlack` and `ABBlock`.
3. **Code review** on the exact diff; remediate every blocker red/green.
4. **Phase 5** — two isolated builds, identical digests, or `still_blocked`.
5. **Phase 6** — exactly one bounded full live smoke.
6. **Conditional re-pin**, verdict, evidence, handoff.

## Phase 1 — the decision, and why

`PhysicalShotRecorder` (the base class the runtime actually holds; `PhysicsShotRecorder` is the sealed
test subclass) exposes four `RecordCollision` overloads. Only two are reachable from a Unity physics
callback:

```
MonoBehaviour.OnCollisionEnter2D
  -> PhysicalSnapshotRuntime.RecordCollisionCallback(Collision2D)
  -> PhysicalSnapshotRuntime.RecordCollision(Collision2D)            :114
  -> PhysicalShotRecorder.RecordCollision(…, PhysicalContactInput[], float)  :515   throws at :521, :532
  -> PhysicalShotRecorder.RecordCollision(…, IEnumerable<string>,   float)   :495   throws at :504, :506
```

The evidence-free overload at `:488` (and its `:551` delegate) has **no product caller** — verified by
`grep -rn RecordCollision Assets/Scripts/`; the only product call site is `ABGameObject.cs:127`.

**Decision.**

- The two *evidence-bearing* overloads reject by `Debug.LogError` + early return, leaving the recorder
  unmutated. Thrown from inside a physics callback, `ArgumentException` aborts the remainder of the
  handler: `ABBirdBlack`'s explosion never plays, no terminal event is reached, and the smoke's 30 s
  finalize deadline expires with the single non-retryable run consumed.
- The evidence-free overload at `:488` **keeps throwing**. It is unreachable from any callback and its
  throw is the API guard that makes "record a collision without evidence" unusable.

**This does not weaken the frozen contract.** In both the old and the new behaviour *no collision event
is emitted*, so `physics_capture_v1` sees nothing, `validate_physics_shot_artifact` is unaffected, and
the smoke's `require_collision` still rejects. The change is only in how the in-process rejection is
surfaced: a logged error instead of an exception that destroys the frame.

Two existing fixtures pin the old throwing behaviour and are updated in lockstep:
`PhysicsShotRecorderTests.CollisionPayloadRejectsMissingOrInvalidEvidence` and
`.CollisionContactSamplesRejectBeforeRecorderMutation`.

## Phase 2 — the wiring, and its scope boundary

`PhysicalSnapshotRuntime.RecordCollisionCallback(collision)` **directly**, at the top of:

- `ABBirdBlack.OnCollisionEnter2D` (`ABBirdBlack.cs:22`)
- `ABBlock.OnCollisionEnter2D` (`ABBlock.cs:145`), hoisted above the `tag == "Bird"` branch so both
  branches record.

Not `base`: `ABGameObject.OnCollisionEnter2D:125-141` also runs the damage model, so hoisting `base`
would change gameplay. The `else` branch keeps its `base` call; the recorder dedupes on
`fixedStep:first:second`, so the non-bird path still yields exactly one event — pinned by a test.

**Stop conditions.** If the fix requires changing the event model, the capture schema, the frozen
contract, or anything the Python consumer reads: stop, record the finding, report. Do not weaken the
contract to fit a payload.

### Fixture design

New EditMode class `GameplayCollisionRecordingTests`, added to `editmode_full_suite.py`'s `CLASSES`.

`Collision2D` and `ContactPoint2D` carry no public constructor for test use, so the fixture synthesizes
them by reflection over the engine's internal fields. The field names were read out of this exact
editor's `UnityEngine.Physics2DModule.dll` before writing the fixture
(`m_Collider, m_OtherCollider, m_Rigidbody, m_OtherRigidbody, m_RelativeVelocity, m_Enabled,
m_ContactCount, m_ReusedContacts, m_LegacyContacts` on `Collision2D`; `m_Point, m_Normal,
m_RelativeVelocity, m_Separation, m_NormalImpulse, m_TangentImpulse, m_Collider, m_OtherCollider,
m_Rigidbody, m_OtherRigidbody, m_Enabled` on `ContactPoint2D`). Every reflection lookup asserts
non-null, so a Unity version bump fails the fixture loudly instead of silently constructing an
unpopulated collision.

Assertion split, per `finding-simplejson-roundtrip-hides-json-types.json`: these fixtures assert the
**recorder API**, not serialized JSON, so the SimpleJSON re-quoting trap does not apply here.

## Phase 4/5/6 traps carried in

- `PYTHONDONTWRITEBYTECODE=1` on every `python` invocation, or remove `scripts/__pycache__/` before a
  build (`package_physics_player.py:105` aborts on untracked product source).
- Commit before building — `git_revision` refuses to package tracked drift from HEAD.
- New `.cs` files need a **committed** `.meta`, or Unity generates an untracked one and the packaging
  gate aborts.
- Run EditMode per test class (`editmode_full_suite.py`); a single unfiltered run crashes in
  `CefBrowserMessageLoop` before flushing XML.
- NUnit XML is the authority; the editor exit code is not.
- Absolute `/bin/` paths for evidence-bearing commands (`ps`→`procs`, `df`→`duf`, `ls`→`lsd` aliases).

## RED proof (before any product change)

`GameplayCollisionRecordingTests` (new, 4 tests) and `PhysicsShotRecorderTests` (20 tests) run per class
through `editmode_full_suite.run_class`, against the untouched product source.

| Test | Result | Failure reason |
|---|---|---|
| `BirdBlackImpactRecordsACollisionEventWithContractGradeEvidence` | RED | `expected exactly one recorded collision event · Expected: 1 · But was: 0` |
| `BirdOnBlockImpactRecordsACollisionEventWithContractGradeEvidence` | RED | same, 0 events |
| `BirdBlackImpactWithoutUsableContactEvidenceCompletesTheHandler` | RED | `Expected log did not appear: [Error] Regex: physics_capture_v1.*contact evidence` |
| `NonBirdImpactOnBlockRecordsExactlyOneCollisionEvent` | GREEN | expected — the `else` branch already reaches `base`; this fixture is the double-count regression guard for the hoist, not a red-then-green case |
| `PhysicsShotRecorderTests.CollisionPayloadRejectsMissingOrInvalidEvidence` | RED | `ArgumentException: Collision events require contact evidence.` at `PhysicsShotRecorder.cs:504` |
| `PhysicsShotRecorderTests.CollisionContactSamplesRejectBeforeRecorderMutation` | RED | `ArgumentException: Collision relative speed must be finite and non-negative.` at `PhysicsShotRecorder.cs:521` |

NUnit XML sha256 at RED: `GameplayCollisionRecordingTests` `c17b7124…c364422`, `PhysicsShotRecorderTests`
`4b1b3b55…83735607`. Editor `process_exit` was `-6` in every run (the known `CefBrowserMessageLoop`
shutdown signal); the XML is the authority.

## Phase 3 — code review and remediation

First review pass on the exact diff returned **Block: 1 BLOCKER, 3 MAJOR, 3 MINOR**. It confirmed the
three claims the change rests on: the dedupe collapses to exactly one event and cannot duplicate
`RawContacts`; fail-closed holds at the wire on every new log-and-return path; no caller depended on the
removed exception; and no contract or consumer drift, with C# `.Distinct().OrderBy(StringComparer.Ordinal)`
matching Python `sorted(set(...))`.

### B-1 (BLOCKER) — fixed, red then green

The collision path handed `UpdateSupport` a **single pair's** contacts, and `UpdateSupport`'s last line
prunes every edge whose pair is absent from the set it is given. Every real collision therefore erased the
support graph of every other pair. Not fail-closed — `support_edges` stayed schema-valid and the smoke
would still have passed, while the pinned player produced degraded ground truth for the whole cohort.
Recorded in `finding-collision-path-erased-support-edges.json`.

Fix: a private `RecordContacts(long, float, PhysicalContactInput[], bool isFullStepSample)`. Public
overloads pass `true`; the collision path passes `false` and skips `UpdateSupport`. Support derivation
stays owned by the full-set `FixedUpdate` sampler, which picks the pair up on its next sample. No event
kind, schema field, taxonomy entry or consumer-read field changed.

| Stage | `PhysicsShotRecorderTests` | NUnit XML sha256 |
|---|---|---|
| RED (test first, product untouched) | 21 total, 20 passed, **1 failed** — `CollisionAtAStepDoesNotErasePriorSupportEdges`: `Expected: 1 · But was: 0` | `e39ff74e…13a14c07` |
| GREEN (after the fix) | 21 total, 21 passed | `ebe03ca6…4f42036c` |

### Accepted MAJOR/MINOR

- **M-3, vacuous assertion — fixed.** With `_defense = 1e9f` and a null `_rigidBody`, the bird-on-block
  life assertion passed whether `ABBlock`'s formula or `base`'s ran. Now `_rigidBody` is bound by
  reflection and `_defense = 47f`, which sits between the two formulas at relativeSpeed 4.25: `ABBlock`
  deals 0.685, `ABGameObject` clamps to 0. **Verified discriminating by mutation**: forcing the block down
  the base path made it fail with `Expected: 9999.3154 · But was: 10000.0`; mutation reverted and the
  file re-verified byte-identical apart from the intended +12 lines.
- **m-1 — fixed.** `NonBirdImpactOnBlockRecordsExactlyOneCollisionEvent` now also asserts
  `RawContacts.Count == 2`, so a dedupe placed after ingestion would fail rather than pass.
- **M-2, m-3 — fixed.** Comment accuracy only: the recorder-test comment now names the two guards it
  actually covers, and the `ABBlock` comment records the load-bearing detail — the collision path checks
  its `fixedStep:first:second` key *before* it ingests contacts.

### Deliberately left out, with reasons

Recorded in `finding-unrecorded-collision-callbacks-out-of-scope.json`.

- **M-1, `ABEgg.cs:10-13`** carries the same unrecorded-collision defect. Authorization covers exactly two
  gameplay callbacks; ABEgg is unreachable on the smoke level; a third build-surface change buys the gate
  nothing. Must be fixed before any cohort that uses a white bird.
- **m-2, smoke harness log scan.** Editing `scripts/smoke_physics_capture.py` forces updating the pinned
  digest `72f6a121…88b6f6` in `mutation_check.py` and adds risk to a single non-retryable run. The smoke
  already fails closed on zero collisions, so the gap costs diagnostics only.

## GREEN proof (after the product change)

Full per-class suite: `python .claude/project-docs/evidence/runtime-repin-gate-20260810/editmode_full_suite.py`
→ **52 tests, 52 passed, 0 failed, 0 skipped, verdict `all_editmode_green`**, receipt written to
`editmode-full.json`. All nine classes green, including the four new gameplay fixtures and the two
lockstep-updated recorder fixtures. Every editor `process_exit` was `-6`; the NUnit XML is the authority.

**Re-run after remediation: 53 tests, 53 passed, 0 failed, 0 skipped, verdict `all_editmode_green`**
(one test added, `CollisionAtAStepDoesNotErasePriorSupportEdges`). Per-class NUnit XML sha256:

| Class | Tests | sha256 |
|---|---|---|
| `ABBirdLaunchTests` | 1 | `440810f2…9568adc8` |
| `ABGameWorldLifecycleTests` | 1 | `9cce2c34…4c4f4316a` |
| `GameplayCollisionRecordingTests` | 4 | `8d0744dc…6f6776d6a0ed4` |
| `LegacyGroundTruthTests` | 3 | `d9d20279…56dcd42b` |
| `PhysicalEntityRegistryTests` | 2 | `bdafaaad…fd09a8d35` |
| `PhysicalShotRecorderTests` | 5 | `8a4d2bf5…dc3fef02c` |
| `PhysicalSnapshotExporterTests` | 2 | `e4dc4006…3eab73cc738` |
| `PhysicsCaptureProtocolTests` | 14 | `4e494ced…08f0819b74` |
| `PhysicsShotRecorderTests` | 21 | `ebe03ca6…4f42036c` |

## Phase 3b — second review pass, and the stop condition it triggered

The second review pass on the exact diff returned: *"The diff itself is clean. All four remediations
hold, and none of the five hard constraints is violated by this change. Nothing in the diff blocks the
pin."* It then raised **F1** as a blocker on *spending* the smoke rather than on the diff — a
**pre-existing** defect the diff would expose for the first time.

### F1/F2 — confirmed, and deliberately not fixed

Recorded in `finding-sidecar-array-order-violates-contract.json`, proven by
`probe_raw_contact_order.py` → `probe-raw-contact-order.json`.

The emitter writes `raw_contacts` **cumulative and step-major** (sorted only within each fixed step),
while `scripts/physics_capture_parsing.py:281` requires the array globally sorted by a key that
**excludes `fixed_step`**. `support_edges` (`:283`) has the same defect class: appended in contact-pair
order while the contract sorts by `supporter_id`, which is whichever body is lower in y.

Confirmed two ways. Structurally, the whole chain was read: `RecordContacts:435` appends to a
`rawContacts` that is **never cleared**; `FixedUpdate:109` samples every collider every step;
`CreateFinalizedSnapshot:642` passes the cumulative field; `BuildContactsJson:148-155` iterates and
never sorts. Empirically, the probe feeds emitter-shaped arrays to the **production parser**
`_parse_state` and gets `deterministic_order at physics_state.jsonl:1: raw_contacts`, while the
identical set globally sorted parses; same separation for `support_edges`.

| Probe case | Result |
|---|---|
| `single_step_only` | parsed |
| `emitter_shaped_cumulative_3_steps` | **rejected** — `deterministic_order … raw_contacts` |
| `globally_sorted_same_set` | parsed |
| `f2_emitter_shaped_support_edges` | **rejected** — `deterministic_order … support_edges` |
| `f2_globally_sorted_support_edges` | parsed |

**Failure site corrected.** The reviewer estimated phase `capture-physics-rollout`. It is one phase
later — `validate-artifact`, `smoke_physics_capture.py:1151` — which is worse: `require_collision`
(`:1131`) has already passed, so the single non-retryable run is fully consumed before the failure
appears. `max_frames=1` does not help; the one persisted state record still carries every contact from
every fixed step.

**Not introduced here.** `git diff --name-only` confirms `PhysicsCaptureProtocol.cs` is untouched. No
smoke has ever passed `require_collision`, so no run has ever reached the parser — which is why the gate
never saw this.

**Stop condition invoked.** The brief: *"If the fix turns out to require changing the event model, the
capture schema, the frozen contract, or anything the Python consumer reads — STOP, record the finding,
and report back instead of improvising."* The order of these arrays **is** read and validated by the
Python consumer. The natural fix — sorting once in `CreateFinalizedSnapshot` so the producer conforms
to the already-frozen contract — is the opposite of weakening it, but it is still outside the two
authorized surfaces, so the call is not this wave's to make. Candidate fixes and the regression tests
they need are written into the finding.

**Consequence for the wave.** Phase 5 does not touch the parser and was completed. Phase 6 was
deliberately **not** spent: the run would have been consumed into a known-failing invariant, buying
nothing the probe has not already established at zero cost.

### Reviewer findings not acted on

F3 (collision evidence looked up by entity pair, ignoring collider identity — minor on single-collider
prefabs), F4 (the "never throw in a physics callback" property is argued, not enforced by a `try/catch`
in `RecordCollisionCallback`), F5 (`PhysicsShotRecorder.cs` is 814 lines, over the project's 800-line
rule), F6 (`currentStep`/`currentTime` write-only), F7 (the contact stream is unbounded in shot
duration; ~180k contacts at the 120 s ceiling would trip `RecordLimitExceeded`). None blocks the gate;
all are cheaper to fix in the same wave that fixes F1, since that wave already reopens this file.

## Error log

- **Fixture defect, caught and corrected before it could be mistaken for a product finding.** The first
  RED run failed `BirdBlackImpact…` with a `NullReferenceException` at `ABBirdBlack.cs:24`
  (`_trailParticles._shootParticles`) rather than on the event assertion. Cause: `_trailParticles` is
  created in `ABBird.Start` (`ABBird.cs:61`), not in `Awake`, so `AwakeComponent(bird)` never built it.
  `Start` also does `Resources.Load` and schedules an `Invoke`, neither usable in EditMode, so the
  fixture now sets that one field directly, exactly as `Start` sets it. Re-run confirmed both record
  fixtures then fail on `Expected: 1 · But was: 0` — the real reason.
- **Fixture defect during remediation.** The M-3 fix first tried to run only `ABGameObject`'s half of the
  initialisation chain via `AwakeComponent(block, typeof(ABGameObject))`. `Awake` is `protected virtual`,
  and `MethodBase.Invoke` dispatches **virtually**, so it re-entered `ABBlock.Awake` → `SetMaterial` →
  `NullReferenceException` on the unloadable sprite table. Two gameplay fixtures failed with
  `TargetInvocationException`. A base implementation cannot be invoked on its own this way; the fixture
  now binds the single field (`_rigidBody`) by reflection instead, matching how it already handles
  `_trailParticles`. The unused `AwakeComponent(MonoBehaviour, Type)` overload was removed.
