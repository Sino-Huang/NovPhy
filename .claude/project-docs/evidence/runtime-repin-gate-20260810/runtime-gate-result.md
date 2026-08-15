# Runtime Gate Result

**Status: `still_blocked`** (current authority — third session, 2026-08-11)

The first and second session records are preserved verbatim below. Nothing in them was rewritten; what this session supersedes is named explicitly in §S3.9, and the second session's machine-readable verdict is preserved byte-identically at `runtime-gate-verdict.session-2.json` (sha256 `c2c7c11e…8510a54d`).

---

# Third session (2026-08-11) — current authority

**Status: `still_blocked`**

Wave `runtime-repin-gate-20260810`, third implementation session. **No smoke was spent. No retry, no re-pin, no publication, and no cohort collection occurred.**

The mission was: *"Make a bird-reachable collision recordable on the level the smoke plays, prove the rebuilt player deterministic, and spend exactly one bounded full live smoke against it."* The first two were achieved. The third was deliberately withheld, because a pre-existing defect discovered mid-session makes the smoke's later phase a known-failing invariant against a non-retryable run budget.

## S3.1 First failed invariant

> `physics_capture_v1` requires `raw_contacts` globally sorted by `(entity_a_id, entity_b_id, collider_a_id, collider_b_id, point.x, point.y, contact_id)` — a key that **excludes `fixed_step`** (`scripts/physics_capture_parsing.py:281`).

The Unity emitter does not satisfy it. It sorts `raw_contacts` only **within** each fixed step and then concatenates steps into a list that is **never cleared**:

- `PhysicsShotRecorder.RecordContacts:406-435` builds `stepContacts`, sorts with `CompareContacts:751-763` (the parser's key minus `contact_id`), then `rawContacts.AddRange(stepContacts)`.
- `rawContacts` has no reset anywhere in the file.
- `PhysicalSnapshotRuntime.FixedUpdate:104-112` calls `RecordUnityContacts` every fixed step over `FindObjectsOfType<Collider2D>()`, so every resting pair contributes contacts at every step.
- `CreateFinalizedSnapshot:642` passes the cumulative field; `PhysicsCaptureProtocol.BuildContactsJson:148-155` iterates in list order and never sorts.

**Phase and command site.** `validate-artifact`, at `scripts/smoke_physics_capture.py:1151` → `validate_physics_shot_artifact` → `scripts/physics_artifact_validation.py:196` → `parse_physics_sidecars` → `_parse_state`. Expected error: `deterministic_order at physics_state.jsonl:1: raw_contacts`.

**Why that placement is terminal for this wave.** `require_collision` (`:1131`) passes *before* this check. The single non-retryable run would be fully consumed before the failure appeared. `max_frames=1` does not mitigate it: the one persisted state record still carries every contact from every fixed step, because the snapshot is the cumulative finalized batch.

**Proved without spending the run.** `probe_raw_contact_order.py` reconstructs the emitter's output shape and feeds it to the **production parser** — it does not re-implement the ordering predicate. Exit 0, `f1_confirmed` and `f2_confirmed`:

| Probe case | Result |
|---|---|
| `single_step_only` | parsed |
| `emitter_shaped_cumulative_3_steps` | **rejected** — `deterministic_order … raw_contacts` |
| `globally_sorted_same_set` | parsed |
| `f2_emitter_shaped_support_edges` | **rejected** — `deterministic_order … support_edges` |
| `f2_globally_sorted_support_edges` | parsed |

**F2, the same defect class.** `support_edges` (`:283`) is appended in contact-pair order while the contract sorts by `supporter_id`, which is whichever body is lower in y. It is pruned each step rather than cumulative, so violation is geometry-dependent rather than guaranteed — but it is not an invariant either way.

**Not introduced here.** `git diff --name-only` confirms `PhysicsCaptureProtocol.cs` is untouched. No smoke has ever passed `require_collision`, so no run has ever reached the Python parser — which is precisely why the gate never saw this, and why *this* session's wiring is what would expose it.

**Reviewer estimate corrected.** The code reviewer placed the failure in phase `capture-physics-rollout`. It is one phase later, which is strictly worse for the run budget.

Evidence: `finding-sidecar-array-order-violates-contract.json`, `probe-raw-contact-order.json`.

## S3.2 Why it was recorded rather than fixed

The mission's TODO-2 scope check, verbatim: *"If the fix turns out to require changing the event model, the capture schema, the frozen contract, or anything the Python consumer reads — STOP, record the finding, and report back instead of improvising."*

The order of these two arrays **is** read and validated by the Python consumer. Changing the order the emitter writes changes exactly that.

To be explicit about what this is not: the natural fix — sorting once at finalization so the producer conforms to the already-frozen contract — is the *opposite* of weakening it. It is out of scope on surface grounds, not on merit. The candidate fixes and the regression tests they need are written into the finding for the wave that owns that surface.

## S3.3 TODO-1 — empty-evidence policy, decided and pinned

`ArgumentException` thrown inside a Unity physics callback aborts the remainder of the handler: `ABBirdBlack`'s explosion never plays, no terminal event is reached, and the smoke's 30 s finalize deadline expires with the single run consumed.

**Decision.** The two *evidence-bearing* overloads now reject by `Debug.LogError` + early return, leaving the recorder unmutated. The evidence-free overload at `:488` **keeps throwing** — it has no product caller (verified by `grep -rn RecordCollision Assets/Scripts/`; the only product call site is `ABGameObject.cs:127`), and its throw is the API guard that makes "record a collision without evidence" unusable.

**This does not weaken the contract.** In both the old and the new behaviour *no collision event is emitted*, so `physics_capture_v1` sees nothing and `require_collision` still rejects. Only the in-process surfacing changed. Two existing fixtures were updated in lockstep and are the pin.

## S3.4 TODO-2 — gameplay wiring, red then green

`PhysicalSnapshotRuntime.RecordCollisionCallback(collision)` called **directly** (not `base`) at the top of `ABBirdBlack.OnCollisionEnter2D` and at the top of `ABBlock.OnCollisionEnter2D`, hoisted above the `tag == "Bird"` branch so both branches record. Not `base`, because `ABGameObject.OnCollisionEnter2D:125-141` also runs the damage model. The `else` branch keeps its `base` call; the recorder dedupes on `fixedStep:first:second`, so the non-bird path still yields exactly one event — pinned by a test.

New EditMode class `GameplayCollisionRecordingTests` (4 tests), added to `editmode_full_suite.py`. `Collision2D`/`ContactPoint2D` are synthesized by reflection over the engine's internal fields, read out of this exact editor's `UnityEngine.Physics2DModule.dll`; every lookup asserts non-null, so a Unity bump fails loudly rather than silently constructing an unpopulated collision.

| Test | RED (before any product change) |
|---|---|
| `BirdBlackImpactRecordsACollisionEventWithContractGradeEvidence` | `Expected: 1 · But was: 0` |
| `BirdOnBlockImpactRecordsACollisionEventWithContractGradeEvidence` | same, 0 events |
| `BirdBlackImpactWithoutUsableContactEvidenceCompletesTheHandler` | `Expected log did not appear: [Error] Regex: physics_capture_v1.*contact evidence` |
| `NonBirdImpactOnBlockRecordsExactlyOneCollisionEvent` | GREEN by design — the double-count regression guard for the hoist |
| `PhysicsShotRecorderTests.CollisionPayloadRejectsMissingOrInvalidEvidence` | `ArgumentException` at `:504` |
| `PhysicsShotRecorderTests.CollisionContactSamplesRejectBeforeRecorderMutation` | `ArgumentException` at `:521` |

**GREEN, full per-class suite: 9 classes, 53 tests, 53 passed, 0 failed, 0 skipped, verdict `all_editmode_green`** (`editmode-full.json`). Every editor `process_exit` was `-6` — the known `CefBrowserMessageLoop` shutdown signal, reached after the XML was flushed. The NUnit XML is the authority.

## S3.5 Code review — two passes, one blocker fixed

**B-1 (BLOCKER), fixed red→green.** The collision path handed `UpdateSupport` a **single pair's** contacts, and `UpdateSupport`'s last line prunes every edge whose pair is absent from the set it is given. Every real collision therefore erased the support graph of every other pair. **Not fail-closed** — `support_edges` stayed schema-valid and the smoke would still have passed, while the pinned player produced degraded ground truth for the whole cohort.

Fix: a private `RecordContacts(long, float, PhysicalContactInput[], bool isFullStepSample)`. Public overloads pass `true`; the collision path passes `false` and skips `UpdateSupport`. Support derivation stays owned by the full-set `FixedUpdate` sampler. No event kind, schema field, taxonomy entry or consumer-read field changed.

| Stage | `PhysicsShotRecorderTests` | NUnit XML sha256 |
|---|---|---|
| RED | 21 total, 20 passed, **1 failed** — `Expected: 1 · But was: 0` | `e39ff74e…13a14c07` |
| GREEN | 21 total, 21 passed | `ebe03ca6…4f42036c` |

Evidence: `finding-collision-path-erased-support-edges.json`.

**Accepted MAJOR/MINOR, all remediated.** M-3 (a vacuous life assertion — now discriminating, verified by mutation: forcing the base path failed with `Expected: 9999.3154 · But was: 10000.0`); m-1 (`RawContacts.Count == 2` added, so a dedupe placed after ingestion would fail rather than pass); M-2 and m-3 (comment accuracy).

**Second pass:** *"The diff itself is clean. All four remediations hold, and none of the five hard constraints is violated by this change. Nothing in the diff blocks the pin."* It then raised F1 as a blocker on **spending** the smoke, not on the diff.

**Not acted on, with reasons:** ABEgg M-1 (same defect, unreachable on this level, third build-surface change before a non-retryable run — must be fixed before any white-bird cohort); m-2 smoke log scan (would force a `mutation_check.py` digest update; costs diagnostics only); F3–F7 (collider-identity lookup, no `try/catch` enforcing the never-throw property, 814-line file over the 800-line rule, write-only `currentStep`/`currentTime`, unbounded contact stream). All are cheaper in the wave that fixes F1, since it already reopens this file.

## S3.6 Phase 5 — deterministic, passed

Two builds of the exact committed source into isolated non-production stages via `NOVPHY_PHYSICS_STAGE`; the driver refuses to build into `PRODUCTION` or `STAGED_PIN`. Both exited 0, no orphan package managers reaped.

**`deterministic: true`, `drift: []`, 151 provenance input files compared.** The only differing key between the two build records is `stage`, the isolated output path, which differs by design.

| Artifact | Identical digest |
|---|---|
| archive | `2bdd498a928204f5923ef84770b361b6ba31dfa5681867028870237cf048847e` |
| `9001-player.x86_64` | `d74bf3f869525a6731b992e30e3beb62da14484c16a6e1ad7a0c73c30ff976fa` |
| `9001_Data/Managed/Assembly-CSharp.dll` | `f3557f2bea8f8ee4a40b47d89a091e2e16c45d2245b7840e833fe14768f0108b` |
| `UnityPlayer.so` | `53b0b8d1d21031c097721b1bf10bf8cd23c34663f871d606e28bd276bd171c28` |
| `provenance.json` | `723260383068a209a39d74d4e37e019246db5c94e5be7ada38d7672e4e6745b8` |

Provenance identity: `git_head ad2822a92688ff6b9e52428eb24d2dc6537165ad`, `git_tree 231116add9189e06c28f4ab51b4d535766701448`, Unity `2019.4.41f2 (6b23d448b533)`. Excluded member: `unity-build.log` (wall-clock timestamps and temporary paths; published beside the archive, and the archive digest matched byte-for-byte regardless).

**Cross-wave sanity.** Against the second session's build, `UnityPlayer.so` and `9001-player.x86_64` are **byte-identical** and only `Assembly-CSharp.dll` moved (`5d83af30…d098e94` → `f3557f2b…f0108b`) — exactly what a C#-only diff should produce.

Receipts: `phase5-builds/determinism-receipt.json` (sha256 `ddcc1c92…3348b89b`), `phase5-builds/phase5-runs.json`, `phase5-builds.stdout`.

## S3.7 Phase 6 — deliberately not spent

The run would have been consumed into a known-failing invariant (§S3.1), buying nothing the probe has not already established at zero cost. Command that would have been run, and was not:

```
python scripts/smoke_physics_capture.py --stage <verified-candidate> --output-dir <run> --report <report>.json
```

Consequently the conditional re-pin authorization — *"overwrite the staged pin ONLY after the full smoke accepts against the rebuilt candidate"* — was never triggered.

## S3.8 Identity, cleanup and protected roots

| Fact | Value |
|---|---|
| Branch | `physics-unity-2019.4` |
| HEAD at session start | `6f25ced4cfc43ca6d6205b916f1a867534f93a1e` |
| HEAD at Phase 5 | `ad2822a92688ff6b9e52428eb24d2dc6537165ad` |
| Tracked drift at Phase 5 | 0 |
| Unpushed commits | 7 · nothing pushed |
| Staged pin, before and after | `429cac1d748bed417b917d2838dc203d090668977dc8e56f5bac9a80ea95f2de` — **unchanged** |
| Port 2004 | 0 in `/proc/net/tcp`, 0 in `/proc/net/tcp6` |
| Stranded processes | 0, bound by `/proc/<pid>/exe` |
| `scripts/__pycache__/` | absent |

Commits this session: `c455eb3` (`fix`, source), `15bfdac` (`test`), `ad2822a` (`docs`, evidence). HEAD moved because `package_physics_player.py`'s `git_revision` gate refuses to package tracked drift from HEAD.

**Process-identity note.** `pgrep -f 'Unity|9001-player'` reported one match — this session's own `zsh`, whose command line contains the worktree path `physics-unity-2019.4`. Bound by `/proc/<pid>/exe` against the editor, player and package-manager executables, the real count is **0**. Recorded because it is exactly the identity-inference trap the mission forbids.

**Protected roots, all verified untouched:** `.omo/`, `.claude/logs/`, `sciencebirdsgames/physics-v1/` (all `??`, unmodified); `.claude/project-docs/knowledge/` (6 untracked entries, **0 modified or deleted**, 21 tracked files unchanged); the F1–F4 review artifacts and `review-work.md` (`??`, unmodified). The main checkout's Unity project and the protected rollout data were not touched.

## S3.9 What this supersedes, and what did not happen

Superseded in the second-session record below:

- **S2.1's terminal blocker.** *"No object reachable by a bird shot records a collision at all"* is now **fixed at the source**, proved by `GameplayCollisionRecordingTests` red-then-green. The level geometry is unchanged; what changed is that the bird and the struck block now reach the recorder.
- **S2.2's out-of-scope ruling.** The two gameplay callbacks and the empty-evidence policy were authorized for this session and have been changed.
- **S2's Phase 5 digest set.** Superseded by §S3.6, which is for the new source.

Nothing else in the first or second session record is rewritten.

**Did not happen:** no smoke spent, no bounded run retried, **no re-pin**, **no publication**, **no cohort collection**, no Todo-8 health report, no F1–F4 / JEPA / SPSG / controller work, and no weakening of the frozen schema, taxonomy, or Python consumer.

## S3.10 Next step

1. Fix F1 by sorting once at `PhysicsShotRecorder.CreateFinalizedSnapshot:639-644` — `rawContacts` by `CompareContacts` extended with `ContactId`, `supportEdges` by `(SupporterEntityId, SupportedEntityId, SupportId)`. Preferred over sorting in `PhysicsCaptureProtocol`, which would put ordering policy in the serializer rather than in the recorder that owns the invariant. Per-step ordering and `PointIndex` assignment stay untouched, so `contact_id`s do not change.
2. Add the three regression tests named in the finding; `probe_raw_contact_order.py` is the Python one in executable form already.
3. Fold in F3–F7 and ABEgg in the same wave, since it already reopens `PhysicsShotRecorder.cs`.
4. Re-run Phase 5 — the F1 fix changes `Assembly-CSharp.dll`, so archive `2bdd498a…48847e` **cannot be reused**.
5. Then spend the single full smoke on the rebuilt candidate. Only then does the conditional re-pin authorization apply.

---

# Second session (2026-08-11) — superseded in part, preserved verbatim

**Status: `still_blocked`**

Wave `runtime-repin-gate-20260810`, second implementation session. Recorded from machine-produced evidence in this directory. **No smoke was spent. No retry, no re-pin, no publication, and no cohort collection occurred.**

## S2.1 First failed invariant

> Smoke acceptance requires *at least one genuine collision with non-empty sorted unique `contact_ids` and a finite non-negative `relative_speed`.*

On the level this smoke actually plays, **no object reachable by a bird shot records a collision at all**, so the criterion is unreachable regardless of parameters, and regardless of the emitter fix.

- Level: `tasks/task_template_designer/Assets/StreamingAssets/Levels/novelty_level_0/type2/Levels/3_9_6_1.xml`, selected by `scripts/build_physics_player.sh:47` rewriting `config.xml` so `ui_level 1` resolves to it.
- Bird is a `BirdBlack`; `ABBirdBlack.cs:22` overrides `OnCollisionEnter2D` and never calls base, so the recorder is never reached from the bird.
- All 8 Platforms carry zero `m_Script` (no `ABGameObject`); Ground is a bare `BoxCollider2D` (`SymbolicGameState.cs:428`).
- `ABBlock.cs:145` calls base only in the **non-bird** `else` branch, so a bird striking a block records nothing.
- Only Pigs record — and both are fully enclosed by vertical platform walls, so direct bird-on-pig contact is structurally unreachable from the slingshot at `x = -12`.
- Aim cannot fix it: both the old `[-50, 40]` and the handoff-verified `[-80, 7]` pulls saturate the drag clamp (`_dragRadius 1`, `ABBird.cs:233-234`), so only launch elevation changes (≈39.7° → ≈5.0°), not speed.

Phase: 6 (single bounded full live smoke). Command that would have been run, and was not:

```
python scripts/smoke_physics_capture.py --stage <verified-candidate> --output-dir <run> --report <report>.json
```

Evidence: `finding-smoke-level-geometry-risk.json` (every claim re-verified against the level XML, the prefabs and the C# sources after a code review overturned an earlier, wrong "risk is near zero" bound in the same file).

**The single permitted full smoke was deliberately not spent.** With no retry available, running it against a configuration proved unable to produce a recordable collision would consume the one run that the post-fix candidate needs.

## S2.2 Why this was not fixed inside the wave

The mission's scope rule for TODO-2 permits fixing the collision payload in the capture path and requires stopping when the fix reaches further. This one reaches further:

- It requires calling `PhysicalSnapshotRuntime.RecordCollisionCallback(collision)` from **two gameplay classes** — `ABBirdBlack.OnCollisionEnter2D`, and the bird branch of `ABBlock.OnCollisionEnter2D` — not from the recorder. A plain `base` call is not available: `ABGameObject.OnCollisionEnter2D:125-141` also runs the damage model, so hoisting base would change gameplay.
- It interacts with a deliberate fail-closed `ArgumentException` at `PhysicsShotRecorder.cs:531`. Thrown inside a Unity physics callback it aborts the remainder of the handler → the `BirdBlack` explosion never plays → no terminal event → the smoke's 30 s finalize deadline expires with the single run consumed.

Decision: record the finding and report, per the mission's explicit instruction, rather than edit two gameplay classes and a fail-closed throw path against a non-retryable run budget.

## S2.3 TODO-2 scope check and outcome

The empty payload was diagnosed to a specific liveness defect, and it is **in scope and already fixed on this branch**:

- Root cause at staged commit `e2d19ae`: `PhysicalSnapshotRuntime.RecordCollision(Collision2D)` selected `contactIds` from `shotRecorder.RawContacts` filtered on `contact.FixedStep == Clock.FixedStep`. `OnCollisionEnter2D` fires *inside* the physics step, before `FixedUpdate` calls `RecordUnityContacts` for that step, so the store held no rows and the query returned empty. `PhysicsCaptureProtocol.AppendPayload` emits the `contact_ids`/`relative_speed` pair only when `ContactIds.Count > 0`, so the wire value became `{}`.
- No schema, event-model, or Python-consumer change was needed, and the frozen contract was not weakened.
- The fix is already committed ahead of the staged build: `7a2dd02` (build contacts directly from `collision.contacts`) and `97c4dd6` (reject empty evidence and non-finite/negative speed atomically; dedupe and sort ids). **The staged binary is stale relative to its own source branch.**
- Evidence: `finding-collision-payload-root-cause.json`, which also records and corrects an earlier mis-attribution to the four-argument overload.

**Test gap this session closed.** Recorder fixtures asserted the recorder API and protocol fixtures asserted the envelope, but nothing asserted the *serialized collision payload* — the exact seam where an empty list silently becomes `{}` on the wire. Added `PhysicsCaptureProtocolTests.Request70CollisionPayloadCarriesTheContactEvidenceTheContractRequires`, written before any emitter change and proved red first.

- RED against the mutation that reproduces `e2d19ae`'s emitter (`contactIds: new string[0], relativeSpeed: 0f`): fixture **Failed**, 13/14.
- Source restored byte-identically to `0987580887fb07603d1b4174effbdfc83c6cfef8b8554d4d8eaed7934cc879d6`.
- GREEN at HEAD: fixture **Passed**, 14/14.
- Verdict `red_then_green` in `collision-payload-redgreen.json`; harness `collision_payload_redgreen.py`; NUnit XML under `collision-payload-redgreen/`.

**A second defect was found inside the fixture itself and closed** (`finding-simplejson-roundtrip-hides-json-types.json`): a review-requested JSON-*type* assertion went RED against correct product output, because it ran against `collision["payload"].ToString()` and this SimpleJSON build stores every scalar as text and re-quotes it on `ToString`. The product (`PhysicsCaptureProtocol.cs:225`, `AppendFloat` at `:245-248`) was never wrong. The assertion now matches the raw serialized envelope, anchored on `contact_ids` and pinned to exactly one match. Durable rule recorded: assert JSON *structure* through SimpleJSON and JSON *scalar types* through the raw serialized string; never mix the two.

## S2.4 Phase 5 — completed and **PASSED**

Phase 5 does not depend on the blocker, so it was run to completion. The exact committed source was built twice into isolated non-production stages via `NOVPHY_PHYSICS_STAGE`. Both builds exited 0 with no orphaned `UnityPackageManager` daemons reaped.

`determinism-receipt.json` → `"deterministic": true`, `"drift": []`, `"drift_count": 0`, **151 provenance files compared**, schema `novphy_deterministic_build_comparison_v1`.

Identical across `build-a` and `build-b`:

| Artifact | sha256 |
|---|---|
| archive `novphy-physics-player-2019.4.41f2.tar.gz` | `d4e55bc4f684ecd4699c81d0c039ab43ab62c70ccf1a5b42d2455e0732147562` |
| `9001-player.x86_64` | `d74bf3f869525a6731b992e30e3beb62da14484c16a6e1ad7a0c73c30ff976fa` |
| `9001_Data/Managed/Assembly-CSharp.dll` | `5d83af307f293fc1cd374cf3e7489dc4cc7f9db3a4b4a184c508c44cfd098e94` |
| `UnityPlayer.so` | `53b0b8d1d21031c097721b1bf10bf8cd23c34663f871d606e28bd276bd171c28` |
| `provenance.json` | `1dc3097ef98640e9db13261cfebbbde417066c0e87ae0eceaa06820501590136` |
| package input `manifest.json` | `05677cc3199d5fff4aac54096877e795518487918e53810f477a228e5d1e28fb` |
| package input `packages-lock.json` | `3101c351984e6a73a1be7ad76d1a67c1b7638a6616554e50000b9672175ebe50` |
| Unity editor executable | `32252cb8eca087743e500596e093061a906203703915c2d3c2fb2f8a372bc150` |

Provenance project identity in both: `git_head 045296d6ed9f749d8ea12ca2e4b345d72e5dfce8`, `git_tree 40f30b59dd8fbf38fc2387d7c0aaf8481a146b89`.

**One member excluded from comparison, declared rather than dropped silently:** `unity-build.log` — Unity diagnostic output carrying wall-clock timestamps and temporary build paths; it is published beside the archive, not inside it, so it does not affect the archive digest, which matched byte-for-byte anyway.

**One abort and its cause, recorded rather than smoothed over.** `build-a` first failed at `package_physics_player.py:105` with `PackagingError: untracked product source: !! scripts/__pycache__/`. Cause was mine: this session's own `python -m unittest` runs wrote bytecode into the gate's untracked scope. `scripts/__pycache__/` held 48 `.pyc` files only, and `package_physics_player.py` imports stdlib exclusively, so removing it could not change packaging behaviour. Removed, re-run with `PYTHONDONTWRITEBYTECODE=1`, both builds exited 0. The directory is absent at the end of this session.

## S2.5 Source and candidate identity

**Source (HEAD moved during this session — see below):**
- Branch `physics-unity-2019.4`; HEAD `045296d6ed9f749d8ea12ca2e4b345d72e5dfce8`, tree `40f30b59dd8fbf38fc2387d7c0aaf8481a146b89`.
- Session start HEAD, as required by the mission: `7f1e8727782c985370a4c1b482292a20e6787918`.
- `scripts/smoke_physics_capture.py` `72f6a12183df97755ab715919557d70eb7cf5c59e9c8311dcab0c9925288b6f6`
- `tests/test_smoke_physics_capture.py` `3cd544c93d9b31d73e5026c4e87238b00dfb174acc9f75a30ae52a03135a91c0`
- `PhysicsCaptureProtocolTests.cs` `8ea5ab0b76b9bea1402121238bf57180c0ec344d97cfe80924bb03e41aa012f2`
- `PhysicsShotRecorder.cs` `0987580887fb07603d1b4174effbdfc83c6cfef8b8554d4d8eaed7934cc879d6`
- `PhysicsCaptureProtocol.cs` `38334eae168097a25e8606c4475ab79f8fea7a514ffd3315af731b05215bc514` (unmodified this session)

**Why HEAD moved.** `package_physics_player.py`'s `git_revision` gate refuses to package when tracked source drifts from HEAD, so Phase 5 cannot run against uncommitted work. The three commits are the wave's own required outputs:

| Commit | Subject |
|---|---|
| `a1067e5` | `docs(project-docs): describe the compressed knowledge index` |
| `87365fb` | `fix(smoke): use the handoff-verified release offset for the known action` |
| `045296d` | `test(physics-capture): pin the request-70 collision payload at the wire` |

**Candidate: unchanged.** The staged pin is still `429cac1d748bed417b917d2838dc203d090668977dc8e56f5bac9a80ea95f2de`. The Phase 5 archive `d4e55bc4…` exists only inside the two non-production evidence stages and was **not** staged, published, or smoked.

## S2.6 Listener, request identities, and collision

**Not obtained by this session.** No live run was launched, because the acceptance criterion in §S2.1 has no reachable success path against this configuration. The first session's listener binding stands as the last live observation and is preserved verbatim in §5 below. No handwritten substitute for a request or collision receipt exists and none was fabricated.

## S2.7 Tests, mutation, review

- Focused Python: `python -W error::ResourceWarning -m unittest tests.test_smoke_physics_capture` — **75 tests, OK** (74 in the first session; +1 from this session's smoke-source change).
- Mutation harness: **8 of 8 mutations turn the suite RED**; source restored `72f6a121…` *identical*. The pinned baseline in `mutation_check.py` was updated in lockstep with the smoke-source edit, as the mission requires — otherwise the harness refuses to run.
- Unity EditMode, full suite: **8 classes, 48 tests, 48 passed, 0 failed, 0 skipped**, verdict `all_editmode_green` (`editmode-full.json`, per-class NUnit XML under `editmode-full/`). Editor `2019.4.41f2-6b23d448b533`, sha256 `32252cb8…`.
- Red/green on the new fixture: `red_then_green` (§S2.3).
- **Authority note.** NUnit XML result attributes are the authority. Every EditMode invocation on this host exits with a signal inside CEF shutdown, reached strictly *after* the run finished and *after* the XML was flushed. No compiler or editor failure occurred before discovery. A single unfiltered run crashes in `CefBrowserMessageLoop` before flushing its result file — hence the per-class invocation recorded in `editmode-full.json`, and the harness deadlock finding in `finding-editmode-harness-deadlock.json`.
- Review: the code-reviewer was invoked on the exact diff. Its finding on the level's bird type is what overturned the earlier "risk is near zero" claim (§S2.1) and its finding on `AsFloat` coercion is what produced the fixture correction (§S2.3). Both were remediated with red/green evidence; the README hunk it flagged was split into its own commit `a1067e5`.

## S2.8 Cleanup and protected roots

- Port 2004: **free** (`0` listening sockets).
- Stranded processes: **none**. Bound by `/proc/<pid>/exe` resolution, not by name — the one name-substring match on this host is a `zellij` server whose socket path contains `unityhub`, which is not a Unity process. Match count by exe: **0**.
- `scripts/__pycache__/`: **absent**.
- Staged root `sciencebirdsgames/physics-v1/` **unchanged**: archive digest still `429cac1d748bed417b917d2838dc203d090668977dc8e56f5bac9a80ea95f2de`.
- Every pre-existing dirty and untracked path listed in the first session's §7 is preserved and uncommitted.

## S2.9 What this session changed in the record above

The first session's record is unaltered. Two of its statements are superseded by later evidence and are corrected here, not edited there:

1. §8 states *"Phase 5 … was not reached."* Phase 5 was reached and passed in this session — see §S2.4.
2. §1 attributes the blocker to the empty collision payload alone. That defect is real and already fixed on this branch (§S2.3); the *terminal* blocker is now the level-geometry finding in §S2.1, which the payload fix does not resolve.

## S2.10 What did not happen

No smoke was spent. No retry of any bounded run. No re-pin — `sciencebirdsgames/physics-v1/` was not overwritten and its archive digest is byte-identical to the session start. **No publication occurred.** No cohort collection, no Todo 8 health report, and no F1–F4 / JEPA / SPSG / controller work.

## S2.11 Smallest next step

1. Call `PhysicalSnapshotRuntime.RecordCollisionCallback(collision)` directly (not `base`) at the top of `ABBirdBlack.OnCollisionEnter2D`, and at the top of `ABBlock.OnCollisionEnter2D` hoisted out of the `else` branch so both branches record.
2. Decide the empty-evidence policy at `PhysicsShotRecorder.cs:531`: a `Debug.LogError` plus early return is equally fail-closed at the wire — no event is emitted and the smoke's `require_collision` still rejects — without aborting the physics handler.
3. Add EditMode fixtures proving a `BirdBlack` impact and a bird-on-block impact each produce a recorded collision event; red before, green after.
4. Re-run Phase 5, then spend the single full smoke on the rebuilt candidate. Only then does the conditional re-pin authorization apply.

---

---

# Superseded-in-part: first session record (verbatim, unaltered)

Retained exactly as written on 2026-08-11 by the first session. Corrections are recorded in §S2.9 above, not applied here.

**Status: `still_blocked`**

Wave: `runtime-repin-gate-20260810`. Recorded 2026-08-11 from machine-produced evidence in this directory. No re-pin, no publication, no cohort collection, and no retry of any bounded run occurred.

---

## 1. First failed invariant

> Acceptance requires at least one genuine collision with non-empty sorted unique `contact_ids` and finite non-negative `relative_speed`.

The staged player cannot produce it. Archive `429cac1d748bed417b917d2838dc203d090668977dc8e56f5bac9a80ea95f2de` emits `collision` events whose payload is the empty object `{}`, while the frozen capture contract requires `contact_ids` and `relative_speed`. `validate_physics_shot_artifact` therefore rejects every shot that contains a collision, and a shot with no collision fails the acceptance criterion directly. Both branches reject; there is no third.

- Phase: 6 (single bounded full live smoke).
- Command that would have been run, and was not: `python scripts/smoke_physics_capture.py --stage sciencebirdsgames/physics-v1 --output-dir <run> --report .claude/project-docs/evidence/world-model-physics-instrumentation/task-8-smoke.json`.
- Evidence: `finding-collision-payload-blocks-the-final-smoke.json`; upstream record `.claude/project-docs/evidence/enriched-cohort-oracle-labels/task-7-collection-blocker.json`.
- Fix location: player-side C# in the Unity project, reachable only through rebuild -> republish -> re-smoke, which is a re-pin this wave may not perform.

**The single permitted full smoke was deliberately not spent.** With no retry available, running it to re-confirm a documented defect would leave nothing for the run after the fix. That is the whole reason this verdict is `still_blocked` rather than `rejected_by_smoke`.

## 2. Second invariant, found and closed inside this wave

The gate's own mapped-code anchor was unsatisfiable. Unity 2019.4 never file-backs `Assembly-CSharp.dll`: Mono mmaps the class libraries, and Unity loads the user assembly through its own loader. The first live diagnostic rejected a correctly-identified player with `has 0 mapped Assembly-CSharp.dll images` — a gate defect, not a candidate defect (`finding-unity-does-not-map-the-user-assembly.json`, `player-mapping-probe.json`).

Corrected anchor: `UnityPlayer.so`, mapped from the candidate root, digested in `provenance.json`, and the native code that actually owns the socket. The assembly remains pinned by device, inode and digest read through `/proc/<pid>/root/`. This is one anchor stronger than the original chain, which had no digest over any mapped code.

**Deviation flagged:** the mission names "mapped `Assembly-CSharp.dll`" as a binding input. That input is not obtainable from this runtime. The substitute is recorded here rather than passed over in silence.

## 3. All captured diagnostics

| Artifact | What it establishes |
|---|---|
| `listener-diagnostic.json` | First live listener-only run, pre-correction. Correct process, rejected by the unsatisfiable anchor. |
| `listener-diagnostic-corrected.json` | Second live listener-only run, post-correction: `diagnostic_accepted` at `phase: listener-bound`. |
| `player-mapping-probe.json` | 91 file-backed regions in the live player, six framework DLLs, zero `Assembly-CSharp.dll`, nine mappings inside the verified clone. |
| `player-mapping-probe-wrong-port.json` | The run that drove a random port through `--physics-port` and saw no listener; the flag does not reach the player. |
| `finding-unity-does-not-map-the-user-assembly.json` | The mapped-anchor gate defect. |
| `finding-collision-payload-blocks-the-final-smoke.json` | The terminal candidate defect. |
| `phase-4b-unity-editmode.json`, `unity-editmode/` | Unity EditMode NUnit XML, treated as authority. |
| `review-remediation.json` | Review pass 6, its seven MAJORs, their fixes, and the mutation results. |
| `mutation_check.py` | The harness that proves each new assertion bites. |
| `notes.md`, `task_plan.md` | Working record, including the launch-topology and mapped-anchor corrections. |

Two prior diagnostic timeouts (exit 124) are explained and closed: an undersized 300 s ceiling against a measured 243.6 s protected receipt charged twice per pass, and an ordering defect in which `run_smoke` bound the physics port *before* connecting the agent. A bounded probe proved the jar does not own that port and does not spawn the player that does until an agent connects (60 s of polling: `player_procs=0 port2004=0` at every sample), so the bind had no reachable success path in either mode. Order is now connect -> configure -> bind, pinned by a call-order test.

## 4. Candidate and source identity

**Candidate (unchanged by this wave):**
- Staged archive: `429cac1d748bed417b917d2838dc203d090668977dc8e56f5bac9a80ea95f2de`
- `provenance.json` bytes: `3eae856a912c53fe6f1af41243551b504603bf7309e57d1c6c345a63a11bfcf9`
- `UnityPlayer.so`: `53b0b8d1d21031c097721b1bf10bf8cd23c34663f871d606e28bd276bd171c28`
- `Assembly-CSharp.dll`: `051aac49739a3b09628f03e9f0fc7c0ca5e17b42a66dc7f1707542fad8d73622`
- Unity source candidate: `97c4dd6f8b0a61b77b5b39a261ec0fddfd1abdba`

**Source:**
- Branch `physics-unity-2019.4`; recovery HEAD `fc3e34c5f6115b55751874df21c3a9286bd09a5e`, tree `8893df9ed6f455b984fefb096c95a906ce752e12`.
- `scripts/smoke_physics_capture.py` sha256 `f61ccfa52ef5bed5e156e8480f32f407c13286276df00d8ec676e9671fc84978`
- `tests/test_smoke_physics_capture.py` sha256 `d79e5a0bae96736a8f06fc0bb154e7d95470ef715445973287e0a26f10223823`
- Tooling commit: `b121136` (`feat(physics-smoke): bind the physics listener to process-backed identity`).

## 5. Listener and process binding actually achieved

From `listener-diagnostic-corrected.json` — the corrected gate against the live staged candidate, listener-only, no gameplay:

- Socket inode `209799923` on `0100007F:07D4`, the only listening socket on the port, fully attributed.
- Owner pid `700284`; exe `/tmp/novphy-physics-smoke-ei9aq7tc/player/9001.x86_64`; cwd the clone root.
- Mapped `UnityPlayer.so` `103:02` inode `69088604`, digest equal to the provenance runtime digest.
- `Assembly-CSharp.dll` via `/proc/700284/root/...`, `103:02` inode `69088102`, digest equal to the provenance assembly digest.
- Provenance bytes -> staged archive digest -> tree rooted at `launched_pid` 700231 (size 2), owner inside it.
- Diagnosis carried alongside the accept: `host_uninspectable_pid_count = 785`. The scan was partial by host policy; the invariant that closes the impostor hole is inode reconciliation, and it held.

**Request identities and collision: not obtained.** Both require the full smoke, which was not run. No handwritten substitute exists and none was fabricated.

## 6. Tests and review

- Focused: `python -W error::ResourceWarning -m unittest tests.test_smoke_physics_capture` — **74 tests, OK**.
- Adjacent physics tooling: `test_build_physics_player`, `test_package_physics_player`, `test_physics_capture_contract`, `test_verify_physics_player`, `test_verify_physics_capture_docs` — **60 tests, OK**.
- Unity EditMode: **47 tests across 8 classes, 0 failed, 0 skipped, 0 compiler errors** on editor `2019.4.41f2-6b23d448b533`. NUnit XML is the authority, not the process exit code: every invocation exits 134 from a SIGABRT inside `CefShutdown`, reached from `CallbacksDelegator:RunFinished -> EditorApplication:Exit` — strictly after the run completed and after the XML was written. No compiler or editor failure occurred before discovery.
- Review: six passes on the exact diff. Final pass 0 BLOCKER, 7 MAJOR, ~17 MINOR; all seven MAJORs remediated with tests verified red by mutation (`review-remediation.json`).
- Mutation: **8 of 8 mutations turn the suite red**, including the two that survived at review time (`_process_view` as identity, `_process_tree` as direct-children-only). Source restored byte-identical afterwards.

## 7. Cleanup and protected roots

- Last live run: `agent: disconnected`, `engine: pid=700231:exit=143`, `xvfb: pid=700215:exit=0`, `physics_listener_inodes_after: []`, `physics_port_clear: true`, `temporary_clone_removed: true`, no cleanup failures. No stranded processes and no occupied port remain from this wave.
- `sciencebirdsgames/physics-v1/` unchanged: archive digest matches, and all three files still carry their 2026-08-06 23:55 staging mtimes.
- Protected receipt equal across the run (`protected_unchanged: true`).
- Every pre-existing dirty and untracked path is preserved and uncommitted: `.omo/`, `.claude/logs/`, the knowledge-compression additions, the F1-F4 review artifacts, the staged package files, and the pre-existing single-line modification to `.claude/project-docs/README.md`.

## 8. What did not happen

No re-pin. No publication. No overwrite of `sciencebirdsgames/physics-v1/`. No cohort collection. No Todo 8 health report. No retry of any bounded run, and no second smoke — the single permitted full live smoke was not consumed. Phase 5 (two isolated deterministic builds) was not reached: it is gated on the tooling commit, which lands with this wave, and the smoke it feeds is provably unable to pass against this candidate.

## 9. Smallest next step

Fix the collision payload in the Unity player's C# capture path so `collision` events carry sorted unique `contact_ids` and a finite non-negative `relative_speed`, then run Phase 5 (two isolated builds, identical archive/player/assembly/package-input/provenance digests) and spend the single full smoke on the rebuilt candidate. That sequence requires re-pin authorization, which this wave does not hold.

---

# Current verdict — third continuation, 2026-08-12

**Status: `repin_complete`**

This section supersedes the historical `still_blocked` verdicts above as current authority; those sections remain unchanged as the audit trail of earlier candidates. The authorized T1 implementation is commit `d5be336be778103ac2ae883d4d946a1df3eaf540`. Closure records began from branch `physics-unity-2019.4` at HEAD `4bde6262e99c4c8d38bfdccc1af3df240d63b159`.

## Verification closure

- Preserved C# RED: `PhysicsShotRecorderTests` discovered 25 tests, 23 passed, and exactly the two ordering regressions failed. NUnit XML sha256: `787cba1af6aae45c0dcb6ac14dde56ab09efe32a5597711d2ad6ffc78a5456db`. Product source was restored byte-identically to `d6bc41af198c986e8ce371131c617f2c0d125b88f02a30388bd33cdcc6d3a2cd`; the RED run was not repeated.
- Python retention/ordering regression: 3/3 passed.
- Full per-class Unity EditMode: 59/59 passed, 0 failed, 0 skipped.
- Mutation proof: 9/9 mutations turned the suite RED; smoke source restored byte-identically to `ba3c8772534a5e535c1ba7d16faa1427dae342e6a2f7a8e9017f5449e2d05aea`.
- Deterministic builds: build A exit 0; build B exit 0; 151 provenance files compared; zero drift; both archives sha256 `de59061350f78f79420d76ec33f1c506aa17c1cfc25d197cdd2f5f770874e838`.

The first mutation attempt was externally killed at the tool's 120-second ceiling while mutation 2 was active. The actual failure was recorded before repair. The residue was classified EASY: one certain line in `scripts/smoke_physics_capture.py`, no design decision. It was restored exactly, and the single bounded retest passed 9/9 with byte-identical restoration.

## Single full smoke

Exactly one full smoke was consumed against build A. Machine-readable report: `.claude/project-docs/evidence/world-model-physics-instrumentation/task-8-smoke.json`. Output: `.claude/project-docs/evidence/runtime-repin-gate-20260810/session-5-full-smoke/`.

- `status: accepted`; `phase: complete`.
- Provenance archive sha256: `de59061350f78f79420d76ec33f1c506aa17c1cfc25d197cdd2f5f770874e838`.
- Four collision events were observed. The selected collision carried sorted unique contact id `contact:361:-632:0:-646|world:static:-466:-466:0`, fixed step 361, and finite non-negative relative speed `10.6197062`.
- Request identities shared capture id `capture-e5f5d8ab461e466ab27839ba6781f94c` with strictly increasing sequences `1 -> 2`.
- Listener binding and rebinding matched; the stability receipt preserved all 14 identity fields.
- `recorder_refusals: []`; `cleanup.physics_port_clear: true`; temporary clone removed; protected roots unchanged; no cleanup failure was reported.

Two delegated executor sessions failed before `run_smoke`'s first filesystem action. Both times, audits found the named output directory absent, the designated report still stale, the old pin unchanged, no owned process, and port 2004 free. They did not consume the smoke. The direct fallback then ran the exact bounded command once; it accepted and was not retried.

## Conditional re-pin

The pre-smoke staged archive remained `429cac1d748bed417b917d2838dc203d090668977dc8e56f5bac9a80ea95f2de`. Only after the accepting smoke, the authorized conditional re-pin replaced exactly these files under `sciencebirdsgames/physics-v1/`:

1. `archive.sha256`
2. `novphy-physics-player-2019.4.41f2.tar.gz`
3. `unity-build.log`

The staged archive bytes, `archive.sha256`, build-A archive, and smoke provenance now all equal `de59061350f78f79420d76ec33f1c506aa17c1cfc25d197cdd2f5f770874e838`.

## Authority boundary and next decision

No runtime publication occurred. No cohort collection occurred. Repository closure commits are finalized separately and do not publish the runtime artifact. The re-pinned candidate is eligible for a separate publication decision, but this result does not authorize publication. The next action is an explicit owner decision on whether to publish the re-pinned archive; until then, no publication command or cohort command may run.
