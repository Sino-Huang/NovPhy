# Runtime Gate Result

**Status: `still_blocked`** (current authority — second session, 2026-08-11)

The first session's record is preserved verbatim below under *"Superseded-in-part: first session record"*. Nothing in it was rewritten; the sections that this session advanced are named explicitly in §S2.9.

---

# Second session (2026-08-11) — current authority

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
