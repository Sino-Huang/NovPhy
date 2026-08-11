# Handoff: NovPhy Physics Re-pin Gate — Next Session

> **CURRENT HANDOFF — third session, 2026-08-11.** Read **Part A2 only**. Part A is the second session's handoff and Part B the first's, both preserved verbatim; where any two disagree, the newest wins and says so explicitly.

---

# Part A2 — current handoff (third session, 2026-08-11)

**Wave verdict:** `still_blocked`. Verdict of record: `runtime-gate-result.md` §"Third session" and `runtime-gate-verdict.json`. The second session's machine-readable verdict is preserved byte-identically at `runtime-gate-verdict.session-2.json`.

## A2.0 Read this first, in this order

1. This part (A2). **Do not read Part A or B for current state** — both are superseded on the blocker, and Part A's terminal blocker is now fixed.
2. `finding-sidecar-array-order-violates-contract.json` — the blocker, with the fix already designed.
3. `runtime-gate-verdict.json` — machine-readable, this session.
4. `session-3-plan.md` §"Phase 3b" — the probe table and the reviewer findings not acted on.

Roadmap only from `/mnt/array/sukaih/Project/NovPhy/docs/high_level_plans/bg_ns_jepa_research_execution.md`.

## A2.1 State at the end of this session

| Fact | Value |
|---|---|
| Worktree | `/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4` |
| Branch | `physics-unity-2019.4` |
| HEAD | `ad2822a92688ff6b9e52428eb24d2dc6537165ad` |
| Tree | `231116add9189e06c28f4ab51b4d535766701448` |
| Tracked drift | 0 at Phase 5; the verdict/handoff commit follows |
| Unpushed | 7 commits, nothing pushed |
| Staged pin | `429cac1d748bed417b917d2838dc203d090668977dc8e56f5bac9a80ea95f2de` — **unchanged, not re-pinned** |
| Port 2004 | free (0 / 0) |
| `scripts/__pycache__/` | absent |
| EditMode | 9 classes, 53/53 green |
| `tests.test_smoke_physics_capture` | 75 OK |
| Smoke runs spent this wave | **0** |

## A2.2 What is done, and what is left

**Done, and committed:**

- The gameplay wiring works. `ABBirdBlack` and `ABBlock` now call `PhysicalSnapshotRuntime.RecordCollisionCallback` directly. **Part A's terminal blocker — "no object reachable by a bird shot records a collision at all" — is fixed at the source**, red-then-green.
- The empty-evidence policy is decided and pinned: evidence-bearing overloads log and return; the evidence-free overload (no product caller) still throws.
- A reviewer blocker was found and fixed: the collision path was erasing every other pair's support edges.
- **Phase 5 passed.** Deterministic, `drift: []`, 151 provenance inputs compared.

**Left:** one blocker, then the smoke.

## A2.3 The blocker, and the fix that is already designed

The emitter writes `raw_contacts` **cumulative and step-major**; `scripts/physics_capture_parsing.py:281` requires it globally sorted by a key that **excludes `fixed_step`**. `support_edges` (`:283`) has the same defect class. Both confirmed against the production parser by `probe_raw_contact_order.py` (exit 0).

**Fix (preferred, already specified):** sort once at `PhysicsShotRecorder.CreateFinalizedSnapshot:639-644` — `rawContacts` by `CompareContacts` extended with `ContactId`, `supportEdges` by `(SupporterEntityId, SupportedEntityId, SupportId)`. Do **not** sort in `PhysicsCaptureProtocol`: that puts ordering policy in the serializer rather than in the recorder that owns the invariant. Per-step ordering and `PointIndex` assignment stay untouched, so `contact_id`s do not change.

**Why this session did not do it.** It changes something the Python consumer reads, which is the mission's named stop condition. To be explicit: the fix makes the producer *conform* to the frozen contract — it is not a weakening, and it should be authorized on that basis.

**Why it was not merely theoretical.** It would have failed at `validate-artifact` (`smoke_physics_capture.py:1151`) — *after* `require_collision` passes at `:1131`, so the single non-retryable run would have been fully consumed first.

## A2.4 Authorization you will need

The session-3 authorization covered exactly two gameplay callbacks and the recorder's empty-evidence policy. **The next wave needs authorization for `PhysicsShotRecorder.CreateFinalizedSnapshot`** — the array ordering the Python consumer validates. The conditional re-pin authorization from §0a/§A4 is still on record and still unused; it remains conditional on a passing smoke.

## A2.5 Bundle these into the same wave

All of them reopen `PhysicsShotRecorder.cs` or the same seam, so they are nearly free once F1 is authorized:

- **ABEgg M-1** (`ABEgg.cs:10-13`) — same unrecorded-collision defect. Unreachable on the smoke level, but **must** be fixed before any white-bird cohort.
- **F3** collision evidence looked up by entity pair, ignoring collider identity (minor on single-collider prefabs).
- **F4** the never-throw-in-a-physics-callback property is argued, not enforced by a `try/catch` in `RecordCollisionCallback`.
- **F5** `PhysicsShotRecorder.cs` is 814 lines, over the project's 800-line rule.
- **F6** `currentStep`/`currentTime` are write-only.
- **F7** the contact stream is unbounded in shot duration; ~180k contacts at the 120 s ceiling would trip `RecordLimitExceeded`.
- **m-2** smoke-harness log scan for `physics_capture_v1: refusing` — requires updating the pinned digest in `mutation_check.py` in the same commit.

## A2.6 Traps this session paid for. Do not re-discover them.

- **`MethodBase.Invoke` dispatches virtually.** A `protected virtual` base implementation cannot be invoked on its own by reflection — `AwakeComponent(block, typeof(ABGameObject))` re-entered `ABBlock.Awake`. Bind the individual field instead.
- **`ABBird.Start` (not `Awake`) creates `_trailParticles`**, and also does `Resources.Load` plus an `Invoke`, neither usable in EditMode. Set that one field directly.
- **A defense value of `1e9f` makes a life assertion vacuous** — both the `ABBlock` and `ABGameObject` damage formulas clamp to the same result. Pick a value between them (47f at relativeSpeed 4.25) and prove it discriminating by mutation.
- **`_parse_support:260` requires `support_id == "support:{supporter}->{supported}"`.** An invented id fails on format *before* the ordering check runs, so a probe using one proves nothing. This cost one wrong `f2_not_reproduced`.
- **`pgrep -f` on a Unity-ish pattern matches your own shell**, whose command line contains the worktree path `physics-unity-2019.4`. Bind by `/proc/<pid>/exe`, never by name.
- Carried forward and still true: `PYTHONDONTWRITEBYTECODE=1`; commit before building; new `.cs` files need a committed `.meta`; run EditMode per class; NUnit XML is the authority (editor exits `-6` in CEF shutdown).

## A2.7 Verify before you start

```
git rev-parse --abbrev-ref HEAD                 # physics-unity-2019.4
git rev-parse HEAD                              # ad2822a… or later
git status --porcelain --untracked-files=no     # empty
sha256sum sciencebirdsgames/physics-v1/novphy-physics-player-2019.4.41f2.tar.gz
#   429cac1d748bed417b917d2838dc203d090668977dc8e56f5bac9a80ea95f2de
grep -ci ':07D4' /proc/net/tcp /proc/net/tcp6   # 0 0
test -d scripts/__pycache__ && echo PRESENT || echo absent   # absent
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_smoke_physics_capture   # 75 OK
```

Do **not** reuse Phase 5's archive `2bdd498a…48847e`. The F1 fix changes `Assembly-CSharp.dll`, so the build must be re-run and re-verified.

## A2.8 Suggested opening move

Get authorization for `CreateFinalizedSnapshot`'s array ordering, then: write the two EditMode ordering fixtures red → apply the sort → green → run `probe_raw_contact_order.py` against emitter-shaped data derived from a real finalized snapshot → re-run Phase 5 → spend the one smoke. The conditional re-pin applies only after that smoke accepts.

---

# Part A — second session's handoff (2026-08-11, superseded, preserved verbatim)

**Wave verdict:** `still_blocked`. Verdict of record: `runtime-gate-result.md` §"Second session" and `runtime-gate-verdict.json` (both in this directory).

## A0. Read this first, in this order

1. This Part A.
2. `runtime-gate-verdict.json` — the machine-readable verdict, every digest.
3. `runtime-gate-result.md` — the narrative verdict; the first session's record is preserved verbatim at the bottom.
4. `finding-smoke-level-geometry-risk.json` — **the terminal blocker.** Read it before planning anything.
5. `finding-collision-payload-root-cause.json` — why the payload was empty, and why it is already fixed.
6. `finding-simplejson-roundtrip-hides-json-types.json` — a durable EditMode assertion trap.
7. Part B below, then `notes.md`, `task_plan.md`.
8. Roadmap, and only from here: `/mnt/array/sukaih/Project/NovPhy/docs/high_level_plans/bg_ns_jepa_research_execution.md`.

## A1. State at the end of this session

| Fact | Value |
|---|---|
| Working tree | `/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4` |
| Branch | `physics-unity-2019.4` |
| HEAD | the `docs(runtime-gate): record the still_blocked verdict and Phase 5 determinism` commit — the evidence commit, whose parent is `045296d` |
| Phase 5 provenance HEAD / tree | `045296d6ed9f749d8ea12ca2e4b345d72e5dfce8` / `40f30b59dd8fbf38fc2387d7c0aaf8481a146b89` |
| Unpushed | 4 commits ahead of `origin/physics-unity-2019.4` (`7f1e8727…`); nothing was pushed |
| Staged pin (unchanged) | `429cac1d748bed417b917d2838dc203d090668977dc8e56f5bac9a80ea95f2de` |
| Port 2004 | free |
| Stranded processes | none (bound by `/proc/<pid>/exe`, not by name) |
| Single full smoke | **not spent** |

Three commits landed: `a1067e5` (README knowledge-index line, the old TODO-6), `87365fb` (smoke known-action offset + the blocker annotation), `045296d` (the wire-level collision-payload fixture). HEAD moved from the mission's start point `7f1e8727…` because `package_physics_player.py`'s `git_revision` gate refuses to package tracked drift, and Phase 5 needed a clean tree.

**Corrects Part B §0:** the HEADs named there (`d83c1487…`) are two waves stale. Use the table above.

## A2. What is done, and what is left

**Done and passing:**
- **Phase 5 — deterministic.** Two isolated builds via `NOVPHY_PHYSICS_STAGE`, 151 provenance files compared, zero drift. Archive `d4e55bc4f684ecd4699c81d0c039ab43ab62c70ccf1a5b42d2455e0732147562`, player `d74bf3f8…`, `Assembly-CSharp.dll` `5d83af30…`, `UnityPlayer.so` `53b0b8d1…`, `provenance.json` `1dc3097e…`. Receipts in `phase5-builds/`. **This is already satisfied for the current source** — after the fix in A3 you must re-run it, because the source will change.
- **The empty collision payload is fixed on this branch**, by `7a2dd02` and `97c4dd6`, both of which postdate the staged build `e2d19ae`. The staged binary is stale relative to its own source. No emitter change was needed or written.
- **The wire seam is now covered.** `PhysicsCaptureProtocolTests.Request70CollisionPayloadCarriesTheContactEvidenceTheContractRequires`, proved red against a mutation that reproduces `e2d19ae`'s emitter, then green. EditMode full suite 48/48 across 8 classes.

**Left, and blocking:** the smoke's acceptance criterion needs a genuine collision, and on the level the smoke plays **nothing a bird can reach records one**. See A3.

## A3. The terminal blocker, and the exact fix

Level `novelty_level_0/type2/Levels/3_9_6_1.xml` (chosen by `scripts/build_physics_player.sh:47` rewriting `config.xml` for `ui_level 1`):

- Bird is `BirdBlack`; `ABBirdBlack.cs:22` overrides `OnCollisionEnter2D` and never calls base → the recorder is never reached from the bird.
- 8 Platforms carry zero `m_Script`; Ground is a bare `BoxCollider2D` → neither records.
- `ABBlock.cs:145` calls base only in the **non-bird** `else` branch → a bird hitting a block records nothing.
- Only Pigs record, and both are walled in by vertical platforms → structurally unreachable from the slingshot at `x = -12`.
- **Aim cannot fix this.** Both `[-50, 40]` and `[-80, 7]` saturate the drag clamp (`_dragRadius 1`, `ABBird.cs:233-234`); only elevation changes (≈39.7° → ≈5.0°), never speed.

The fix, in order — do not skip step 2:

1. Call `PhysicalSnapshotRuntime.RecordCollisionCallback(collision)` **directly, not `base`**, at the top of `ABBirdBlack.OnCollisionEnter2D`, and at the top of `ABBlock.OnCollisionEnter2D` hoisted out of the `else` branch so both branches record. A `base` call is wrong here: `ABGameObject.OnCollisionEnter2D:125-141` also runs the damage model, so hoisting base would change gameplay.
2. **Decide the empty-evidence policy at `PhysicsShotRecorder.cs:531`** before wiring more objects to the recorder. It currently throws `ArgumentException`. Thrown inside a Unity physics callback it aborts the handler → the `BirdBlack` explosion never plays → no terminal event → the smoke's 30 s finalize deadline expires with the single run consumed. A `Debug.LogError` plus early return is equally fail-closed *at the wire* — no event is emitted, so the smoke's `require_collision` still rejects — without killing the handler.
3. Add EditMode fixtures proving a `BirdBlack` impact and a bird-on-block impact each produce a recorded collision event. Red before, green after.
4. Re-run Phase 5 (`phase5_build_twice.py` + `compare_builds.py` in this directory), then spend the single full smoke on the rebuilt candidate.

This is two gameplay classes plus a fail-closed throw path — beyond the previous wave's TODO-2 scope, which is why it was recorded and reported instead of improvised on a non-retryable run budget.

## A4. Authorization still on record

Unchanged from Part B §0a and **still conditional**: re-pin `sciencebirdsgames/physics-v1/` only *after* a passing full smoke against the rebuilt candidate. Publication is **not** authorized — stop and report. Cohort collection is **not** authorized.

## A5. Corrections to Part B — read these, they will cost you a run otherwise

1. **`mutation_check.py`'s pinned baseline is now `72f6a12183df97755ab715919557d70eb7cf5c59e9c8311dcab0c9925288b6f6`**, not the `f61ccfa5…` printed in Part B §5. The rule stands: edit `scripts/smoke_physics_capture.py` and you must update that constant in lockstep, or the harness refuses to mutate a file it has not verified.
2. **`tests.test_smoke_physics_capture` is now 75 tests**, not 74.
3. **Part B §3's TODO-1 is obsolete.** The collision payload is not the thing to fix; A3 is.
4. **Part B §3's TODO-2 is done** (`87365fb`) — and the offset it applies is *insufficient on this level*, which is exactly why A3 exists. The source carries that warning in a comment at `perform_known_action`.
5. **Part B §3's TODO-3 (Phase 5) is done and passed** for the current source.
6. **Part B §3's TODO-6 is done** (`a1067e5`).
7. **Part B §6 trap 3's exit code is imprecise.** EditMode invocations on this host exit with a signal inside CEF shutdown — observed as `-6`/`134` depending on how it is read. The point is unchanged: the exit code is not authoritative, the NUnit XML is.
8. **A single unfiltered EditMode run crashes in `CefBrowserMessageLoop` before flushing its result file**, producing no XML despite executing every test. Run per test class — `editmode_full_suite.py` in this directory does exactly that. See `finding-editmode-harness-deadlock.json`.
9. **Do not let `python -m unittest` run before a build.** It writes `scripts/__pycache__/`, which `package_physics_player.py:105` rejects as untracked product source, aborting the build. Use `PYTHONDONTWRITEBYTECODE=1`, or remove the directory before building.

## A6. New trap this session paid for

**In EditMode fixtures, assert JSON *structure* through SimpleJSON and JSON *scalar types* through the raw serialized string — never mix the two.** This SimpleJSON build stores every scalar as text and re-quotes it on `ToString()`, so a parsed node cannot tell a JSON number from a JSON string. A type assertion made against `node.ToString()` went RED against provably correct product output; the previous `AsFloat` form was GREEN and would have passed a contract-violating string. Neither carried information about the wire. Full record: `finding-simplejson-roundtrip-hides-json-types.json`.

## A7. Verify before you start

```bash
cd /mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4
/usr/bin/git rev-parse --abbrev-ref HEAD                # physics-unity-2019.4
/usr/bin/git log --oneline -5                           # top: docs(runtime-gate): record the still_blocked verdict…
                                                        # its parent 045296d is the Phase 5 provenance HEAD
/usr/bin/git rev-list --count origin/physics-unity-2019.4..HEAD   # 4, nothing pushed
/usr/bin/sha256sum sciencebirdsgames/physics-v1/novphy-physics-player-2019.4.41f2.tar.gz
#   429cac1d748bed417b917d2838dc203d090668977dc8e56f5bac9a80ea95f2de
/bin/grep -c ' 07D4 .* 0A ' /proc/net/tcp /proc/net/tcp6   # 0 and 0
test -d scripts/__pycache__ && echo REMOVE-IT || echo clean

PYTHONDONTWRITEBYTECODE=1 python -W error::ResourceWarning -m unittest tests.test_smoke_physics_capture   # 75 OK
PYTHONDONTWRITEBYTECODE=1 python .claude/project-docs/evidence/runtime-repin-gate-20260810/mutation_check.py  # 8/8 red
```

## A8. Suggested opening move

A3 step 2 — decide the `PhysicsShotRecorder.cs:531` empty-evidence policy — before writing any gameplay wiring. Every other step in A3 widens exposure to that throw, and it is the one failure mode that burns the single run without producing a diagnosis.

---

---

# Part B — first session's handoff (verbatim, unaltered)

Retained exactly as written. Where it disagrees with Part A, Part A is current; the specific corrections are enumerated in A5.

**Written** 2026-08-11 by the session that closed wave `runtime-repin-gate-20260810`.
**Wave verdict:** `still_blocked`. Verdict of record: `runtime-gate-result.md` (same directory).

---

## 0. Read this first, in this order

1. This file.
2. `runtime-gate-result.md` — the verdict, with every digest and receipt.
3. `notes.md` — the working log; sections are historical receipts, appended to and never rewritten.
4. `task_plan.md` — phase checkboxes and the error log.
5. `review-remediation.json` — review pass 6, its seven MAJORs, and the mutation results.
6. Roadmap, and only from here: `/mnt/array/sukaih/Project/NovPhy/docs/high_level_plans/bg_ns_jepa_research_execution.md`.

Working tree: `/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4`, branch `physics-unity-2019.4`.
Branch HEAD and `origin/physics-unity-2019.4` are both `d83c148739188e93a4c5aa509d13989928d4361f`. Nothing is unpushed.

---

## 0a. Authorization on record

**Re-pin of `sciencebirdsgames/physics-v1/` is AUTHORIZED.** Granted by the repository owner on 2026-08-11, in response to this handoff. Scope, exactly:

- **Authorized:** rebuild the player from the fixed source and overwrite the staged pin in `sciencebirdsgames/physics-v1/`, **only after the full smoke accepts** against that build. A re-pin before a passing smoke is not covered.
- **NOT authorized:** publication. Stop and report before publishing; that is a separate decision the owner has not made.
- **NOT authorized:** cohort collection, in any form.

Everything else in §4 stands unchanged. This authorization supersedes only the first bullet of §4's "Do not" list.

---

## 1. Where the gate actually stands

**Working, verified, committed:** the smoke binds the physics listener to process-backed identity, end to end, against the live staged candidate. This was proven live, not argued (`listener-diagnostic-corrected.json`, `diagnostic_accepted` at `phase: listener-bound`).

The chain, in the order it is enforced:

```
socket inode on port 2004  (loopback-only, fully attributed, exactly one owner)
  -> owning pid via /proc/<pid>/fd
  -> /proc/<pid>/exe, /proc/<pid>/cwd
  -> mapped UnityPlayer.so: device, inode, sha256
  -> the same runtime digest inside provenance.json
  -> Assembly-CSharp.dll via /proc/<pid>/root/: device, inode, sha256
  -> the same assembly digest inside provenance.json
  -> provenance.json bytes digest
  -> staged archive digest
  -> membership in the process tree rooted at the pid this run launched
```

Re-confirmed at the end of the identity reads against `/proc/<pid>/stat` `starttime` and the descriptor table, so an owner that exits mid-observation cannot have its pid recycled into a spliced binding.

**Blocked, and not fixable from Python:** the staged player emits `collision` events with payload `{}`, while the frozen capture contract requires `contact_ids` and `relative_speed`. `validate_physics_shot_artifact` therefore rejects every shot containing a collision — and a shot with no collision fails the acceptance criterion directly. Both branches reject. The fix is player-side C#.

---

## 2. The rollout you pointed at — checked, and what it does and does not prove

`data/novphy_rollouts_dataset_20260708_171531/train/novelty_level_4_type010401_00141_0_1_010401_4_1/shot_001/rollout.mp4`

**It does not move the blocker.** That shot is from the legacy cohort (collected 2026-07-19) and is `legacy_rgb_v1`: RGB frames plus `metadata.json`, with no physics sidecar at all. Verified directly — the strings `collision`, `contact_ids`, `relative_speed`, `physics_capture`, `capture_contract` and `capture_id` appear nowhere in either `shot_001/metadata.json` or the cohort `manifest.json`. The bird visibly striking the ice triangle is real; nothing recorded it as a structured event, because that cohort predates physics capture entirely. So it neither confirms nor contradicts the empty-payload defect.

**It is still useful, for a different reason.** It is a known-good, reproducible shot that physically produces a bird-on-ice-triangle contact, which is exactly what the final smoke needs once the player is fixed:

| Field | Value |
|---|---|
| `ui_level` | 1 |
| `action_type` | `drag_hold_release` |
| `coordinate_frame` | `slingshot_relative` |
| `drag_start` | `[97, 227]` |
| `drag_release` | `[-80, 7]` |
| `tapTime` / `holdTime` | `0` / `1000` |
| slingshot reference | `gameX 97, gameY 227, canvasX 97, canvasY 252` |
| capture | 104 frames, 30 fps, 5.0 s |
| validation | `accepted`, `gameplay-valid` |

Carry this into the smoke's `perform_known_action` instead of whatever arbitrary shot it takes now, so the one permitted run is not spent on a shot that happens to miss.

**If you want to overturn the blocker instead**, the only evidence that does it is a raw `physics_capture_v1` sidecar — the actual event objects, not a summary — from a shot with a collision, together with the archive digest of the player that produced it. If that digest is `429cac1d…` and the payload is populated, the defect is conditional, not universal, and Phase 6 becomes worth running as-is. If the digest is anything else, the fix already exists somewhere and the task is a re-pin, not a repair.

---

## 3. Remaining todos, in dependency order

### TODO-1 — Fix the collision payload in the Unity player *(blocking everything below; needs re-pin authorization)*
Make `collision` events carry sorted unique non-empty `contact_ids` and a finite non-negative `relative_speed`, matching the frozen contract. Add or extend an EditMode fixture that fails on an empty payload before you touch the emitter, so the fix is proven rather than assumed.
- Evidence: `finding-collision-payload-blocks-the-final-smoke.json`; upstream `.claude/project-docs/evidence/enriched-cohort-oracle-labels/task-7-collection-blocker.json`.
- Suspect area: the shot recorder's collision callback path; `PhysicsShotRecorderTests` and `PhysicalShotRecorderTests` already exist and pass, so they do not currently cover payload population.

### TODO-2 — Point the smoke's known action at a collision-producing shot
Use the table in §2. Small, self-contained, and doable *before* TODO-1 lands — it costs nothing and removes a way for the single smoke to be wasted.

### TODO-3 — Phase 5: two isolated deterministic builds
Build the exact committed source twice into non-production stages via `NOVPHY_PHYSICS_STAGE` (`scripts/build_physics_player.sh:7` already honours it — no script edit needed). Require identical archive, player, `Assembly-CSharp.dll`, package-input and provenance digests. `compare_builds.py` in this directory is written and unused. Any nondeterminism ends the wave `still_blocked`.
- Packaging is deterministic by construction (`scripts/package_physics_player.py:202`: PAX tar, uid/gid 0, empty uname/gname, mtime 0, `gzip -n -9`; `write_manifest` embeds no timestamp), so drift would originate in Unity's own output, not in packaging.
- `git_revision` refuses to package while tracked product source differs from HEAD, so **commit before building**.

### TODO-4 — Phase 6: exactly one bounded full live smoke
One run, one verified build, no retry, no fallback candidate, no collision-free success path, no handwritten evidence. Acceptance needs all of: one exact listener/process/package binding; two request-70 responses with the same non-empty `capture_id` and strictly increasing sequence; at least one genuine collision with non-empty sorted unique `contact_ids` and finite non-negative `relative_speed`; the same candidate process and package for both the request and the collision; complete cleanup; unchanged protected roots and staged player.

### TODO-5 — Finish the verdict
Write `ready_for_repin_approval` or `still_blocked` into `runtime-gate-result.md`, carrying exact source/tree, both deterministic digest sets, the selected smoke candidate, the listener/process binding, request identities, the collision, tests/review, and cleanup/protected receipts.

### TODO-6 *(optional, one line)*
`.claude/project-docs/README.md` has a pre-existing one-line knowledge-compression edit, deliberately left uncommitted by this wave because the wave's constraints forbade incorporating knowledge-compression additions. If the next wave's constraints allow it: `docs(project-docs): describe the compressed knowledge index`.

---

## 4. Constraints that carry forward

**Do not:**
- Re-pin, publish, or overwrite `sciencebirdsgames/physics-v1/` without explicit authorization. Any commit to the Unity project changes the archive SHA and forces a full rebuild → republish → re-smoke.
- Clean, reset, move, or incorporate `.omo/`, `.claude/logs/`, the knowledge-compression files, the F1–F4 review artifacts, or the staged production files.
- Weaken the frozen schema, taxonomy, or Python consumer to make a payload fit.
- Collect the Milestone 0 enriched cohort, emit the Todo 8 health report, or start F1–F4 / JEPA / SPSG / controller work.
- Touch the main checkout's Unity project or the protected rollout data.
- Infer process identity from names or command substrings. Reject zero or multiple owners, absent or ambiguous mappings, stale listeners, and any digest drift.

**Do:**
- Lock branch, HEAD, dirty state, staged archive, protected roots, editor/package inputs, candidate provenance, process state and ports *before* editing.
- Use absolute `/bin/` paths for evidence-bearing commands. Shell aliases shadow them here: `ps`→`procs`, `df`→`duf`, `ls`→`lsd`.
- Treat NUnit XML as authority for Unity tests. Every EditMode invocation exits **134** from a SIGABRT inside `CefShutdown`, reached after the run finished and after the XML was written. A compiler or editor failure *before* discovery is `still_blocked`.
- Preserve listener, request, collision, cleanup and protected-root facts in the report even when a later assertion fails.

---

## 5. State to verify before you start

```bash
cd /mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4
git rev-parse --abbrev-ref HEAD                 # physics-unity-2019.4
git rev-parse HEAD                              # must equal origin/physics-unity-2019.4
git rev-parse origin/physics-unity-2019.4       # and the HEAD named at the top of this file
git status --porcelain | grep -v '^??'          # only:  M .claude/project-docs/README.md
sha256sum sciencebirdsgames/physics-v1/novphy-physics-player-2019.4.41f2.tar.gz
#   429cac1d748bed417b917d2838dc203d090668977dc8e56f5bac9a80ea95f2de
grep -c ' 07D4 .* 0A ' /proc/net/tcp /proc/net/tcp6   # 0 and 0 — port 2004 must be free
/bin/ps -eo pid,cmd | grep -Ei 'Xvfb|game_playing_interface|9001\.x86_64' | grep -v grep
```

Regression suites, both expected green:

```bash
python -W error::ResourceWarning -m unittest tests.test_smoke_physics_capture            # 74 OK
python -W error::ResourceWarning -m unittest \
  tests.test_build_physics_player tests.test_package_physics_player \
  tests.test_physics_capture_contract tests.test_verify_physics_player \
  tests.test_verify_physics_capture_docs                                                 # 60 OK
python .claude/project-docs/evidence/runtime-repin-gate-20260810/mutation_check.py       # 8/8 red, source restored
```

`mutation_check.py` pins its baseline digest `f61ccfa52ef5bed5e156e8480f32f407c13286276df00d8ec676e9671fc84978`. **If you edit `scripts/smoke_physics_capture.py`, update that constant** or the harness refuses to run — deliberately, so it can never mutate a file it has not verified.

---

## 6. Traps this wave paid for. Do not re-discover them.

1. **The jar does not own port 2004, and does not spawn the player that does until an agent completes its handshake.** Order is connect → configure → bind. Binding first has no reachable success path in either mode; it can only time out. Pinned by `test_the_agent_connects_before_the_physics_port_is_probed`.
2. **`--physics-port` does not reach the player.** Driving a random port through it produces no listener (`player-mapping-probe-wrong-port.json`). The player uses its own default.
3. **Unity 2019.4 never file-backs `Assembly-CSharp.dll`.** Mono mmaps the class libraries; Unity loads the user assembly through its own loader. The mapped anchor is `UnityPlayer.so`. This deviates from the mission's literal wording and is flagged in `runtime-gate-result.md` §2 — do not "restore" the old anchor, it is unsatisfiable.
4. **Teardown must be keyed on the process group, not the leader.** The player is a grandchild of the JVM; a reaped leader used to skip `killpg` entirely and strand the player on port 2004, poisoning every later run on this host. The group census excludes zombies on purpose — the leader stays a zombie until reaped and holds no socket.
5. **Do not treat `FileNotFoundError` from a `/proc/<pid>/fd` readlink as a blind spot.** `iterdir()` holds its own descriptor, closed before the readlink loop runs, so every process on the host including the test process lands in the census. Tried, measured, reverted; the reason is recorded in-source beside the `continue`.
6. **Roughly 785 pids on this host are opaque to `/proc/<pid>/fd`.** Rejecting on opacity makes the gate reject unconditionally. The invariant that closes the impostor hole is inode reconciliation — every listening inode on the port must have an attributed owner — and the opaque count is reported as diagnosis alongside the accept.
7. **`scripts/smoke_physics_capture.py` exceeds the project's 800-line ceiling, knowingly.** Extraction is deferred because the tests patch through `smoke.X` bindings and cross-module internal calls would silently break patchability. It is a style rule, not a correctness invariant.
8. **Budget the protected receipt.** It measured 243.6 s and is charged on both the before and after pass; a 300 s ceiling is not enough and produced two exit-124 timeouts.

---

## 7. Suggested opening move

TODO-2 first — it is small, needs no authorization, and removes a way to waste the single smoke. Then ask for re-pin authorization for TODO-1, since nothing below it can proceed without a player that emits the payload.
