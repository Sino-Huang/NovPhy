# Handoff: NovPhy Physics Re-pin Gate — Next Session

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
