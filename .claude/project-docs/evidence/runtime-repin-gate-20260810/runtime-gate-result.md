# Runtime Gate Result

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
