# Task Plan: Unity Runtime Re-pin Gate

## Goal
Finish with either `ready_for_repin_approval` backed by deterministic build and one live-smoke evidence package, or `still_blocked` naming the first failed invariant.

## Scope Boundaries
- Do not re-pin, publish, overwrite `sciencebirdsgames/physics-v1/`, or collect a cohort.
- Preserve all pre-existing dirty and untracked files, including knowledge compression, `.omo/`, `.claude/logs/`, and F1-F4 review artifacts.
- Modify only `scripts/smoke_physics_capture.py`, `tests/test_smoke_physics_capture.py`, and this runtime-gate evidence directory unless new evidence requires a narrower explicit addition.
- Run exactly one final full live smoke, only after focused tests, Unity fixtures, and two deterministic builds pass.

## Phases
- [x] Phase 1: Lock branch, source, staged pin, protected roots, editor/package inputs, processes, and ports.
- [x] Phase 2: Diagnose listener ownership through a bounded non-gameplay path.
- [x] Phase 3: Add red tests and implement exact listener/process provenance checks with durable diagnostics.
- [x] Phase 4: Run focused Python tests and affected Unity NUnit fixtures.
- [x] Phase 4c: Bounded listener-only diagnostic (twice: pre-correction, then against the corrected anchor).
- [ ] Phase 5: Commit verified tooling, build twice from the exact clean commit, and compare digests. **Not reached** — see Status.
- [ ] Phase 6: Run exactly one bounded final full live smoke. **Deliberately not run** — provably cannot pass against this candidate, and there is no retry.
- [x] Phase 7: Review exact source, remediate blockers, record authority evidence, and commit scoped evidence.

## Key Questions
1. Which existing smoke boundary launches the process tree and where did substring-based identity lose the listener PID?
2. Can socket ownership be resolved unambiguously from port 2004 and bound to `/proc` identity before gameplay?
3. Are both isolated builds byte-identical across archive, player, assembly, and provenance receipts?
4. Does the single final smoke prove request identity, collision validity, process provenance, cleanup, and protected-root equality?

## Decisions Made
- Use the latest task-7 boundary JSON as current authority; historical blockers remain evidence only.
- Store working records under `.claude/project-docs/evidence/runtime-repin-gate-20260810/`.
- Treat the staged archive and all protected roots as read-only.

## Errors Encountered
- Initial process snapshot invoked the user-installed `procs` alias instead of `/bin/ps`; reran with `/bin/ps`. This did not mutate state. Same class recurred with `df` aliased to `duf`; use absolute `/bin/` paths for every evidence-bearing command.
- `apply_patch` is unavailable in the shell; use dedicated Write/Edit tools after reading targets.
- `SmokeError` was a frozen slots dataclass subclassing `Exception`; unittest's traceback assignment hit the frozen `__setattr__` and masked every real failure. Converted to a plain `Exception` subclass.
- The first review demanded rejecting whenever any process was opaque to `/proc/<pid>/fd`. Implemented, then measured: on this shared host ~700 pids are EACCES, so the gate rejected unconditionally. The invariant kept is inode reconciliation (every listening inode on the port must have an attributed owner), which is what actually closes the co-resident-impostor hole; the opaque pid count is reported as diagnosis.
- The second review found no blockers but three MAJORs, and all three were introduced by the first round's own remediations: the atomic report publish left a prior run's `accepted` receipt readable at the designated path; `Path.replace` raised `ENOTEMPTY` out of the exception handler when a prior quarantine already occupied the destination, leaving `shot_001` published and skipping the `accepted_shot` pop; and the cleanup-path pop sat inside the quarantine `try`, so a failed move left a rejected report still naming an accepted shot. Remediation converts both quarantine sites to one `quarantine_artifact` helper, pops `accepted_shot` before any move, and removes both the stale receipt and the staging file when publish fails.
- Two existing tests pre-created `shot_001` to simulate publication. The pre-existing-artifact rule (a prior run's artifact is not this run's to quarantine) correctly reclassified them, so both fixtures now create the shot during the run via a `wait_for_listener` side effect.
- The third and fourth reviews each again found a defect introduced by the preceding round's own remediation, both in the evidence-persistence layer and neither in the identity chain. Pass 4's BLOCKER was reachable with an ordinary unprivileged symlink loop: `Path.exists()` swallows `ELOOP` while a raw `stat()` propagates it, so an unprobeable name read as "absent" and a prior run's evidence could be quarantined by this run. Probing is now an explicit `stat`, an unprobeable name is recorded as blind, and blind means untouchable.
- A residual style violation is accepted deliberately: `scripts/smoke_physics_capture.py` exceeds the project's 800-line ceiling. Extraction is deferred because the tests patch through `smoke.X` bindings and cross-module internal calls would silently break patchability. It is a style rule, not a correctness invariant, and every remediation round so far has introduced a new defect.
- The listener-only diagnostic timed out twice (exit 124) for two independent reasons. First, a 300s ceiling against a measured 243.6s `active_data` protected receipt charged on both the before and after passes. Second, and decisively, `run_smoke` bound the physics listener *before* connecting the agent — but a bounded probe proved the jar does not own that port and does not spawn the player that does until an agent connects (60s of polling: `player_procs=0 port2004=0` at every sample). The bind had no reachable success path in either mode. Reordered to connect -> configure -> bind, pinned by a call-order test verified red against the old ordering.
- The fifth review found 0 BLOCKER, 4 MAJOR, 3 MINOR, and again every MAJOR was introduced by the preceding round's own remediation — this time in the teardown the reorder created. The relocated `bridge.disconnect()` could replace the in-flight exception and destroy the run's only recorded rejection reason; the 10s agent read timeout now sat on the critical path of Unity cold start; `terminate` skipped `killpg` entirely once the JVM leader was reaped, orphaning the Unity grandchild still holding port 2004 (which is exactly the pair of stranded `Xvnc` processes observed earlier); and the port-clear check raced that grandchild's exit. All four are remediated with tests verified red by mutation.
- The order test pinned only `["connect", "bind"]` while the rationale rests on the completed *handshake*. Swapping configure past the bind kept it green. It now records and asserts `["connect", "configure", "bind"]`.
- The gate's central mapped-code invariant was unsatisfiable: Unity 2019.4 never maps `Assembly-CSharp.dll`. See `notes.md`, "Mapped-Anchor Correction". The anchor is now `UnityPlayer.so`, with the assembly pinned by digest through the listener's own root. This deviates from the mission's literal wording and is flagged, not silent. **Closed**: the corrected gate then bound the live staged candidate exactly (`listener-diagnostic-corrected.json`, `diagnostic_accepted` at `phase: listener-bound`), with clean teardown and a clear port.
- The staged candidate cannot satisfy the full smoke's collision criterion, for a player-side reason no Python change can reach. See `notes.md`, "Staged Candidate Cannot Pass the Final Smoke". The one permitted smoke was deliberately not spent on it.
- The sixth review found 0 BLOCKER, 7 MAJOR. Unlike passes 2-5 this one was not dominated by self-inflicted regressions: it validated the identity chain, and its two most useful findings were coverage gaps proven by surviving mutations (`_process_view` as identity, `_process_tree` as direct-children-only) plus one genuinely unpinned input, the launch environment. All seven are remediated with tests verified red by mutation; see `review-remediation.json`.

## Status
**still_blocked.** One blocker was found and closed inside this wave (the unsatisfiable mapped-assembly anchor; the corrected gate now binds the live candidate exactly). The remaining blocker cannot be fixed within this wave's permissions: the staged player's empty collision payload makes the single permitted full smoke a guaranteed rejection, and the fix is player-side C# reachable only through a re-pin. Focused Python 74/74 green with `ResourceWarning` promoted to error; 60/60 across five adjacent physics tooling suites; eight mutations confirm each new gate assertion bites, with source restored byte-identical afterwards. Phases 5 and 6 (deterministic builds, final smoke) were not reached: the builds are gated on the tooling commit, and the smoke is provably unable to pass against this candidate.

---

## Second session (2026-08-11) — appended, nothing above rewritten

### Phases
- [x] TODO-1 (point the known action): applied the handoff-verified offset `[-80, 7]` — commit `87365fb` — then discovered it is *insufficient on this level* and annotated the source at `perform_known_action` with the reason and the finding path.
- [x] TODO-2 (collision payload): diagnosed in scope; already fixed on this branch by `7a2dd02` and `97c4dd6`, which postdate the staged build `e2d19ae`. No emitter change written. Closed the missing wire-level fixture instead, red before green.
- [x] Phase 5 (two isolated deterministic builds): **PASSED.** 151 provenance files compared, zero drift.
- [ ] Phase 6 (single full live smoke): **deliberately not spent.** No reachable success path — see below.
- [ ] Re-pin: not performed. Conditional on a passing smoke.
- [x] Verdict, evidence, handoff, commits.

### Errors Encountered
- The fixture's review-requested JSON-*type* assertion went RED against provably correct product output. It ran against `collision["payload"].ToString()`, and this SimpleJSON build stores every scalar as text and re-quotes on `ToString`, so a parsed node cannot express a JSON type at all. The previous `AsFloat` form was GREEN and would have passed a contract-violating string value. Neither state carried information about the wire. Fixed by matching the raw serialized envelope, anchored on `contact_ids` and pinned to exactly one match. Recorded in `finding-simplejson-roundtrip-hides-json-types.json`.
- Phase 5 `build-a` aborted at `package_physics_player.py:105` with `PackagingError: untracked product source: !! scripts/__pycache__/`. Self-inflicted: this session's own `python -m unittest` runs wrote bytecode into the gate's untracked scope. The directory held 48 `.pyc` files only and `package_physics_player.py` imports stdlib exclusively, so removal could not change packaging behaviour. Removed; both builds re-run with `PYTHONDONTWRITEBYTECODE=1` and exited 0.
- A single unfiltered EditMode run crashes in `CefBrowserMessageLoop` before flushing its result file, producing no XML despite executing every test. Per-class invocation via `editmode_full_suite.py` is the working form.
- HEAD had to move (`7f1e8727…` → `045296d6…`) because `git_revision` refuses to package tracked drift from HEAD. The three commits are the wave's own required outputs.

### Status
**still_blocked.** The terminal blocker is no longer the collision payload — that is fixed on this branch and the staged binary is simply stale relative to its own source. It is level geometry: on `novelty_level_0/type2/Levels/3_9_6_1.xml`, no object a bird can reach invokes the recorder. `BirdBlack` overrides `OnCollisionEnter2D` without calling base; platforms and ground carry no `ABGameObject`; `ABBlock` records only on its non-bird branch; only pigs record, and both are walled in. Aim cannot fix it — both pulls saturate the drag clamp, so only elevation changes. The fix reaches two gameplay classes plus the fail-closed throw at `PhysicsShotRecorder.cs:531`, which is beyond this wave's TODO-2 scope, so it was recorded and reported rather than improvised against a non-retryable run budget. Phase 5 ran anyway and passed. Tests: 75/75 focused Python, 8/8 mutations red with byte-identical restore, 48/48 EditMode across 8 classes, one fixture red-then-green. No smoke spent, no retry, no re-pin, no publication, no cohort collection; the staged pin is byte-identical to the session start.

---

## Third session (2026-08-12) — authorized T1 continuation

### Phases
- [x] Cleared the packaging blocker by deleting exactly `scripts/__pycache__/`, the only deletion authorized for this continuation.
- [x] Preserved the accepted C# RED proof: 25 discovered, 23 passed, exactly two ordering regressions failed; XML sha256 `787cba1af6aae45c0dcb6ac14dde56ab09efe32a5597711d2ad6ffc78a5456db`; source restored to `d6bc41af198c986e8ce371131c617f2c0d125b88f02a30388bd33cdcc6d3a2cd`.
- [x] Python ordering regression: 3/3 passed, including both `DETERMINISTIC_ORDER` rejection guards.
- [x] Full per-class EditMode: 59/59 passed across nine classes.
- [x] Mutation proof: 9/9 RED; `scripts/smoke_physics_capture.py` restored byte-identically to `ba3c8772534a5e535c1ba7d16faa1427dae342e6a2f7a8e9017f5449e2d05aea`.
- [x] Phase 5: build A and build B exited 0; 151 provenance files compared; zero drift; archive sha256 `de59061350f78f79420d76ec33f1c506aa17c1cfc25d197cdd2f5f770874e838`.
- [x] Phase 6: exactly one full smoke accepted the fresh build. Report: `.claude/project-docs/evidence/world-model-physics-instrumentation/task-8-smoke.json`; output: `session-5-full-smoke/`.
- [x] Conditional re-pin: after acceptance, copied exactly `archive.sha256`, `novphy-physics-player-2019.4.41f2.tar.gz`, and `unity-build.log` into `sciencebirdsgames/physics-v1/`; all bind to `de59061350f78f79420d76ec33f1c506aa17c1cfc25d197cdd2f5f770874e838`.

### Errors Encountered
- The first mutation run was externally terminated at 120 seconds while mutation 2 was applied. The failure was recorded verbatim. The residue was a certain one-line EASY repair; exact restoration was verified before the one allowed retest, which passed 9/9.
- Three independent advisory/review agent sessions returned blank terminal results after reading evidence. Their failure was not treated as approval; the receipts were reconciled directly before the smoke.
- Two delegated smoke executors failed at the session layer before `run_smoke` created its output directory. Audits proved `NOT_CONSUMED`; the exact command was then run once directly and was not retried.

### Current Status
**`repin_complete`.** The accepted smoke recorded four collisions; the selected collision carried sorted unique `contact_ids` (`contact:361:-632:0:-646|world:static:-466:-466:0`) and finite non-negative `relative_speed` (`10.6197062`). Request identities used one capture id with sequence `1 -> 2`; listener binding remained stable; `recorder_refusals` was empty; port 2004 cleared; the temporary clone was removed; protected roots were unchanged. Publication remains a separate owner decision and is not authorized. Cohort collection did not occur; repository closure commits are finalized separately.
