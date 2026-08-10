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
