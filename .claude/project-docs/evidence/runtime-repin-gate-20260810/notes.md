# Notes: Unity Runtime Re-pin Gate

## Authority
- Recovery HEAD: `fc3e34c5f6115b55751874df21c3a9286bd09a5e`.
- Current verdict: `still_blocked`; no re-pin authorized. The verdict of record is `runtime-gate-result.md`; this file is the working log behind it. Earlier sections are historical receipts and are appended to, never rewritten.
- Tooling commit produced by this wave: `b121136` (`feat(physics-smoke): bind the physics listener to process-backed identity`).
- Final reviewed Unity source candidate: `97c4dd6f8b0a61b77b5b39a261ec0fddfd1abdba`.
- Prior live smoke used `7a2dd02`, observed request sequences `2 -> 3`, then failed because candidate process matching returned zero.

## Initial Observations
- Branch is `physics-unity-2019.4`; recovery HEAD and tree are exact.
- Port 2004 is unoccupied at baseline.
- No owned Unity player, Science Birds Java interface, or Xvnc process was found; the only textual matches were the inspection command itself and an unrelated zellij server named `unityhub`.
- Pre-existing dirty/untracked files match protected knowledge, logs, review artifacts, `.omo`, and staged package files; none may be incorporated or cleaned.

## Diagnostic and Implementation Findings
- Existing smoke launched Java without `--physics-port`, relying on Unity default 2004, and had no socket-inode/PID attribution. It only matched process/cwd after request-70, which explains the prior zero-candidate rejection.
- Added exact TCP listener discovery from `/proc/net/tcp`/`tcp6`, socket inode ownership from `/proc/<pid>/fd`, process-tree snapshotting, and candidate executable/cwd/assembly/archive checks.
- Added fail-closed request-70 identity and collision validators plus a listener-only mode. Focused smoke tests now pass 18/18.
- The implementation is not yet ready for live execution: the listener-only path still needs a careful source review and a bounded diagnostic run with the exact staged candidate.

## Review Findings and Residuals
- Pass 1: 2 BLOCKER, 8 MAJOR — all remediated with red/green tests.
- Pass 2: 0 BLOCKER, 3 MAJOR, 7 MINOR. Every MAJOR sat in the evidence-persistence/quarantine layer added by pass 1's remediations, not in the identity chain. The identity chain (inode -> pid -> exe/cwd -> maps device+inode+digest -> provenance bytes -> archive digest -> process tree) was traced and every ambiguity path terminates in a `ListenerBindingError`.
- Documented residual, not closable through `/proc`: a foreign-uid process holding a *dup* of the candidate's own listening fd (SCM_RIGHTS or a privilege-dropping fork) contributes no owner tuple while the inode stays attributed to the candidate. It is recorded on `require_full_attribution`, and accepted bindings now carry `uninspectable_pid_count` so a reader can tell a sighted scan from a blind one.
- Confirmed sound on challenge: a single absent `/proc/net` table is determinable (that address family cannot hold sockets); every table absent is not, and now rejects.

## Phase 5 Preconditions (verified read-only, 2026-08-11)
- `scripts/build_physics_player.sh:7` honours `NOVPHY_PHYSICS_STAGE`, so both isolated builds can publish outside the protected `sciencebirdsgames/physics-v1/` stage without editing the script.
- `scripts/package_physics_player.py:202` normalizes the tar (uid/gid 0, empty uname/gname, mtime 0, PAX) and compresses with `gzip -n -9`; `write_manifest` (line 139) embeds no timestamp. The packaging layer is therefore deterministic by construction, so any archive-digest drift would originate in Unity's own build output, not in packaging.
- `git_revision` refuses to package while tracked product source differs from HEAD, so the tooling commit is a hard precondition of both builds.
- Listener-only diagnostic surface: `--listener-only` (`scripts/smoke_physics_capture.py`) starts the display and jar, connects and configures the agent, binds the listener, and returns `diagnostic_accepted` without loading a level or shooting.

## Launch Topology Correction (2026-08-11)
An earlier note here claimed `--listener-only` "binds the listener and returns `diagnostic_accepted` without connecting the agent". That was false, and it made the bind unreachable in *both* modes.

- Measured with a bounded non-gameplay probe: with `game_playing_interface.jar` running and no agent connected, 60s of polling at 5s intervals reported `player_procs=0 port2004=0` at every sample. `engine.log` states it is waiting for the first agent to connect.
- The jar does not own the physics port. The Unity player it spawns on the agent handshake does. So `run_smoke` binding before `connect-agent` had no reachable success path — it could only ever time out, in the diagnostic and in the full smoke alike.
- Remediation: `run_smoke` now orders connect -> configure -> bind, then either stops (`listener_only`) or proceeds to level load and shot. `bridge.disconnect()` moved to a single `finally` covering both modes; it now trails artifact validation, which talks to no agent.
- Pinned by `test_the_agent_connects_before_the_physics_port_is_probed`, which records call order and asserts `["connect", "bind"]`. Verified red against the reverted ordering (`['bind', 'connect']`), then green after restore; source restored byte-identical to sha256 `7597d3a199d68537c9d1ab95326d7d63ac0b2f9ac32c1c03782ce8a35b0cfafb`.
- The two prior diagnostic timeouts (exit 124) are now explained by two independent causes: an undersized 300s ceiling against a measured 243.6s `active_data` protected receipt per pass, and this ordering defect, which would have timed out even with an adequate budget.

## Mapped-Anchor Correction (2026-08-11)
The first live diagnostic bound the port to the right process and then rejected it with `port listener owner pid=675476 has 0 mapped Assembly-CSharp.dll images`. A bounded non-gameplay probe settled which side was wrong.

- The live player maps 91 file-backed regions, six of them framework DLLs (`mscorlib`, `System`, `System.Core`, `System.Xml`, `Mono.Security`, `netstandard`), and **zero** `Assembly-CSharp.dll`. Nine mappings come from inside the verified clone, including `UnityPlayer.so 103:02 69088593` and `9001.x86_64 103:02 69088076`; `exe` and `cwd` both resolve inside that clone.
- So this is a **gate defect, not a candidate defect**: the running code is the staged code, and the required anchor was one Unity never produces. Mono mmaps class-library assemblies; Unity loads the *user* assembly through its own loader, so it is never file-backed.
- Correction: the mapped anchor is `UnityPlayer.so` — mapped from the candidate root, the native code that actually owns the socket, and digested in `provenance.json`. The assembly is still pinned, by device, inode, and digest read through the listener's own root (`/proc/<pid>/root/...`), which a candidate in another mount namespace cannot redirect to our copy.
- Chain after the change: socket inode -> pid -> exe/cwd -> mapped `UnityPlayer.so` device+inode+digest -> provenance runtime digest -> assembly device+inode+digest via process root -> provenance assembly digest -> provenance bytes -> staged archive digest -> launched process tree. That is one anchor stronger than before, not weaker: the old chain had no digest over any *mapped* code.
- **Deviation flagged:** the mission names "mapped `Assembly-CSharp.dll`" as a binding input. That binding is not obtainable from this runtime. The substitute is taken deliberately and recorded rather than passed over in silence.

## Corrected Binding Verified Live (2026-08-11)
The corrected gate was re-run against the live staged candidate through the same bounded listener-only path. It returned `status: diagnostic_accepted`, `phase: listener-bound`, `error: null` — no gameplay, no level load, no shot, and the single permitted full smoke untouched.

- Receipt: `listener-diagnostic-corrected.json`.
- Binding: socket inode `209799923` on `0100007F:07D4` (the only listening socket on the port, fully attributed) -> pid `700284` -> exe `/tmp/novphy-physics-smoke-ei9aq7tc/player/9001.x86_64`, cwd the clone root -> mapped `UnityPlayer.so` `103:02` inode `69088604` sha256 `53b0b8d1d21031c097721b1bf10bf8cd23c34663f871d606e28bd276bd171c28` -> provenance runtime digest, equal -> `Assembly-CSharp.dll` read through `/proc/700284/root/...`, `103:02` inode `69088102`, sha256 `051aac49739a3b09628f03e9f0fc7c0ca5e17b42a66dc7f1707542fad8d73622` -> provenance assembly digest, equal -> provenance bytes sha256 `3eae856a912c53fe6f1af41243551b504603bf7309e57d1c6c345a63a11bfcf9` -> staged archive `429cac1d748bed417b917d2838dc203d090668977dc8e56f5bac9a80ea95f2de` -> launched tree (`launched_pid` 700231, size 2, owner inside it).
- Diagnosis carried alongside the accept, not instead of it: `host_uninspectable_pid_count = 785`. The scan was partial by host policy; the invariant that closes the impostor hole is inode reconciliation, and it held.
- Teardown: `agent: disconnected`, `engine: pid=700231:exit=143`, `xvfb: pid=700215:exit=0`, `physics_listener_inodes_after: []`, `physics_port_clear: true`, `temporary_clone_removed: true`, `protected_unchanged: true`, no cleanup failures. The grace poll added this pass is what makes `physics_port_clear` true rather than racing the grandchild's exit.
- This closes the gate defect above. It does **not** unblock the wave; the next section does not depend on it.

## Staged Candidate Cannot Pass the Final Smoke (2026-08-11)
Re-verified against current on-disk state, not taken from memory: `sciencebirdsgames/physics-v1/archive.sha256` is `429cac1d748bed417b917d2838dc203d090668977dc8e56f5bac9a80ea95f2de`, and the probe confirms the same archive is what runs.

`task-7-collection-blocker.json` records that exact archive's player emitting `collision` events with an empty payload `{}` while the frozen contract requires `contact_ids` and `relative_speed`, so `validate_physics_shot_artifact` rejects every shot containing a collision. Prior instrumentation passed only because the accepted shot contained no collision at all.

The mission's acceptance requires at least one genuine collision with non-empty sorted unique `contact_ids` and finite non-negative `relative_speed`. So the single permitted full smoke is guaranteed to reject, whatever the listener does. **It was not run**: with no retry available, spending it to re-confirm a documented defect would leave nothing for a run after the fix. The fix is player-side C#, and reaching it requires rebuild -> republish -> re-smoke, which is a re-pin this wave may not perform.

## Review Pass 6 and Its Remediation (2026-08-11)
0 BLOCKER, 7 MAJOR, ~17 MINOR against the exact uncommitted diff. Machine-readable record: `review-remediation.json`. Three things are worth stating in prose.

- **The review validated the identity chain rather than only faulting it.** The `UnityPlayer.so` anchor is genuinely strong (a mapped `device:inode` is namespace-global and is compared against a pre-launch `stat()`); removing the old `assembly_path` equality was correct because it compared a value to itself; and reading through `/proc/<pid>/root/` is strictly stronger than a host-path read, degenerating to it in the same-namespace case.
- **Two mutations survived the whole suite at review time** — `_process_view` replaced by the identity function, and `_process_tree` replaced by a direct-children-only walk. Both are real coverage gaps, and the second is not academic: the Unity player is a grandchild, so a direct-children-only tree would place the real listener outside the launched tree. Both are now red (`mutation_check.py`).
- **The one genuinely new attack surface was the launch environment.** Every digest, the archive, and tree membership can all pass while `LD_PRELOAD` loads native code that is in none of them and forges exactly the collision and capture payloads this gate exists to measure. `launch_environment` now refuses six interposition variables, strips `LD_LIBRARY_PATH` (refusing it would make the gate unrunnable on this CUDA host), and records both.

Also hardened beyond the review: a non-loopback physics bind now rejects instead of being recorded and accepted; any `ListenerObservation` field neither compared nor declared unchecked fails the bind closed; an exception outside the handled tuple can no longer unwind through the outer `finally` carrying `status: accepted`; and `--listener-only` writes to its own report path so a diagnostic cannot exit 0 at the designated smoke marker.

One remediation was tried and reverted, deliberately: treating `FileNotFoundError` from a `/proc/<pid>/fd` readlink as a blind spot. `iterdir()` holds its own descriptor, which is closed by the time the readlink loop runs, so *every* process on the host — including the test process — landed in the blind-spot census. The reason is recorded in the source beside the `continue`, so it is not re-attempted.

Verification: focused suite 74/74 with `ResourceWarning` promoted to error; five adjacent physics tooling suites 60/60; eight mutations each turn the suite red, with the source restored byte-identical to `f61ccfa52ef5bed5e156e8480f32f407c13286276df00d8ec676e9671fc84978`.

## Evidence Paths
- Mapped-anchor finding: `finding-unity-does-not-map-the-user-assembly.json`.
- Terminal blocker: `finding-collision-payload-blocks-the-final-smoke.json`.
- Live map table: `player-mapping-probe.json` (and `player-mapping-probe-wrong-port.json`, the run that drove a random port through `--physics-port` and saw no listener).
- First live diagnostic (pre-correction, rejected): `listener-diagnostic.json`.
- Second live diagnostic (post-correction, accepted): `listener-diagnostic-corrected.json`.
- Unity EditMode: `phase-4b-unity-editmode.json`, `unity-editmode/`.
- Review pass 6 and its remediation: `review-remediation.json`; mutation harness `mutation_check.py`.
- Deterministic build comparison: not run — the tooling commit is its precondition and the wave ended at `still_blocked` before it.
- Final smoke: deliberately not run; see above.
