# #132 decisions log

## Decision P-license (2026-10-24): per-artifact licenses of the E1 agents and their transport, verified before any execution

**Trigger.** Rule P-license and this ticket's licensing note: the MIT tag on phy-q/benchmark does not by itself grant rights to the bundled JARs. Each artifact's provenance and license must be verified before any agent runs. No agent has been executed.

**What was verified.** Clones: phy-q/benchmark @ `514491c5`, phy-q/NovPhy @ `0da1258c`. The JAR contents were inspected (`unzip -l`, `META-INF/MANIFEST.MF`, the bundled license and notice files). The upstream source repositories' LICENSE files were fetched. The full record, with sha256 hashes for every artifact and license file, is in `.local-artifacts/issue-132-aibirds-agents-v1/licenses.json`. Local copies of the license texts are in `licenses/`.

| Artifact (sha256 prefix) | Origin | License evidence | Effective license | Permission |
| --- | --- | --- | --- | --- |
| `data_lab_agent_v5.jar` (`c11ddd6e`) | Datalab Birds (AIBIRDS 2014/15). Recompiled for the Science Birds interface (packages `ab.*`, `dl.*`, `external.*`; classes dated 2021-07-13). Byte-identical to the phy-q/benchmark copy. | No agent license inside the JAR; it carries only Apache notices for bundled libraries. The source repo gitlab.fit.cvut.cz/spetlrad/angry-birds-ai (`c21ae70c`, "License files.") LICENSE says CC BY-NC-SA 3.0, under the AIBIRDS framework copyright (Ge, Gould, Renz et al.). | **CC BY-NC-SA 3.0** | non-commercial research evaluation only; run unmodified; never copied or redistributed by #127/#132 |
| `eagle_wings_agent_v6.jar` (`5759b9c9`) | Eagle's Wing (team HeartyTian, AIBIRDS 2017). Recompiled for Science Birds (contains `Eagle Wings.iml`; classes dated 2021-07-13). Byte-identical to the phy-q/benchmark copy. | No agent license inside the JAR; it carries only third-party notices. The standalone repo heartyguy/AI-AngryBird-Eagle-Wing (`593bd151`) LICENSE says CC BY-NC-SA 3.0. | **CC BY-NC-SA 3.0** | same as Datalab |
| `PigShooter.py` (`3fc0c387`) and its Phy-Q Python client | Phy-Q authors | phy-q/benchmark LICENSE: MIT, © 2021 Cheng-Xue | **MIT** | vendored verbatim with the upstream LICENSE (the upstream closure, because our repo's `Client/` and `SBEnvironment/` carry local edits) |
| `game_playing_interface.jar` (`b8ebda18`) | Science Birds Basic Game Playing Software, the AIBIRDS competition server (MANIFEST: Built-By aibirds4, JDK 13.0.2, Main-Class `server.ABServer`, built 2020-11-05). The same file is in the phy-q/benchmark Linux.zip. Not tracked in this repo. | gitlab.com/aibirds/sciencebirdsframework `License/LICENSE`: **AGPL-3.0-or-later** (since its first commit, 2019-12-23) | AGPL-3.0-or-later | Run unmodified as a separate process; AGPL places no condition on running the program. Never vendored, linked, modified or conveyed. Every NovPhy engine run since #46 already uses it as its transport (`build_physics_player.sh` copies it into the #104 player). D5's "GPL code is skipped" applies to vendored arm code, not to this transport. |
| Engine (`sciencebirdsgames/Linux.zip`, `c45c8b44`) | Byte-identical to phy-q/NovPhy@0da1258. The #104 player is our own build of `tasks/task_template_designer`. | `task_template_designer/License/LICENSE`: GPL-3.0-or-later (Ferreira; Renz et al.) | GPL-3.0-or-later | existing infrastructure, unchanged |

**Options considered.**
- (a) Treat the repo's MIT tag as covering the JARs. Rejected: the tag does not relicense third-party agents.
- (b) Drop both Java agents as "license forbids use". Rejected: CC BY-NC-SA 3.0 permits non-commercial use with attribution.
- (c) Use the JARs unmodified for non-commercial research evaluation only, never redistributed, with attribution.

**Choice: (c).**
- Datalab and Eagle's Wing run unmodified for non-commercial research evaluation only.
- No JAR is copied into the repo by #127 or #132.
- The JARs are attributed to their teams and to the AIBIRDS framework authors.
- PigShooter's MIT code is vendored with its LICENSE.
- The AGPL transport is used unmodified, as in every earlier engine run.

**Observation for the owner (no #132 action).** `data_lab_agent_v5.jar` and `eagle_wings_agent_v6.jar` have been tracked in this public repo since `68480cd0` (2026-06-05), before #127. CC BY-NC-SA allows non-commercial sharing with attribution under the same license, so this is not a violation in itself. It does conflict with the program's "never redistribute" policy. #132 adds no copy.

## Decision P-engine (2026-10-24): E1 engine path (choice in the stock engine, verdict in the #102 channel) and the 20-level smoke design

**Trigger.** Rule P-engine requires a smoke before the E1 run. The smoke measures determinism and the engine seconds per execution. The engine path the agents use has to be fixed before the smoke runs. This decision fixes the path and the smoke design. It reads no evaluation-split content.

**Facts established by single-level probes** (fit level `issue-109-sc-fit-type010103-001`, scratch directory, not part of any result):
- **Our engine.** Our `sciencebirdsgames/Linux.zip` is byte-identical to phy-q/NovPhy@0da1258 (Unity 2019.3.4f1). The #104 player is our own instrumented build of the same project (2019.4.41f2, physics-capture wired into the pipeline). Agents cannot connect to the #104 player outside `capture_pipeline_v2`.
- **Startup hangs and their fixes.** Launched by the jar, the stock player hangs at "Desktop 0x0" until it gets the #104 wrapper's screen arguments (`scripts/9001-player-wrapper.sh`). It also hangs in audio init while a stale user PulseAudio server is reachable. Setting `PULSE_SERVER` to a dead socket fixes this. It affects audio only.
- **Java agent port.** Datalab and Eagle's Wing hard-code `localhost:2004`. Each worker gives the JVM its own hosts file (`-Djdk.net.hosts.file`), mapping `localhost` to a private loopback address. Parallel workers therefore never share a port, and the JARs are not modified.
- **PigShooter level load.** On this build, the Phy-Q environment's `load_level(1)` (request 51) sent from the main menu raises `IndexOutOfRangeException` in `SymbolicGameState.GetGTJson` and never answers (engine log). The runtime holds exactly one level, so the PigShooter driver sends that load as request 53 (load next available level). That is the request the Java agents and the #104 pipeline already use. PigShooter's policy and environment code are unchanged.
- **Shot transfer is exact.**
  - The game logs every received shot (`DragDetails`).
  - In stage B, the #104 player received exactly the agent's wire release point in 2/2 probes: Datalab (−232, 439) and Eagle's Wing (−140, 547, tap 2905).
  - The ground-truth slingshot used as the anchor was identical in two consecutive requests, in normal and in `--dev` mode.
  - The release time never reaches the engine (#95 `release_inertness`).
- **Verdicts on the probe.** Stage-A native status and stage-B `pig_removed` agreed (Fail/false, Pass/true). Stage-B wall time was 136–155 s per execution with 2 workers. PigShooter shot (36, 434), Fail.

**Options considered.**
- (a) Agents play the #104 player directly. Not possible: the player's capture barrier needs `capture_pipeline_v2`.
- (b) Agents play the stock engine, and their native level-clear is the result. This uses a different verdict channel and a different build from the #102 rows they are compared with.
- (c) Two stages: the agent chooses its shot in its native environment (stock engine and jar), and that exact shot is executed on the frozen #104 player by the #102 executor. The verdict is `capture_pipeline_v2.branch_outcome` (pig removal, #85 channel), as for every #102 row.

**Choice: (c).**
- **Stage A.** Each attempt runs in a fresh runtime copy (reset to start), on a private Xvnc display with the #104 Mesa llvmpipe.
  - The config.xml uses Phy-Q's attributes, with the level listed twice as Phy-Q does.
  - Jar flags follow Phy-Q: none for the Java agents, `--dev` for PigShooter.
  - A relay records the agent's first shot request verbatim and injects the anchor ground truth just before it. A second shot is never forwarded.
  - The engine's evaluation CSV gives the native status.
- **Stage B.** The recorded shot is converted to the slingshot-relative drag and run once per distinct (level, shot) through `run_closed_loop_evaluation.prepare_engine` / `engine_limits` and `capture_pipeline_v2.run_campaign`. Identical shots on a level share one execution, the #102 sharing rule.
- **Reporting.** The stage-B `pig_removed` is the E1 result. The stage-A native status is reported next to it as a cross-check.
- **Not counted as a pass.** A polar shot, a missing shot, or a typed engine failure is never retried and counts as not passed.

**Smoke design.**
- **Levels:** 20 fit-split levels, ordinals 1–10 of type010103 and of type010105, never evaluation levels (#102's smoke also used fit levels).
- **Attempts:** 3 agents × 2 resets each in stage A, then stage B for every distinct recorded shot. At most 8 CPU workers.
- **Determinism rule (frozen here):**
  - An agent is deterministic iff, on every smoke level where both resets shot, the two wire shots (code, fx, fy, t1, t2) are identical. It then runs pass@1 (one attempt per N1 level).
  - Otherwise it runs 3 attempts per level (pass@1 and pass@3).
- **Measurements:** seconds per execution for stage A and stage B, stage-A/B verdict agreement, and the transfer check (stage-B `DragDetails` equals the wire shot).
- **Projection:** posted as a separate P-engine decision with the smoke numbers, before any N1 execution.

**Reason.** Validity, under the decision priority:
- The agents act natively. The decision state and the jar protocol are exactly what Phy-Q ran them on.
- The verdict channel, the build and the executor are the ones behind the #102 rows the native-action table sits next to.

## Decision P-engine (2026-10-25): E1 smoke result, attempts per agent and the N1 projection against the 190 worker-hour allowance

**Trigger.** Rule P-engine: the 20-level smoke measures determinism and per-execution cost, and its projection is decided before any N1 execution. The smoke design, the engine path and the determinism rule were frozen in the P-engine decision of 2026-10-24 (issuecomment-6382714673). The smoke used fit levels only; no evaluation-split content was read.

**Smoke (`.local-artifacts/issue-132-aibirds-agents-v1/smoke/smoke.json`).**
- **Coverage:** 20 fit levels × 3 agents × 2 resets = 120 stage-A attempts, all complete with a shot (0 polar, 0 typed failures). 99 distinct (level, shot) stage-B executions on the #104 player, all complete.
- **Transfer:** every stage-B execution received exactly the agent's wire release point (120/120, the game's own `DragDetails`). The slingshot anchor was noise-free in 120/120 attempts.
- **Verdict agreement:** stage-A native level status and stage-B `pig_removed` agree in 120/120 attempts. For orientation only (fit levels, not a result): Datalab passed 0.175, Eagle's Wing 0.125, PigShooter 0.15.
- **Determinism** (an agent is deterministic iff both resets give the identical wire shot on every compared level):

  | Agent | Levels compared | Identical shot after reset | Verdict | Attempts per N1 level |
  | --- | --- | --- | --- | --- |
  | Datalab | 20 | 10 | varies | 3 (pass@1 and pass@3) |
  | Eagle's Wing | 20 | 0 | varies | 3 |
  | PigShooter | 20 | 11 | varies (unseeded `random` in target and trajectory choice) | 3 |

- **Engine seconds per execution** (8 workers, host load 40–77):
  - Stage A (agent chooses, stock engine): Datalab 28.8 s, Eagle's Wing 25.7 s, PigShooter 51.3 s.
  - Stage B (#102 executor, #104 player): 178.4 s per distinct execution.
  - Sharing: identical shots on a level share one stage-B execution (#102 rule, frozen 2026-10-24). In the smoke this gave 0.825 executions per attempt (99/120).

**Projection for N1** (400 levels × 3 agents × 3 attempts = 3,600 attempts):

| | Datalab | Eagle's Wing | PigShooter | Total |
| --- | --- | --- | --- | --- |
| With the smoke's sharing rate (0.825) | 58.6 h | 57.6 h | 66.1 h | **182.4 worker-h** |
| Upper bound, no sharing | 69.1 h | 68.0 h | 76.6 h | 213.6 worker-h |

#132 has already spent **6.19 worker-h**: the smoke plus the pre-smoke single-level probes, counted against the shared allowance. Projected E1 total: 188.6 of 190 worker-h, leaving about **1.4 worker-h** for A3. With three attempts per level instead of the smoke's two, more shots repeat, so sharing is expected to be a little higher [INFERENCE].

**Options considered.**
- (a) Run pass@1 only for every agent. Rejected: the frozen rule gives agents whose shots vary three attempts.
- (b) Take the verdict from stage A to save the stage-B cost. Rejected: the verdict channel was frozen on 2026-10-24, and the native-action rows must use the #102 verdict channel.
- (c) Run the frozen protocol with an ordering that keeps any shortfall unbiased and a hard stop at the allowance.

**Choice: (c).**
- **Order:** attempt 1 for every level and agent first, then attempt 2, then attempt 3. Within each attempt, levels follow a fixed pseudo-random order (sha256 of `132:<level>`), in chunks of 20 levels.
- **Stop rule:** a chunk starts only if the engine worker-hours spent so far, plus that chunk's projected cost, stay within 190. Otherwise the run stops, typed `engine_budget`.
- **If the stop fires:** pass@1 is complete for every agent, and pass@3 is reported on the completed random subset of levels, with the subset disclosed.
- **Engine window:** engine processes run only 07:05–17:30 Melbourne (owner rule, 2026-10-25). The runner starts an attempt only if its worst case finishes inside the window, and exits 75 to resume the next day.
- **Workers:** at most 8.
- **Frozen inputs:** written to `n1/run-plan.json` before the first N1 execution.
- **Pass:** stage-B `pig_removed`. Missing or polar shots and typed engine failures count as not passed and are never retried.
- **Reporting:** the results go in the separately labelled native-action table next to #102's rows. They feed no disposition and no program claim.

**Consequence for A3** (decided on #128): the projected remainder (about 1.4 worker-h) is below A3's native-task fidelity projection (about 4 worker-h per variant). A3's fidelity therefore runs only if E1's realised cost leaves enough; otherwise A3 stays tokens-only, typed `engine_budget`.

**Reason.** This applies the frozen P-engine rule (varying agents get three attempts) without weakening the agents and without changing the verdict channel. The attempt-major, randomised order means a budget stop can only shrink the pass@3 sample, not bias it.
