# Issue #132: AIBIRDS heuristic agents acting natively on the #102 N1 levels (#127 E1, native-action table)

Parent: #127 (external baselines program). Ticket: [#132](https://github.com/Sino-Huang/NovPhy/issues/132). E1 answers
the program's question Q5: how do AIBIRDS champion agents, choosing their own continuous shot, compare with our
closed-loop rows? The answer is **descriptive only**. It sits in a separately labelled native-action table next to the
#102 N1 rows. It is not the #102 grid protocol, carries no δ-rule token, and enters no disposition and no program claim.
The #102, #104 and #109 records and stores are read-only inputs. Nothing here edits the ICLR 2026 submission.

| | |
| --- | --- |
| Agents | Datalab (`sciencebirdsagents/HeuristicAgents/data_lab_agent_v5.jar`), Eagle's Wing (`eagle_wings_agent_v6.jar`), PigShooter (Phy-Q `HeuristicAgents/PigShooter.py`), as shipped by [phy-q/benchmark](https://github.com/phy-q/benchmark) @ `514491c` |
| Runner | `scripts/run_issue_132_aibirds_agents.py` (levels / smoke / project / run / publish / validate; `--pigshooter-driver` is internal) |
| Modules | `world_model/baselines/phyq/vendor/` (phy-q/benchmark @ `514491c`, MIT, 44 files verbatim; the only edits make imports package-relative; per-file sha256 in `VENDOR.json`). E1 uses the PigShooter closure; the same tree serves #128 A3 |
| Tests | `tests/test_issue_132_aibirds_agents.py` (23 tests, CPU; the publication tests use synthetic records only) |
| Artifacts | `.local-artifacts/issue-132-aibirds-agents-v1/`: `licenses.json` + `licenses/`, `decisions.md`, `smoke/smoke.json`, `n1/run-plan.json`, `summary.json`, `compute.json`, `findings.md`, `records-manifest.json`, `envs/`. Stage-A attempt records, stage-B engine attempts, verdicts and the stock-player copy stay local; they are hashed in `records-manifest.json` |
| Freeze | `n1/run-plan.json` was frozen 2026-10-24T19:13:06Z (sha256 `ad5d89e1…`), after the smoke and the P-engine projection decision and before any N1 execution. It fixes attempts per agent, per-attempt costs, the level order and the allowance-stop rule |
| N1 run | Four daytime engine windows, 25–28 Oct 2026 (07:05–17:30 Melbourne), ≤ 8 workers. Exit 0 at 14:00 AEDT on Wed 28 Oct. All 3,600 planned attempts were recorded; no allowance stop |
| Validation | `python -u scripts/run_issue_132_aibirds_agents.py --validate` **exit 0**: the records manifest re-hashed (sha256 `9eada796…`), `summary.json` / `compute.json` / `findings.md` re-derived byte for byte, and 12 stage-B verdicts recomputed from the engine results. Publication commit `b18c954f` |

GPU time is zero. Engine time is CPU only and counted against the 190 CPU worker-hours that E1 shares with A3's
native-task fidelity (rule P-engine, owner decision D3). Intervals are level-clustered percentile bootstraps (10 000
draws, PCG64 7201, #102's own `p94.bootstrap`) and are DESCRIPTIVE.

## Outcome

Native-action table on the 400 #102 N1 levels, copied exactly from `findings.md`. #102 rows are as published; #102
cells are level × seed, E1 cells are levels.

| Row | pass@1 | pass@3 | levels (pass@1 / pass@3) |
| --- | --- | --- | --- |
| E1 Datalab | 0.273 [0.230, 0.318] | 0.307 [0.263, 0.352] | 400 / 400 |
| E1 Eagle's Wing | 0.110 [0.080, 0.142] | 0.233 [0.193, 0.275] | 400 / 400 |
| E1 PigShooter | 0.120 [0.090, 0.152] | 0.217 [0.177, 0.258] | 400 / 400 |
| #102 random (as published) | 0.068 [0.053, 0.082] | 0.198 [0.172, 0.223] | 400 levels x 3 seeds |
| #102 prior (as published) | 0.140 [0.107, 0.175] | 0.323 [0.278, 0.370] | 400 levels x 3 seeds |
| #102 U (F-15-macro) (as published) | 0.151 [0.131, 0.172] | 0.424 [0.384, 0.465] | 400 levels x 3 seeds |
| #102 C* (C-F-15) (as published) | 0.148 [0.126, 0.172] | 0.466 [0.427, 0.505] | 400 levels x 3 seeds |
| #102 J (J-1e-04) (as published) | 0.123 [0.098, 0.150] | 0.394 [0.352, 0.438] | 400 levels x 3 seeds |
| #102 J_dec (J-dec-1e-01) (as published) | 0.139 [0.117, 0.162] | 0.445 [0.403, 0.487] | 400 levels x 3 seeds |
| #102 oracle (as published) | 0.775 [0.733, 0.815] | 0.775 [0.733, 0.815] | 400 levels x 3 seeds |

pass@2 (in `summary.json`): Datalab 0.295 [0.250, 0.340], Eagle's Wing 0.180 [0.142, 0.217], PigShooter
0.180 [0.142, 0.217].

Reading (descriptive; no token, no claim):
- **Datalab's first shot.** Datalab's pass@1 point estimate (0.273) is the highest of the rows above apart from the
  oracle. Its gain from the second and third attempts is small (+0.035 from pass@1 to pass@3). Mean attempts to pass,
  among passed levels, is 1.15.
- **Eagle's Wing and PigShooter.** Their pass@1 point estimates (0.110, 0.120) lie between #102's random and prior
  rows. Their pass@3 (0.233, 0.217) lies between #102's random and prior pass@3.
- **Why the two protocols differ.** Every #102 model row excludes its failed launches from a fixed 16-launch grid
  between shots. The agents choose afresh after every reset and can repeat a failed shot. The pass@3 columns therefore
  measure different retry mechanisms, and the E1 and #102 rows are not a paired contrast.

### Per-agent outcomes (400 episodes each)

| Agent | passed | failed (3 attempts, no pass) | typed failure | attempts counted | mean attempts to pass | agent-side status vs stage-B pig removal |
| --- | --- | --- | --- | --- | --- | --- |
| Datalab | 123 | 272 | 5 (no transferable shot) | 973 | 1.154 | 1.0 |
| Eagle's Wing | 93 | 299 | 8 (no transferable shot) | 1084 | 1.753 | 1.0 |
| PigShooter | 87 | 308 | 5 (no transferable shot) | 1080 | 1.621 | 1.0 |

- **Typed failures.** Every typed failure is an attempt that produced no transferable shot. No stage-B execution
  failed.
- **Verdict agreement.** The agent-side engine level status (stage A, stock build) and the stage-B pig removal (#104
  player, #85 channel) agree on every counted attempt.

## Question and protocol

### Agents and licences (rule P-license; decision posted before any execution)

Each artifact's provenance and licence were verified before anything ran
([decision](https://github.com/Sino-Huang/NovPhy/issues/132#issuecomment-6382271495); full record with sha256s in
`licenses.json`):

| Artifact (sha256 prefix) | Origin | Effective licence | Use |
| --- | --- | --- | --- |
| `data_lab_agent_v5.jar` (`c11ddd6e`) | Datalab Birds (AIBIRDS 2014/15), recompiled by the Phy-Q authors for the Science Birds interface; byte-identical to the phy-q/benchmark copy | CC BY-NC-SA 3.0 (source repo gitlab.fit.cvut.cz/spetlrad/angry-birds-ai; no agent licence inside the JAR) | non-commercial research evaluation only, run unmodified, never copied or redistributed by #127/#132 |
| `eagle_wings_agent_v6.jar` (`5759b9c9`) | Eagle's Wing (team HeartyTian, AIBIRDS 2017), recompiled the same way | CC BY-NC-SA 3.0 (standalone repo heartyguy/AI-AngryBird-Eagle-Wing) | as Datalab |
| `PigShooter.py` and the Phy-Q Python client | Phy-Q authors | MIT (phy-q/benchmark) | vendored verbatim with the upstream LICENSE |
| `game_playing_interface.jar` (`b8ebda18`) | Science Birds basic game-playing software (AIBIRDS competition server) | AGPL-3.0-or-later (gitlab.com/aibirds/sciencebirdsframework) | run unmodified as a separate process, as in every NovPhy engine run since #46; never vendored or redistributed |
| Engine (`sciencebirdsgames/Linux.zip`, `c45c8b44`) | Science Birds as modified by the AIBIRDS/Phy-Q/NovPhy team; byte-identical to phy-q/NovPhy @ `0da1258` | GPL-3.0-or-later (Unity project licence) | existing infrastructure, unchanged |

The phy-q/benchmark MIT tag covers the Phy-Q authors' own code. It does not relicense the recompiled third-party
agents.

**Owner attention needed.** `data_lab_agent_v5.jar` and `eagle_wings_agent_v6.jar` have been tracked in this
public repository since commit `68480cd0` (2026-06-05), before #127. CC BY-NC-SA 3.0 permits non-commercial sharing
with attribution under the same licence, so this is not a violation in itself. It does conflict with the program's
"never redistribute" policy. #132 adds no copy. Whether the two JARs stay tracked is the owner's decision.

### Engine path and verdict channel ([decision](https://github.com/Sino-Huang/NovPhy/issues/132#issuecomment-6382714673))

Agents cannot connect to the #104 player outside `capture_pipeline_v2`. Each execution therefore has two stages:

1. **Stage A (the agent chooses, natively).**
   - Every attempt gets a fresh copy of the stock Science Birds build and the stock `game_playing_interface.jar` (reset
     to the level start), on a private Xvnc display with the #104 Mesa llvmpipe.
   - The level's `config.xml` uses Phy-Q's own attributes, with the level listed twice as Phy-Q does. Jar flags follow
     Phy-Q: none for the Java agents, `--dev` for PigShooter.
   - A relay between agent and jar records the agent's first shot request verbatim (code, fx, fy, t1, t2). It reads
     the slingshot from two ground-truth requests just before forwarding the shot, and never forwards a second shot.
   - The engine's own evaluation CSV gives the agent-side level status (a cross-check).
2. **Stage B (the verdict, as #102).**
   - The recorded shot, converted to a slingshot-relative drag with the stage-A anchor, is executed once on the frozen
     #104 player by the #102 executor (`run_closed_loop_evaluation.prepare_engine` / `engine_limits`,
     `capture_pipeline_v2.run_campaign`).
   - Pass = `capture_pipeline_v2.branch_outcome` pig removal (#85 channel), exactly as every #102 row.
   - Identical shots on a level share one execution, the #102 sharing rule.

Adaptations needed to run the unmodified agents on this host (all transport-level, none in an agent's policy):
- **Player startup.** The stock player hangs at startup unless launched with the #104 wrapper's screen arguments
  (`scripts/9001-player-wrapper.sh`). It also hangs in audio initialisation while a stale user PulseAudio server is
  reachable. `PULSE_SERVER` is pointed at a dead socket, which affects audio only.
- **Java agent port.** Datalab and Eagle's Wing hard-code `localhost:2004`. Each worker gives the JVM its own hosts
  file (`-Djdk.net.hosts.file`), which maps `localhost` to a private loopback address. The JARs are unmodified.
- **PigShooter level load.** On the NovPhy build, the Phy-Q environment's `load_level(1)` (request 51) sent from the
  main menu raises `IndexOutOfRangeException` in `SymbolicGameState.GetGTJson` and never answers. The runtime holds
  exactly one level, so the driver sends that load as request 53 (load next available level). That is the request the
  Java agents and the #104 pipeline use. PigShooter's policy and environment loop are Phy-Q's own
  (`MultiAgentTestOnly.py` / `HeuristicAgentThread` flow, `reward_type='passing'`, speed 100).
- **Shot transfer is exact.** In the smoke, the #104 player received exactly the agent's wire release point in 120/120
  attempts (checked against the game's own `DragDetails` log). The anchor was noise-free in 120/120.

### Smoke, determinism and attempts ([decision](https://github.com/Sino-Huang/NovPhy/issues/132#issuecomment-6388390880))

- **Smoke scope.** 20 fit-split levels (ordinals 1–10 of type010103 and of type010105; never an evaluation level) ×
  3 agents × 2 resets: 120 stage-A attempts, all complete with a shot. 99 distinct stage-B executions, all complete.
  Agent-side status and stage-B pig removal agreed 120/120.
- **Determinism.** The rule was frozen before the smoke: an agent is deterministic iff both resets give the identical
  wire shot on every compared level.

  | Agent | identical after reset | verdict | attempts per N1 level |
  | --- | --- | --- | --- |
  | Datalab | 10 / 20 | varies | 3 |
  | Eagle's Wing | 0 / 20 | varies | 3 |
  | PigShooter | 11 / 20 (unseeded `random` in its target and trajectory choice) | varies | 3 |

- **Projection.** With the smoke's per-execution costs and sharing rate (0.825), the projection was 182.4 worker-h,
  plus 6.19 already spent (smoke and probes). That left about 1.4 of 190 for A3.
- **Run order and stop rule** (frozen in `n1/run-plan.json`):
  - attempt 1 for every level and agent, then attempt 2, then attempt 3;
  - within an attempt, chunks of 20 levels in a fixed pseudo-random order (sha256 of `132:<level>`);
  - a chunk starts only if the spent worker-hours plus its projection stay within 190, else the run stops typed
    `engine_budget`.

  The stop never fired.

### pass@k definition

- **Episode rule.** #102's rule (`run_closed_loop_evaluation.episodes` / `episode_view`). Attempt k is the agent's
  k-th reset to the level start. The episode ends at the first pass or at the first typed failure (no transferable
  shot, or a stage-B execution that did not complete). Attempts after a typed failure are not counted.
- **pass@k** = 1 iff the episode passed within its first k attempts.
- **Bootstrap.** #102's own `p94.bootstrap`, level-clustered, one cell per level (no model seeds).
- **Deterministic agents.** An agent the smoke had found deterministic would have had one attempt per level and a null
  pass@3. None was.

## Compute

| Item | Worker-seconds | Worker-hours |
| --- | --- | --- |
| N1 stage A (3,600 attempts) | 121,109 | 33.64 |
| N1 stage B (2,673 executions; 0.7425 per attempt) | 409,783 | 113.83 |
| Smoke stage A + B | 21,887 | 6.08 |
| Pre-smoke probes | 387 | 0.11 |
| **Total (counted against the 190 allowance)** | **553,166** | **153.657** |

- **Per-attempt stage A.** Datalab 27.7 s, Eagle's Wing 24.7 s, PigShooter 48.5 s. The agents' own decision times
  were 17.0 s, 14.2 s and 38.0 s.
- **Stage B.** 153.3 s per execution.
- **Against the projection.** Realised cost is 28.7 worker-h below the projection, because more shots repeated (0.7425
  vs 0.825 executions per attempt) and stage B ran faster (153 vs 178 s) on a less loaded host.
- **GPU.** 0 seconds.
- **Remainder for A3.** R = 190 − 153.657 = 36.3 worker-h. Under the #128 P-engine rule (R ≥ 5.0), A3
  Relational-Symbolic's native-task fidelity run is funded
  ([#128 decision](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6464530188)).

## Deviations and disclosures

- **Engine window (owner rule, 2026-10-25).** After a night-time out-of-memory event on the host, during which the
  smoke's final bookkeeping step was cut short, engine processes were restricted to 07:05–17:30 Melbourne. The smoke's
  bookkeeping was completed afterwards from its retained records, with no new engine execution. The runner checks the
  window before every stage-A attempt (≥ 12 min left) and every stage-B round (≥ 32 min left), and exits 75 to resume.
- **Launcher octal bug (harmless).** `n1/daily-launcher.sh` compared the Melbourne time as `[[ $now -lt 0705 ]]`.
  Bash parses values such as `0039` or `0618` as invalid octal, so the launcher sometimes started `--run` outside the
  window: 9 launches at night or before 07:05. Each time the runner's own window check exited 75 within 70 s,
  before any engine process started (`n1/logs/launcher.log`). No engine work ran outside the window. The fix
  (`10#$now`) was not applied to the in-flight launcher.
- **Typed failures.** The 18 attempts without a transferable shot (5 / 8 / 5) count as not passed and were never
  retried, as frozen.
- **Different protocol from #102.** The agents act on the stock build's decision state through their native
  interface. The verdict uses the #104 player and the #102 channel. An agent's retries cannot exclude its earlier
  failed shots. The E1 rows are therefore a separately labelled native-action table, not a paired contrast with any
  #102 row.
- **Evaluation-split access.** The N1 levels were played only by the frozen N1 run. The smoke and every probe used
  fit-split levels. The publication code was developed and tested on synthetic records and the smoke's fit-level
  records.

## Decisions (posted on #132 before the step each governs; also in `decisions.md`)

- P-license, 2026-10-24: <https://github.com/Sino-Huang/NovPhy/issues/132#issuecomment-6382271495>
- P-engine (engine path, smoke design), 2026-10-24: <https://github.com/Sino-Huang/NovPhy/issues/132#issuecomment-6382714673>
- P-engine (smoke result, attempts, projection, stop rule), 2026-10-25: <https://github.com/Sino-Huang/NovPhy/issues/132#issuecomment-6388390880>
- A3 fidelity design (#128), 2026-10-25: <https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6388391126>; A3 funding after E1 (#128): <https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6464530188>

## Reproduction

```
source ~/miniconda3/etc/profile.d/conda.sh && conda activate novphy && source env.sh
export CUDA_VISIBLE_DEVICES=""
python -u scripts/run_issue_132_aibirds_agents.py --validate        # exit 0 (CPU, no engine)
python -m pytest -q tests/test_issue_132_aibirds_agents.py          # 23 passed
```

Re-running the engine stages (`--smoke`, `--run`) needs the stock build, the #104 player, the Mesa 26.1.2 llvmpipe
build, Java ≥ 13 and the conda env `b127-phyq` (package lists in `envs/`), inside the 07:05–17:30 window.
