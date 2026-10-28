## Result: E1 native-action table on the 400 #102 N1 levels (DESCRIPTIVE)

Datalab, Eagle's Wing and PigShooter (phy-q/benchmark @ `514491c`, run unmodified) each chose their own continuous shot on all 400 #102 N1 levels, with 3 resets per level. All 3,600 planned attempts were recorded, with no allowance stop.

- **Verdict:** pig removal (#85 channel). Each agent's recorded shot was executed once on the frozen #104 player by the #102 executor, as for every #102 row.
- **pass@k:** #102's episode rule. Intervals use #102's own level-clustered percentile bootstrap (10 000 draws, PCG64 7201).
- **Status of these numbers:** they carry no δ-rule token and enter no disposition or program claim.
- **Validation:** `--validate` exits 0. Publication commit `b18c954f`; doc `docs/issue-132-aibirds-agents.md`.

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

**Reading (descriptive only):**
- **Datalab:** its pass@1 point estimate (0.273) is the highest row apart from the oracle. Its later attempts add little: pass@3 is 0.307, and mean attempts to pass among passed levels is 1.15.
- **Eagle's Wing and PigShooter:** their point estimates lie between #102's random and prior rows at both pass@1 and pass@3.
- **The retry mechanisms differ:** the #102 rows exclude failed grid launches between shots, while an agent chooses afresh after each reset. The pass@3 columns therefore measure different things, and the rows are not a paired contrast.

**Per agent (400 episodes each):**

| Agent | Passed | Failed after 3 | Typed failures (no transferable shot) | Agent-side status vs stage-B pig removal |
| --- | --- | --- | --- | --- |
| Datalab | 123 | 272 | 5 | agrees on every attempt |
| Eagle's Wing | 93 | 299 | 8 | agrees on every attempt |
| PigShooter | 87 | 308 | 5 | agrees on every attempt |

No stage-B execution failed.

**Protocol, as frozen before the N1 run:**
- **Determinism:** in the 20-fit-level smoke, an agent repeated the same shot after a reset on 10/20 levels (Datalab), 0/20 (Eagle's Wing) and 11/20 (PigShooter). All three therefore count as varying and got 3 attempts per level.
- **Run order:** attempt 1 for every level first, then attempt 2, then attempt 3. Levels were in a fixed pseudo-random order, and a chunk started only while the engine total stayed within the 190 worker-hour allowance.
- **Engine window:** engine processes ran only in daytime windows (07:05–17:30 Melbourne), 25–28 Oct.

**Compute:**
- **Total:** 153.657 of 190 CPU worker-hours, smoke and probes included.
  - N1 shot choice: 33.6 h.
  - N1 verdict executions: 113.8 h, over 2,673 distinct executions (0.74 per attempt).
  - Smoke: 6.1 h.
- **Versus projection:** the projection was 188.6 h. The difference comes from more shared shots and faster verdict executions.
- **GPU:** 0 s.
- **A3:** the remainder R = 36.3 worker-h funds A3's native-task fidelity run (#128 issuecomment-6464530188).

**Disclosures:**
- **Transport adaptations, none in an agent's policy:**
  - the stock player is launched through the #104 wrapper, with audio disabled;
  - each Java agent's hard-coded `localhost` is mapped to a private loopback address through a JVM hosts file;
  - PigShooter's level load is sent as request 53 instead of 51, which hangs on this build.
- **Launcher bug:** the launcher's time comparison was parsed as octal, so it started 9 times outside the window. Each time the runner's own window check exited 75 within 70 s, before any engine process started.
- **Owner attention:** `data_lab_agent_v5.jar` and `eagle_wings_agent_v6.jar` (CC BY-NC-SA 3.0) have been tracked in this public repo since `68480cd0`, before #127. That conflicts with the program's "never redistribute" policy, and keeping them tracked is the owner's decision. #132 added no copy.

**Decisions:**
- P-license: issuecomment-6382271495
- P-engine, engine path: issuecomment-6382714673
- P-engine, projection and attempts: issuecomment-6388390880
