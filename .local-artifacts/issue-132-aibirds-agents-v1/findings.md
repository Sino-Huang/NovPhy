# #132 E1: AIBIRDS agents acting natively on the #102 N1 levels (native-action table)

native-action, DESCRIPTIVE: the agents choose continuous shots on the #102 N1 levels; not the grid protocol; no disposition, no program claim.

- Levels: the 400 #102 N1 levels (#109 evaluation split, one bird each), reset to the start for every attempt.
- Attempts per level: Datalab 3, Eagle's Wing 3, PigShooter 3 (frozen P-engine decision: every agent's shot varied after a reset in the smoke).
- Pass: stage B: the agent's recorded wire shot, as a slingshot-relative drag, executed once on the frozen #104 player by the #102 executor (capture_pipeline_v2.run_campaign); pass = capture_pipeline_v2.branch_outcome pig_removed (#85 channel), as every #102 row.
- pass@k: #102 episode rule (run_closed_loop_evaluation.episodes / episode_view): attempt k is the agent's k-th reset to the level start; the episode ends at the first pass or the first typed failure (no transferable shot, or a stage-B execution that did not complete), so attempts after a typed failure are not counted; pass@k = 1 iff the episode passed within its first k attempts. Unlike #102, an attempt cannot exclude earlier failed launches (the agent chooses afresh after every reset). One cell per level (no model seeds). An agent the smoke found deterministic gets one attempt per level (a reset replays the same shot), so its pass@3 is not run (null) and pass@1 is its only row.
- Intervals: level-clustered percentile bootstrap, 10000 draws, PCG64 7201 (p94.bootstrap, as #102), DESCRIPTIVE. #102 cells are level x seed; E1 cells are levels.
- Engine: 153.7 of 190 CPU worker-hours (smoke and probes included).

The run completed every planned attempt; no allowance stop.

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

Per agent (DESCRIPTIVE):

- Datalab: outcomes {'failed': 272, 'passed': 123, 'typed_failure': 5}; typed failures {'no_shot': 5, 'stage_b_failed': 0}; mean attempts to pass among passed levels 1.154; agent-side level status vs stage-B pig removal agreement 1.0.
- Eagle's Wing: outcomes {'failed': 299, 'passed': 93, 'typed_failure': 8}; typed failures {'no_shot': 8, 'stage_b_failed': 0}; mean attempts to pass among passed levels 1.753; agent-side level status vs stage-B pig removal agreement 1.0.
- PigShooter: outcomes {'failed': 308, 'passed': 87, 'typed_failure': 5}; typed failures {'no_shot': 5, 'stage_b_failed': 0}; mean attempts to pass among passed levels 1.621; agent-side level status vs stage-B pig removal agreement 1.0.

Reading: descriptive only. The agents choose continuous shots in their native interface; the #102 rows rank a fixed 16-launch grid and exclude failed launches between shots. No delta-rule token is attached to any E1 row and no E1 number enters a disposition or the program claim.

Decisions: p_license https://github.com/Sino-Huang/NovPhy/issues/132#issuecomment-6382271495, p_engine_path https://github.com/Sino-Huang/NovPhy/issues/132#issuecomment-6382714673, p_engine_projection https://github.com/Sino-Huang/NovPhy/issues/132#issuecomment-6388390880, a3_p_engine https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6388391126.
