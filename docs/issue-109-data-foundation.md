# Issue #109 — data foundation: held-out evaluation, capture reliability, training coverage

Parent: #108. Shared rules of #85 apply. Prior dispositions (#15–#98) are read-only
inputs; nothing here edits the ICLR 2026 submission.

| Work item | Runner | Artifacts |
| --- | --- | --- |
| 1 held-out re-score of #96 | `scripts/run_heldout_rescore.py` | `.local-artifacts/issue-109-heldout-rescore-v1/` |
| 2 capture reliability (fixed pipeline) | `scripts/capture_pipeline_v2.py` | — |
| 2–3 rendered smoke, timeout–outcome audit, throughput | `scripts/run_capture_reliability_smoke.py` (+ `scripts/capture_smoke_report.py`) | `.local-artifacts/issue-109-capture-smoke-v1/`, `data/issue-109-capture-smoke/` |
| 4 sealed cohort plan | `scripts/prepare_sealed_cohort.py` | `.local-artifacts/issue-109-sealed-cohort-v1/` |
| 5 n2n-v2 retry campaign | `scripts/run_n2n_retry_campaign.py` | `.local-artifacts/issue-77-n2n-v2/` |

## Root causes of the typed capture failures (evidence from retained artifacts)

Old-pipeline failure inventory: #77 N1 30/208, N2-appearance 22/208, n2n 70/104
(together 82 paired-view mismatches, 35 shot-manifest deadlines, 5 PLAYING
timeouts); #87 oracle 55/289 (51 `TimeoutError: timed out`, 1 PLAYING timeout,
3 sealed-frame stability violations).

1. **#87 `TimeoutError: timed out` is a port-reservation race, not contention.**
   All 51 timed-out attempts have `SocketException: Address already in use` at
   `PhysicsCaptureDirectSocket.StartListening` in their engine log; none of the
   234 executed attempts do. `reserve_ports` probed the lowest free ports in
   28700–28999 and released them; Unity binds its physics port 10–20 s later, so
   every cell dispatched inside that window received the same ports. Every one of
   the 51 has another cell with an overlapping port dispatched 1–2 s apart. The
   loser's bridge then blocks for the 180 s socket timeout (median failed wall
   191 s). This also explains #90's "fewer workers live at dispatch" signature:
   collisions happen when a finishing worker frees the lowest block and two
   launches refill it. Which cell loses is set by dispatch timing, not by the
   slot's outcome.
2. **Paired-view mismatches and #87 stability violations are a render-timing
   artifact of character animation.** The physics state is identical (same fixed
   step and time); the RGB differs only inside the bird/pig sprite: the barrier
   renders at the fixed step while the endpoint renders at frame time, and
   `ABCharacter.Blink` (`Invoke` + `Animator`) can change phase in between. Example:
   #87 `oracle--issue-77-n1-005-a07--a02` differs from its sealed history frame in 26
   pixels, all inside the bird box (eye open vs closed). It is load-dependent:
   `issue-77-n1-002-a00` failed in v1 but passed when re-captured alone. It clusters
   by lineage (n2n 003/004/006 13/13; N1 002 11/13, 005 9/13) because the blink
   phase at the decision step is a level property.
3. **Shot-manifest deadlines cut long branches.** The 180 s wall deadline fired
   while the engine was still writing frames (446–601 of 601 RGB frames present),
   so full-window right-censored branches were cut more often than branches that
   end early. This is an outcome-linked failure mechanism.
4. **PLAYING timeouts** are cold starts under contention (60 s bound, up to 8
   concurrent engine boots).

## Fixed pipeline (`scripts/capture_pipeline_v2.py`)

- Each worker slot owns a private port block (base 29100, 30 ports per slot,
  outside the ephemeral range); a sub-block that is still busy is skipped
  deterministically; a bind conflict in the engine log fails the attempt at once
  as `port_bind_conflict`.
- Engine cold starts are serialized by a file-lock gate (JVM + Unity boot +
  menus); connect 120 s, menus 180 s, barrier readiness 300 s.
- RGB invariance at the decision state is pixel-exact outside the projected
  screen boxes of Bird/Pig nodes (3 px pad) from the paused physics capture;
  fixed step and time must be identical. The corrected-single view is the
  barrier-sealed decision frame, so all views share one frame by construction.
- The shot wait fails only when no chunk or frame appears for 120 s, under a
  900 s wall cap.
- Per-stage wall seconds are recorded for every attempt; the engine-truth outcome
  scan (#85 channel) runs at publish time, off the capture critical path.
- Typed failure taxonomy: `capture_pipeline_v2.failure_class`.

## Results

### 1. Held-out re-score of #96 (zero engine seconds)

Plan frozen 2026-09-26T13:05:06Z before any held-out statistic. Held-out
members 007/008/014/016 (mixed cells: grid 9, offset 9, angle 6, power 9; 1–3
member clusters). F*, F_hindsight, C-F* are the published #96 choices. The
all-member cohort reproduces 215 published #96 values exactly.

| grid contrast | published (15 members) | held-out (3 clusters) | exposed (11) |
| --- | --- | --- | --- |
| J − F* [F-5-macro] | 0.0700 [−0.0057, 0.1437] | 0.0704 [0.0333, 0.1222] | 0.0699 [−0.0234, 0.1589] |
| J − M | −0.0215 [−0.0706, 0.0241] | −0.0519 [−0.1556, 0.0222] | −0.0132 [−0.0698, 0.0366] |
| J − OL | −0.0520 [−0.1192, 0.0173] | −0.1556 [−0.3111, −0.0111] | −0.0237 [−0.0887, 0.0466] |

Held-out tokens are `readiness_or_precision_insufficient` for all three (G4 tie
share 0.333 fails), the same as published: the #96 reading does not change. The
held-out J − F* point estimate matches the exposed one on grid (difference
0.0005), so the grid primary shows no exposure optimism. The other inventories
move in both directions (offset −0.072, angle −0.113, power +0.055). Intervals from
3 clusters have 10 distinct resamples and understate uncertainty. Full tables:
`.local-artifacts/issue-109-heldout-rescore-v1/findings.md`.

### 2–3. Rendered smoke on the fixed pipeline (208 branches, 6 workers)

Plan frozen 2026-09-26T13:28:28Z before any smoke outcome. Membership: all 208
#77 N1 branches (exposed development lineages), each attempted once.

- **Typed failures 0/208 (rate 0.000 < 0.05 gate) → `supported`.** No port
  conflicts, no PLAYING/barrier timeouts, no shot stalls.
- **Timeout–outcome audit on the new pipeline: `bias_bounded_below_margin` →
  `supported`** (bound B = 0/208 ≤ 0.05 margin). This replaces #90's
  `indeterminate` for the fixed pipeline.
- Retrospective audit of the old failures, with outcomes measured for every
  branch on the uniform pipeline (DESCRIPTIVE, member-clustered):
  - #87 timeouts vs executed: pig removal 0.096 vs 0.030 (+0.066 [−0.010, 0.129]),
    clear +0.017, censoring −0.006 → `indeterminate`: below the 0.10 margin, but
    the interval is too wide to call the timeouts outcome-blind.
  - #77 N1 v1 failures vs complete: right-censored 0.833 vs 0.556 (+0.277
    [0.160, 0.469]), which fits the shot-deadline mechanism. The precision guard
    trips (5 target members < 8) → `indeterminate`.
- Determinism: stop kind 178/178 against #77 N1 v1; engine-truth pig removal
  234/234 against #87 and 54/54 against #89.
- Blink evidence: 10/208 decision states differed only inside Bird/Pig boxes
  (26 px). The old pipeline would have failed all 10 as paired-view or stability
  violations.

Throughput (per branch, complete branches):

| stage | mean s | p90 s |
| --- | --- | --- |
| shot capture (30 000 native steps rendered, 601 RGB frames) | 118.2 | 158.1 |
| shot validation (trace parse) | 14.4 | 21.1 |
| menu → PLAYING | 12.4 | 12.0 |
| cleanup (engine/display teardown) | 10.4 | 10.4 |
| replay to decision step 30 000 | 6.5 | 6.7 |
| cold-start gate wait | 4.5 | 12.1 |
| decision seal + 2 s hold | 3.9 | 3.5 |
| other | 1.4 | — |

Per-branch wall mean 174.9 s (p90 221.6) against R3's 563.5 s. Amortized campaign
wall is 29.5 s/branch at 6 workers (R3 117.6 at 8, #77 N1 v1 57.7 at 8), so
122 branches/hour. Disk is 84 MB/branch. Real-time shot rendering dominates
(≈68 % of branch wall). The R3 diagnostics
(`docs/issue-76-b-plus-v2-campaign-proposal.md`) attribute R3's late-ordinal
growth to per-drain full-tree artifact scans. v2 does not do those scans; it
accounts bytes incrementally per attempt. The engine-outcome scan runs at publish time: 35 CPU
s/shot, parallel. Artifacts: `.local-artifacts/issue-109-capture-smoke-v1/`,
gallery `data/issue-109-capture-smoke/index.html`.

**Budget for item 4 / #104:** 29.5 s amortized wall and 84 MB per branch at 6
workers. Storage, not wall time, is the binding constraint: the full union
evaluation roster (24 000 branches) is 197 h and 1.87 TiB.

### 4. Sealed cohort plan (`.local-artifacts/issue-109-sealed-cohort-v1/plan.json`)

Frozen 2026-09-26T15:33:40Z before any capture, with the measured smoke cost
(hash-bound). `python -u -m scripts.prepare_sealed_cohort --validate` re-derives
every level and re-checks disjointness.

- Families: normal rolling type010103 and normal sliding type010105, the families
  the #96 contrasts evaluate.
- Level-disjoint splits per family, in fixed roster order: fit 240, policy 60,
  evaluation 200 (1 000 levels). Fresh seeds from 765 000 000 / 765 500 000, checked
  against 27 reserved ranges. No collision in any of the 5 exposure classes over
  83 prior sources. 160 seeds were rejected before any verdict (slot outside the
  frozen 18-slot vocabulary).
- Candidate identities are deduplicated: one identity per (level, engine seed,
  action), `<level>-cNN`, with an inventory → candidate map. Evaluation levels
  carry angle 13 ∪ grid 16 ∪ offset 11 ∪ power 20 = 60 unique candidates
  (outcome-free densification from 13). Fit/policy levels carry angle ∪ grid = 29,
  so the primary grid inventory lies inside training support. This removes the
  #92 duplicate-identity inflation.
- Training coverage: rolling/sliding training data was 12 N1 lineages (504 windows);
  the issue-71 corpus holds only type010101/02. Both evaluation families now have
  240 fit + 60 policy levels.
- Expected yield (development shares from #96, caveated as optimistic): mixed members
  grid 14/15, offset 9/15, angle 7/14, power 8/15.
- Budget (6 workers, measured): grid-only evaluation prefixes of 20/40/80/120/200
  levels per family = 640/1 280/2 560/3 840/6 400 branches, 5.3/10.5/21.0/31.5/52.5 h,
  50/100/200/300/499 GiB. Full rosters (all 1 000 levels) = 41 400 branches, 340 h,
  3.2 TiB.
- Handoff to #104: before any verdict, #104 fixes the prefix depth per split from
  its power analysis, the evaluation candidate subset, and the campaign limits.

### 5. n2n-v2 retry campaign (`.local-artifacts/issue-77-n2n-v2/`)

Frozen 2026-09-26T15:33:56Z, after the smoke gate cleared and before any capture.
Membership: exactly the 70 n2n-v1 typed failures (48 paired-view, 20 shot-deadline,
2 PLAYING). Result: **70/70 recovered, 0 typed failures → `supported`**; the
combined type010102 normal side is 104/104 branches and 8/8 complete lineages
(v1: 34/104, 0/8). Every recovered branch is a long window (46 right-censored,
24 stable-without-clear, 0 early clear/fail). The v1 failures therefore fell
on long branches, which fits the deadline mechanism. The #77 N2 evaluation is
not reopened; bringing these branches into an evaluation is a later ticket's
frozen decision.

## Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate novphy && source env.sh
python -u -m scripts.run_heldout_rescore --validate
python -u -m scripts.run_capture_reliability_smoke --validate
python -u -m scripts.prepare_sealed_cohort --validate
python -u -m scripts.run_n2n_retry_campaign --validate
python -m pytest -q tests/test_issue_109_*.py
```

Pre-freeze engineering runs of the pipeline (EXPLORATORY, never scored) are kept
under `.local-artifacts/issue-109-capture-dev/`.
