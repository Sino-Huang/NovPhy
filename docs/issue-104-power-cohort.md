# Issue #104 — joint #104 + #110 cohort plan (frozen) and rendered smoke

Parent: #108. Shared rules of #85 apply. Prior dispositions (#15–#113) are read-only inputs; nothing here
edits the ICLR 2026 submission. No cohort level has been captured, fitted or scored.

| Item | Location |
| --- | --- |
| #113 capture-window spec (frozen 2026-10-03T09:03:55Z, zero engine seconds) | `scripts/derive_capture_window_spec.py`, `.local-artifacts/issue-113-capture-window-v1/` |
| Capture player (parent 82db3f42 + window rule) | `scripts/issue_104_window_player.py`; assembly `ac418250…` |
| Renderer (Mesa 26.1.2 / LLVM 22.1.6 llvmpipe, no root) | `scripts/build_capture_mesa.sh` |
| Frozen joint plan (2026-10-03T10:06:16Z, before any capture) | `scripts/prepare_power_cohort.py`, `.local-artifacts/issue-104-power-cohort-v1/` |
| Rendered smoke | `scripts/run_power_cohort_smoke.py`, `.local-artifacts/issue-104-capture-smoke-v1/`, gallery `data/issue-104-capture-smoke/index.html` |
| Engineering runs (EXPLORATORY, exposed #77 N1 branches, never scored) | `.local-artifacts/issue-104-capture-dev/summary.json` |
| Tests | `tests/test_issue_104_power_cohort.py`, `tests/test_native_segment_trace.py` |

## 1. #113 capture-window spec (merged into the plan)

Evidence: every native step of 498 retained single shots (#109 smoke, #77 N2 appearance, #77 n2n).

- At the 12 s cap, the structure is at engine rest in 282 of 284 censored shots. The only moving body is the
  launched bird rolling on the ground (281 / 284). A longer cap would add frames of a lone rolling bird:
  +23 % frames for 16 s, +102 % for 30 s.
- Shots that reach engine rest (20–48 % per level type) stop on the first rest step, so they have 0 settled
  frames after it. That is why #100 saw 25 steady-state frames in 24 731.
- Structure changes end well before the cap. Collapse yield (22–36 % of shots per level type) depends on
  whether the shot reaches the structure, not on the window.

**Rule.** Keep the cap at 30 000 native steps for shots that are not at rest. After `stable_entered` at or
before the cap, record 50 more frames (2 500 steps) and finish with `rest_tail_complete`. `stable_exited`
cancels the tail. `level_clear` and `level_fail` still finish immediately. The rule is engine-side
(`NOVPHY_NATIVE_MAX_SHOT_STEPS`, `NOVPHY_NATIVE_REST_TAIL_STEPS`), and every manifest declares
`maximum_shot_steps` and `rest_tail_steps`. Python readers read these declarations; older traces default to
(30000, 0). The spec does not freeze #113's item-1 predicate definitions.

## 2. Power analysis (work item 1)

The analysis uses the #96 member-level records. It is DESCRIPTIVE and based on development variance. The
design count of required mixed members is the larger of two estimates: the bootstrap half-width scaled by
1/√m, and (1.96·sd/δ)².

| inventory | primary J − F*: members for δ = 0.02 / 0.05 | max over the #96 contrast set, δ = 0.02 / 0.05 | mixed share (development / held-out) |
| --- | --- | --- | --- |
| grid | 208 / 34 | 293 / 47 | 11/11, 3/4 |
| offset | 194 / 32 | 194 / 32 | 6/11, 3/4 |
| angle | 60 / 10 | 344 / 55 | 5/10, 2/4 |
| power | 224 / 36 | 328 / 53 | 5/11, 3/4 |

Depth rule, declared before any capture:

| target | δ | contrasts | candidates | depth (levels per family) |
| --- | --- | --- | --- | --- |
| grid (primary) | 0.02 | every #96 grid contrast | 16 | all 200 |
| offset, angle, power | 0.05 | every #96 contrast | union of 60 | first 60 |

- The share used is the conservative one, min(development, held-out). Depth is rounded up to 20.
- Expected mixed members: grid 300, offset 65, angle 60, power 55.

## 3. Frozen membership (49 560 branches, dispatch order)

| stream | role | levels | branches |
| --- | --- | --- | --- |
| #109 evaluation (grid; union for ordinal ≤ 60) | final_evaluation | 400 | 11 680 |
| #110 evaluation (40 pairs × 8 × 2 sides × 16) | final_evaluation | 640 | 10 240 |
| #110 fit | training (normal: zero-shot, novel: few-shot) | 640 | 10 240 |
| #109 fit (240 / family, 29 candidates) | training | 480 | 13 920 |
| #109 policy (60 / family) | training | 120 | 3 480 |

- Dispatch is ordinal-major within each stream.
- Limits: 16 workers, attempt cap 1 800 s, no retries.
- Stop rules: typed-failure share > 0.05 after 200 branches; < 500 GiB free; artifact cap 1.25 × projection;
  wall cap 302.8 h.
- Projection: 151 h, 4.5 TiB.
- The plan also binds the Gate-B unit (#112 v3: LC : F-15-macro, hybrid, E99) and its target, its
  pre-declared contrast against C*, and its two controls. It does not execute them.

## 4. Capture stack findings (engineering runs, before the freeze)

- **Renderer.** The server's Mesa 25.2.8 / LLVM 20.1.2 changes 228–872 edge pixels on every decision frame
  (0/20 equal to the retained frames). A user-local Mesa 26.1.2 / LLVM 22.1.6 reproduces the retained frames
  byte for byte. The plan pins this renderer, and any other renderer fails typed as `renderer_mismatch`.
- **Player.** On this server, the parent and window players produce identical full native prefixes on 19/20
  branches. The one remaining difference starts late in a level-clear shot.
- **Cross-machine physics.** Physics is not bit-identical across machines. From the bird's first collision
  the float digits differ from the retained traces, and event sequences agree on 16/20.
- **Window.** In single-bird levels the tail is usually cut by `level_fail` once the launched bird dies
  after rest.

## 5. Rendered smoke (168 branches: #110 smoke gate 160 + 8 exposed replays)

Captured once each in 35.5 min on 16 workers.

| gate | result |
| --- | --- |
| S1 capture reliability | 0 / 168 typed failures |
| S2 renderer | 168 / 168 on the frozen renderer |
| S3 pixel identity | 8 / 8 replay decision frames equal the retained #109 frames outside Bird/Pig boxes |
| S4 window rule | 0 violations: 89 censored at the cap, 22 `rest_tail_complete`, 35 clear/fail inside a tail, 22 clear/fail before rest |

**Campaign launch: `supported`.**

#110 per-template criteria: all 80 templates pass, so **45 / 45 cells are supported**. Every template had 0
typed failures, its authored novelty entity in the trace, the expected gravity ((0, 6) on level 6), and a
leftward launch on the right slingshot. The residual-acceleration check:

| templates | force sign | meaning |
| --- | --- | --- |
| magnet, turbulence (incl. inverse) | agree | labels match the observed acceleration |
| fan (level 2), 4 templates | disagree | median projection 0 or −100 on the labelled direction; the fan label's sign or zone needs repair |
| others | not exercised | no contact-free labelled step |

Under the #110 rule, the fan result is a labeler disagreement, repaired in the derivation before any
training; it does not make the cell unsupported. Storm onset fell inside the window on 7 of 10 storm branches.

Settle and collapse per novelty level, and the full template table, are in `findings.md`. The tail produced
917 rest frames over 80 level-0 shots (27 shots with rest), against at most 1 per shot under the old window.

Throughput: 12.7 s/branch amortized (single pass, including the cold-start ramp), branch wall 191.5 s,
88.6 MB/branch. Re-projected campaign: 174 h, 3.99 TiB, inside the frozen caps.

## Deviations and disclosures

- `SERVER_SETUP.md` byte-rewrote `/p/Project/NovPhy` in frozen records, so the #109 sealed-cohort
  `--validate` fails on its #96 input hash. Inputs here are bound to the current bytes, and every level file
  is checked against its frozen sha256 at dispatch.
- After the capture pass, `run_power_cohort_smoke.py` was changed only in its publish path: analysis and media
  now run in a process and thread pool. `run-manifest.json` keeps the runner hash from capture time. The
  capture code, membership and gates did not change.
- `findings.md` of the plan was regenerated once from the frozen `plan.json`, minutes after the freeze and
  before any capture. Only the rendering changed (key order); the plan bytes did not.
- The #85 single-RTX-3090 hardware gate is carried as a single-GPU gate on the 4 × RTX 5090 server. Capture
  uses no GPU.

## 6. Phase 2: campaign capture (complete)

| Item | Location |
| --- | --- |
| Runner (frozen membership, limits, stop rules; resumable, no retries) | `scripts/run_power_cohort_campaign.py` |
| Report (capture accounting, mixed-verdict yield per level type) | `scripts/publish_power_cohort_campaign.py` |
| Campaign directory | `.local-artifacts/issue-104-power-cohort-campaign-v1/` (one sub-directory per stream) |

- Ran 2026-10-03T12:29Z → 2026-10-10T09:04Z on 16 workers. The manifest binds the plan, pipeline, runner,
  player, renderer libraries, smoke summary and the frozen branch digest; a resumed `--run` refuses any
  difference.
- Streams run in the frozen order. The pipeline enforces S2 (free bytes) and S3 (the remaining artifact
  allowance is passed per stream). The runner enforces S1 (typed-failure share > 0.05 once ≥ 200 branches
  have finished, counted over every stream) and S4 (wall over closed runs plus the live run). A stop writes
  `stop.json` with every undispatched branch as `campaign_stopped:<rule>`; `--run` refuses to continue.
- **Placement (deviation from the smoke layout).** `/mnt/array` is mergerfs with an existing-path create
  policy, so a directory's files all land on the branch that holds it; the smoke's attempts sit on one
  branch. One branch (≤ 2.3 TiB free) cannot hold the ~4 TiB campaign, and an attempt's player hard links
  must stay on one branch. Each stream directory is therefore created on one branch before its first
  capture (largest projected stream first, onto the branch with the most unreserved bytes); the plan is
  recorded in `campaign-manifest.json` under `placement`. Capture code and membership are unchanged.
- **GPU.** Capture renders on the CPU (llvmpipe) and no campaign stage used a GPU. The runner sampled every
  GPU every 300 s into `gpu-accounting.jsonl` (1 971 samples). **Deviation: an external job used the GPUs.**
  From 2026-10-06T07:26Z to past the campaign end, a vLLM server of another project
  (`Atten_Residual_LLM_Planning_Analysis`, `gemma-4-12B-it`, tensor parallel over all 4 GPUs, ~30 GB each)
  ran; 1 169 samples show it. It was not started by this campaign and was not stopped. Exposure: #109
  evaluation none, #110 evaluation 1 368 / 10 240 branches, #110 fit, #109 fit and #109 policy all.
  Effect check (`load-replay-check.json`, EXPLORATORY): 160 #109 evaluation branches captured with idle
  GPUs were re-captured while the vLLM job ran; stop kind, pig verdict and the full native trace (every
  native step) are identical on 160 / 160. Branch wall rose 2.7 % (189.6 → 194.8 s, #110 evaluation
  before / after onset); shot capture time, renderer and window checks are unchanged.
- No outcome is read while the campaign runs: `--status` reports progress and typed failures only, and
  the report refuses a campaign that is not terminal (`complete.json` or a finalized `stop.json`).
- The report computes the engine verdict in one pass over the retained chunks with the frozen #85 channel
  functions (the ones `capture_pipeline_v2.branch_outcome` applies), without re-validating every sample;
  on the 160 smoke novelty branches it matches `branch_outcome` on 160 / 160 (verdict, consistency, stop
  kind).

### Results (`findings.md`, `summary.json`, `yield.csv`; `--validate` exit 0)

| stream | scheduled | complete | typed failures | wall h | TiB |
| --- | --- | --- | --- | --- | --- |
| #109 evaluation | 11 680 | 11 680 | 0 | 37.7 | 0.86 |
| #110 evaluation | 10 240 | 10 221 | 19 | 33.9 | 0.81 |
| #110 fit | 10 240 | 10 208 | 32 | 34.9 | 0.80 |
| #109 fit | 13 920 | 13 919 | 1 | 46.3 | 1.03 |
| #109 policy | 3 480 | 3 480 | 0 | 11.6 | 0.25 |
| total | 49 560 | 49 508 | 52 (0.105 %) | 164.5 | 3.75 |

- No stop rule fired (S1 peak share far below 0.05; S4 164.5 h of 302.8 h; S3 3.75 of 5.63 TiB). Renderer
  mismatches 0, #113 window-rule violations 0, channel consistency `agreement` on every complete branch.
- Typed failures (no retries):
  - 32 `initial shot omitted an authored object`: every candidate of `issue-110-evaluation-type010203-novel-005`
    and `issue-110-fit-type010203-novel-006` (fan, level 2). At the decision step `block:0003` and
    `novelty:0000` are missing from the scene, so the shared pre-decision state fails the slot check.
  - 16 `supervisor_interrupted` (#110 fit, `type010404-*-004`): the supervisor crashed at 2026-10-07T03:13Z
    with `ProcessLookupError` reading `/proc/<pid>/stat` of a worker that exited between open and read.
    `process_rss` now skips it (commit cb852fb4, regression test); the in-flight branches stay typed
    failures and `--run` resumed the stream.
  - 1 each: worker SIGSEGV at start, PNG read error, shot wall cap (900 s), shot progress stall (120 s).
- **Mixed-verdict yield (work item 2).** #109 evaluation mixed members vs the frozen design requirement:
  grid 311 (need 293 at δ 0.02), offset 76 (32 at δ 0.05), angle 78 (55), power 47 (53 for the max over
  the #96 contrast set; 36 for the primary). Power is the one inventory below its δ = 0.05 requirement.
  Mixed share per level type: rolling type010103 grid 0.72, angle 0.82, offset 0.73, power 0.42; sliding
  type010105 grid 0.84, angle 0.48, offset 0.53, power 0.37. #110 cells: 0 mixed levels in all 9
  multiple-forces cells and in every level-8 (storm) cell (flagged at design), 0.25–1.0 elsewhere; full
  table in `findings.md`.
- **Gate-B retrain (after the campaign).** The #112 v3 recipe was retrained on the `issue109-fit` stream only
  (`docs/issue-112-decision-relevant-dynamics.md` §7, `scripts/run_decision_dynamics_retrain.py`). Its handoff,
  `.local-artifacts/issue-112-v3-retrain-v1/gate_b_handoff.json`, carries the Gate-B unit and C* checkpoints, three
  usable seeds each, and supersedes the v3 handoff named in this plan's `gate_b` block. The `issue110-fit` normal
  sides named in `fit_merge` are not in it: the recipe fixes the 18-slot E99 encoder.
- **Gate B (executed):** pass, unit grid AUC 0.7444 [0.7269, 0.7620]; see `docs/issue-108-gate-b.md`.

## Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate novphy && source env.sh
python -u -m scripts.derive_capture_window_spec --validate
python -u -m scripts.prepare_power_cohort --validate
python -u -m scripts.run_power_cohort_smoke --validate
python -m pytest -q tests/test_issue_104_power_cohort.py tests/test_native_segment_trace.py

# phase 2
python -u -m scripts.run_power_cohort_campaign --dry-run
python -u -m scripts.run_power_cohort_campaign --run       # resumable; detached launch in run.log
python -u -m scripts.run_power_cohort_campaign --status
python -u -m scripts.publish_power_cohort_campaign --publish   # after complete.json / stop.json
python -u -m scripts.publish_power_cohort_campaign --validate
```

## 7. Work item 3: the #96 contrast set on the held-out split (scored once)

| Item | Location |
| --- | --- |
| Runner | `scripts/run_heldout_96_contrasts.py` (post-freeze amendment A1, registered in `scripts/post_freeze_amendments.py`) |
| Artifacts | `.local-artifacts/issue-104-heldout-contrasts-v1/` (`plan.json`, `smoke.json`, `findings.md`, `summary.json`, `compute.json`, `comparisons.csv`, `frequencies.json`, `records-manifest.json`, `ledger/`; the 54 raw records stay local in `records/`, 128 MB, bound by the manifest) |
| Freeze | Plan frozen 2026-10-18T07:28:19Z (commit 901de655), after a passed smoke and before any evaluation-split score |
| Validation | `python -u -m scripts.run_heldout_96_contrasts --validate` exits 0. It re-derives the cells, re-scores J seed 20260908 on all 512 cells bit for bit, and regenerates every published file and the record manifest byte for byte. `pytest tests/test_issue_104_heldout_contrasts.py` passes (7 tests), as does the amendment-registry test of `tests/test_issue_101_decision_aware_policy.py`. |

### Protocol (frozen; decisions posted on #104 before scoring)

- **Primary reading: the faithful #96 rerun.**
  - Stack: the ICLR #77 N1 checkpoints (hybrid and continuous; seeds 20260908, 20260909 and 20260910) with their #77 controllers.
  - Arms: all 18 #96 arms (F × 9, J, M, T, D, OL, C-F-1/5/15, C-J), with the #96 selector and arm specs.
  - F\* (cross-fit r†), F_hindsight and C-F\* follow the #96 rules.
  - Rollout to endpoint 225 through `request_policy.rollout_endpoints`, which runs t96's operations in the same order.
- **Cells.**
  - The Gate-B cells: the mixed-verdict #109 evaluation levels of this campaign (grid 311, offset 76, angle 78, power 47).
  - Each cell is scored for every seed (1 536 cells, 419 364 candidate rollouts). Inventories are never pooled.
- **Anchors.** The #105 F77 store, hash-pinned: the #71 repaired-adapter parse of each level's decision frame, using the Gate-B frame rule.
- **Outcome-free coordinates.** α_T, Δ_D and the OL weights come from this run's J traces, by the #96 rule. They are pooled per seed over the scheduled cells, which here are the mixed cells only. The values match #96's pattern:
  - seed 20260908: (continuous, 5);
  - the other two seeds: (macro, 1).
- **Cost.** The #111 tie-free cost is primary (pre-declared). The #96 count cost is secondary. Both come from the same endpoints.
- **Statistics.**
  - #96 cell AUC (ties count 0.5).
  - #96 paired contrasts, clustered by level (10 000 draws, PCG64 7201, DESCRIPTIVE).
  - #96 guards G1–G5 and caps, plus G7 (scoring policy).
- **G6, as #104 reads it.** The primary (grid J − F\*) half-width must be ≤ δ = 0.02. Also reported:
  - the #96 plan's literal G6 bound, 0.05;
  - the #104 design δ (grid 0.02; offset, angle and power 0.05);
  - for every contrast and inventory, its half-width and m·(half-width/0.02)², the number of levels that 1/√m scaling would need.
- **Tokens.** Tokens use the #96 rules, and a failed guard turns every token of that cost into `readiness_or_precision_insufficient`.

### Smoke (pre-freeze)

- **Replication on the original #96 cells.** All 3 186 #96 records were re-scored with the faithful path under `deterministic_scoring()`. #96 was scored on the RTX 3090 workstation, and this server is RTX 5090 (`SERVER_SETUP.md` step 4).

  | Agreement with the #96 record | Records (of 3 186) |
  | --- | --- |
  | Chosen ordinal | 3 168 |
  | Full ranking order | 3 066 |
  | Controller schedules | 3 038 |
  | Every count cost within 1e-3 | 2 272 |
  | Bit-exact in every field | 1 048 |

  - Bit-exact reproduction of the count cost happens mainly where the clamped cost saturates.
  - 16 records reproduce bit-exactly with at least two distinct costs. S1 names `decision--angle--issue-77-n1-001--seed20260910--D`, and the list includes J and M records.
  - The #96 G1 criterion (chosen ordinal equal, cost within 1e-3) holds on the #96 G1 J and F-15-continuous cells.
- **Other smoke checks.**
  - The Gate-B cell counts are reproduced.
  - The anchor store's scoring policy and frame rule are confirmed, and 8 levels re-parse bit-identically.
  - In-process re-scoring is bit-identical.
  - Checkpoint identities match.
- **Disclosure: the smoke ran twice.** The first run named an all-tied record for S1, so S1 was tightened to require ≥ 2 distinct costs. Both runs happened before the freeze, and both are kept (`smoke-run1-superseded.json`).

### Result (primary, tie-free cost): `readiness_or_precision_insufficient`; G6 fails

**θ_grid = AUC_J − AUC_F\* (F\* = F-5-macro): +0.0134 [−0.0163, +0.0431], half-width 0.0297** over 933 cells and 311 levels. G1–G5, G7 and caps pass, and the tie share is 0. G6 fails (0.0297 > 0.02), so every token of this cost is `readiness_or_precision_insufficient`. With the #96 plan's literal bound (0.05), G6 would pass, but the token would be the same: the interval contains both 0 and δ.

Achieved half-width of J − F\* against the G6 bound:

| inventory | mixed levels (cells) | J − F\* [F\*] | half-width | ≤ 0.02 (G6) | ≤ design δ | levels for 0.02 by 1/√m |
| --- | --- | --- | --- | --- | --- | --- |
| grid | 311 (933) | +0.0134 [−0.0163, +0.0431] [F-5-macro] | **0.0297** | no | no (0.02) | ≈ 687 |
| offset | 76 (228) | +0.0111 [−0.0520, +0.0743] [F-5-continuous] | **0.0632** | no | no (0.05) | ≈ 758 |
| angle | 78 (234) | −0.1596 [−0.2061, −0.1130] [F-5-macro] | **0.0466** | no | yes (0.05) | ≈ 424 |
| power | 47 (141) | −0.2617 [−0.3360, −0.1921] [F-5-macro] | **0.0720** | no | no (0.05) | ≈ 609 |

**Achieved-half-width statement.** G6 fails on every inventory. On grid, 311 mixed levels give a half-width of 0.030. That is 2.5 times narrower than #96 (0.075 over 14 members), but 1.5 times wider than δ. The #104 design (293 levels for δ 0.02) assumed the development variance; the held-out between-level variance is larger.

Power (47 < 53 levels) misses even its design δ of 0.05, as the yield table predicted. Offset also misses it (0.063). Angle meets it (0.047).

Every contrast (tie-free; estimate, half-width; the full table with the #104 design counts is in `findings.md`):

| contrast | grid | offset | angle | power |
| --- | --- | --- | --- | --- |
| J − F\* | +0.013 [−0.016, +0.043], 0.030 | +0.011 [−0.052, +0.074], 0.063 | −0.160 [−0.206, −0.113], 0.047 | −0.262 [−0.336, −0.192], 0.072 |
| J − F_hindsight | −0.084 [−0.115, −0.054], 0.031 (F-15-cont) | −0.161 [−0.207, −0.113], 0.047 | −0.160 [−0.206, −0.113], 0.047 | −0.310 [−0.368, −0.255], 0.057 |
| J − F(1, macro) | +0.034 [+0.012, +0.056], 0.022 | +0.106 [+0.064, +0.150], 0.043 | +0.105 [+0.071, +0.142], 0.036 | +0.130 [+0.067, +0.194], 0.064 |
| J − F(1, continuous) | +0.027 [+0.008, +0.046], 0.019 | −0.040 [−0.078, −0.001], 0.038 | −0.039 [−0.073, −0.003], 0.035 | −0.103 [−0.174, −0.033], 0.070 |
| J − M | +0.008 [−0.004, +0.020], 0.012 | −0.061 [−0.093, −0.030], 0.031 | −0.049 [−0.079, −0.019], 0.030 | −0.025 [−0.070, +0.019], 0.044 |
| J − OL | +0.040 [+0.025, +0.056], 0.016 | +0.004 [−0.033, +0.044], 0.039 | −0.009 [−0.046, +0.031], 0.039 | +0.010 [−0.040, +0.059], 0.049 |
| J − T | −0.001 [−0.015, +0.013], 0.014 | −0.061 [−0.102, −0.018], 0.042 | −0.060 [−0.094, −0.024], 0.035 | −0.070 [−0.122, −0.023], 0.050 |
| J − D | +0.058 [+0.037, +0.079], 0.021 | +0.010 [−0.035, +0.056], 0.045 | +0.013 [−0.028, +0.056], 0.042 | +0.063 [+0.012, +0.115], 0.051 |
| C-J − C-F\* (C-F-5) | −0.049 [−0.060, −0.039], 0.011 | −0.065 [−0.080, −0.048], 0.016 | −0.060 [−0.075, −0.044], 0.016 | −0.172 [−0.195, −0.150], 0.023 |

- **Grid secondaries.**
  - J − M: +0.008 [−0.004, +0.020].
  - J − OL: +0.040 [+0.025, +0.056], own half-width 0.016.
  - Both tokens are `readiness_or_precision_insufficient` because G6 fails.
  - Under the #96 secondary rule, which applies only the contrast's own half-width, J − OL would read `supported`. This is a DESCRIPTIVE reading, not a token (amendment A1 below).
- **Replication** of J − F\* on offset, angle and power (mean ≥ δ and lower bound > 0): 0 / 3 (DESCRIPTIVE).
- **Reading.**
  - On held-out grid levels the ICLR J ranks near chance: AUC 0.555 [0.539, 0.571]. F\* scores 0.542 and the best fixed request in hindsight (F-15-continuous) 0.639.
  - J − F\* does not establish a ≥ δ advantage, and J trails F_hindsight by 0.084.
  - On power, every fixed request at Δ ≥ 5 ranks well (0.72–0.77) and J does not (0.459).
  - Within-checkpoint adaptivity therefore stays unresolved on this checkpoint. The interval excludes neither 0 nor δ.

### Secondary cost (count, the #96 cost)

θ_grid = +0.0893 [+0.0685, +0.1107], half-width 0.0211. G4 fails, with 395 / 933 grid cells fully tied (share 0.42; #96 had 0.405). G6 fails narrowly (0.0211 > 0.02). Every count-cost token is therefore `readiness_or_precision_insufficient`.

J − F\* half-widths by inventory: grid 0.021, offset 0.040, angle 0.036, power 0.050.

### Secondary reading: the #112 Gate-B predictor's #96-style contrasts (#101, re-tabulated, not re-scored)

Source: #101 `compute.json`. Unit: LC hybrid, E99 Gate-B anchors, tie-free cost. J = `J-pe-1e-04`; J_dec = `J-dec-1e-01`. All #101 guards pass. #101 has no J − F(1, ·), J − T, J − D or C-J − C-F\*.

| contrast | grid | offset | angle | power |
| --- | --- | --- | --- | --- |
| J − F\* (F-5-micro) | −0.039 [−0.048, −0.031], 0.0085 | −0.090 [−0.126, −0.055], 0.0353 | −0.079 [−0.106, −0.054], 0.0258 | −0.006 [−0.014, +0.001], 0.0071 |
| J − F_hindsight | −0.057 [−0.068, −0.046], 0.0110 | as J − F\* | as J − F\* | as J − F\* |
| J − M | −0.001 [−0.003, +0.002], 0.0025 | +0.039 [+0.015, +0.063], 0.0240 | −0.002 [−0.018, +0.015], 0.0164 | +0.032 [+0.015, +0.050], 0.0178 |
| J − OL | +0.013 [+0.010, +0.016], 0.0029 | +0.080 [+0.061, +0.098], 0.0186 | +0.050 [+0.034, +0.065], 0.0158 | +0.011 [+0.003, +0.019], 0.0082 |
| J_dec − F\* | +0.001 [−0.005, +0.008], 0.0065 | −0.152 [−0.176, −0.128], 0.0243 | −0.157 [−0.180, −0.135], 0.0227 | −0.019 [−0.035, −0.006], 0.0145 |
| J_dec − F_hindsight | −0.016 [−0.020, −0.012], 0.0041 | as J_dec − F\* | as J_dec − F\* | as J_dec − F\* |
| J_dec − M_dec | +0.001 [−0.001, +0.002], 0.0012 | +0.001 [−0.004, +0.006], 0.0047 | +0.003 [−0.002, +0.009], 0.0055 | +0.000 [−0.001, +0.002], 0.0011 |
| J_dec − OL_dec | −0.003 [−0.006, +0.000], 0.0033 | −0.023 [−0.032, −0.014], 0.0090 | −0.023 [−0.032, −0.013], 0.0091 | −0.012 [−0.025, −0.003], 0.0110 |

**G6 reading (J − F\*, ≤ 0.02):** grid passes (0.0085), power passes (0.0071), offset fails (0.0353) and angle fails (0.0258). Offset and angle meet their design δ of 0.05. The tokens are #101's own. On this predictor, J − F\* on grid is resolved: J trails F\* (upper bound −0.031 < δ, `not_supported_by_this_experiment`).

### Compute

| Phase | Time |
| --- | --- |
| Scoring (phase 1: 45 records on 4 GPUs; phase 2: 9 records) | 3 802 GPU s, 3 970 s process wall, 1 344 s elapsed |
| Validation | 92 s |
| Smoke runs | 2 × about 310 s |

- Caps: 6 GPU-h and 6 h.
- Engine time: 0 s. No training.
- Hardware: RTX 5090 instead of the #85 RTX 3090.
- Worker PIDs are in `logs/score-pids.txt` (the first four lines there are launcher PIDs; the correction follows) and in `ledger/`.
- **Deviation: a co-scheduled job.** The H-JEPA job of #123 (`eval.py` on GPUs 0–2 from 07:37:36Z; `hjepa_train_ckpt.py` on GPU 3 from 07:40:02Z) shared the GPUs with the last 3 minutes of phase-1 worker 0 and with all of phase 2. The owner agent asked first, and the request was granted. This inflates the GPU-second accounting of T, D and OL. Numerics are unaffected: scoring runs under `deterministic_scoring()`, and the `--validate` re-score (run while the job was active) is bit-identical.

### Post-freeze runner amendment (disclosure)

**A1 (reporting and guard propagation, after scoring).**
- **What the frozen code did.** It gated the grid secondaries J − M and J − OL by the #96 secondary rule: guards other than G6, plus the contrast's own half-width. That published J − OL (tie-free) as `supported` even though G6 had failed. The frozen plan's own guard_failure rule ("any failed guard turns every token of that cost into the stop token") and the round-2 convention say otherwise.
- **What A1 changes.** A1 applies the guard_failure rule to the secondaries too. The #96 secondary rule and the #96 literal-G6 reading stay as DESCRIPTIVE readings.
- **What did not change.** No record, estimate or interval changed. The amendment is registered in `scripts/post_freeze_amendments.py` (sha256 7c9a8d28… → 8cf796f0…), and `load_plan` accepts only the frozen or the registered bytes.

### Reproduction (work item 3)

```bash
python -u -m scripts.run_heldout_96_contrasts --dry-run
python -u -m scripts.run_heldout_96_contrasts --smoke          # pre-freeze; refuses after plan.json
python -u -m scripts.run_heldout_96_contrasts --prepare
CUDA_VISIBLE_DEVICES=k python -u -m scripts.run_heldout_96_contrasts --run --phase 1 --workers 4 --worker k   # k = 0..3, once
python -u -m scripts.run_heldout_96_contrasts --frequencies
CUDA_VISIBLE_DEVICES=k python -u -m scripts.run_heldout_96_contrasts --run --phase 2 --workers 4 --worker k   # once
python -u -m scripts.run_heldout_96_contrasts --publish
python -u -m scripts.run_heldout_96_contrasts --validate
python -m pytest -q tests/test_issue_104_heldout_contrasts.py
```
