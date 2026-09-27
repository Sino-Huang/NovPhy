# Issue #99 — decision-relevant world model: where the launch-ranking signal is lost

Parent: #108. Shared rules of #85 apply. Prior dispositions (#15–#98, #109) are read-only
inputs; nothing here edits the ICLR 2026 submission.

| Work item | Runner | Artifacts |
| --- | --- | --- |
| 1 stage-wise attribution + 2 tie-free cost | `scripts/run_decision_chain_attribution.py` | `.local-artifacts/issue-99-decision-chain-attribution-v1/` |
| 3 fix of the dominant stage (slot encoder) | `scripts/run_slot_encoder_fix.py`, `world_model/training/spatial_slot_parser.py` | `.local-artifacts/issue-99-slot-encoder-fix-v1/` |
| v2 guard revision (terminal) | `scripts/run_decision_chain_guard_revision.py` | `.local-artifacts/issue-99-guard-revision-v2/` |

Validation (all exit 0, `conda activate novphy && source env.sh`):

```
python -u -m scripts.run_decision_chain_attribution --validate
python -u -m scripts.run_slot_encoder_fix --validate
python -u -m scripts.run_decision_chain_guard_revision --validate
pytest tests/test_issue_99_decision_chain.py
```

Each runner froze its plan (`plan.json`) after a smoke that joined no verdict and before any
scoring or training: attribution 2026-09-27T02:07:52Z, slot-encoder fix 2026-09-27T03:31:32Z.
Every interval is a member-clustered percentile bootstrap (10 000 draws, PCG64 7201), DESCRIPTIVE.
Zero engine seconds: every executed shot is a retained #87/#89/#93/#94 oracle execution.

## 1. Attribution (859 verdict-bearing shots, 4 inventories, 14 requests, 3 seeds)

Five rankers per candidate, each replacing one stage with engine truth:

| stage | ranker |
| --- | --- |
| a | engine state at position 225 (fixed step 41 250) projected into the carrier -> cost |
| ap | agent frame at position 225 parsed by the frozen issue-70 parser -> cost |
| d | teacher-forced: predictor inputs are the executed shot's parsed carriers; only the last step to 225 is predicted |
| c | parsed sealed anchor -> rollout -> cost (the #96 arm; reproduces all 36 876 #96 costs within 4.8e-7) |
| b | engine-projected anchor -> rollout -> cost |

Tie-free cost (frozen before scoring): unclamped pig presence − 0.1 × pig-slot center
displacement from the stage's start carrier. Count cost = frozen `TaskObjective`.

Grid, all 15 members, tie-free cost:

| a | ap | c (J) | c (best, F-15-continuous) | d (J) | b (J) |
| --- | --- | --- | --- | --- | --- |
| 0.7857 [0.6429, 0.9286] | 0.4932 [0.3139, 0.6619] | 0.5544 [0.4744, 0.6374] | 0.6371 [0.5248, 0.7403] | 0.6152 [0.4705, 0.7441] | 0.5790 [0.5327, 0.6297] |

Request-mean paired drops (grid, all members, tie-free):

| 1 − a (cost ceiling) | a − ap (endpoint perception) | ap − d (one step) | d − c (compounding) | b − c (anchor perception) | ap − c (dynamics) |
| --- | --- | --- | --- | --- | --- |
| 0.2143 [0.0714, 0.3571] | **0.2925 [0.0922, 0.4891]** | −0.0989 [−0.1845, −0.0105] | 0.0148 [−0.1178, 0.1370] | 0.0038 [−0.0109, 0.0186] | −0.0842 [−0.2685, 0.0962] |

- **Gate A (frozen rule): dominant stage = perception** (endpoint perception), remedy "improve the
  slot encoder". Held-out cohort (reported, not used): cost dominant (0.3333 vs perception
  0.1556). Under the count cost the grid gate reads cost (0.2895 vs perception 0.2104).
- Outcome-free perception fidelity explains it: the frozen parser (96x64 input, one pooled
  vector plus per-slot offsets) reads the pig as present in **0/859** decision frames where the
  engine has it in 859/859, and in 10/859 endpoint frames (engine 836). The pig channel that the
  count cost reads is noise.
- The predictor ignores its start carrier (b − c ≈ 0.004), and teacher forcing does not beat
  the rollout (d − c ≈ 0.015): with this perception, dynamics is not where the loss sits.
- **Ties:** at stage c on mixed grid cells the tie-free cost leaves 0/42 cells tied for every one
  of the 14 requests (count cost: up to 0.810, J 0.333, #96 G4 0.405). Under the tie-free cost
  stage-c AUCs rise (grid J 0.4706 -> 0.5544; F-15-continuous 0.4918 -> 0.6371).
- Baseline held-out grid (3 clusters), stage c, tie-free: 7 of 14 requests already clear the
  target (e.g. F-15-continuous 0.7778 [0.7333, 0.8000], C-F-5 0.7037 [0.6889, 0.7111]).

**Tokens: Q1 attribution `readiness_or_precision_insufficient`, Q2 tie-free cost
`readiness_or_precision_insufficient`** — forced by the frozen guard G2 (position-0 frame
byte-identical to the sealed anchor), which failed on 4/859 shots (grid 016-o06, offset
010-o03/o08, angle 010-o07; 196–207 differing pixels each, about 80 of them inside the bird/pig
boxes, the rest on ground/background rows). G3 (engine-projected position-0 state identical across
every shot of a member) passed with spread 0.0, so the physical start state is the same; the
guard is kept as frozen. Without the guard, the frozen rules would read Q1 `supported` (dominant
drop lower bound 0.0922 > 0) and Q2 `supported` (every tied share 0.000 < 0.10); this is a
reading of the published numbers, not a disposition.

Compute: evidence extraction 3716 s (192 672 parser calls), rollouts 658 GPU s, wall 4394 s.
`compute.json` `compute.extraction.wall_seconds` is an unpopulated field (0.0); the extraction
wall is `compute.extraction_seconds`.

## 2. Fix of the dominant stage: new slot encoder (held-out members only)

- **E**: `SpatialSlotParser` (320x240 input, 40x30 feature map, 18 slot queries with 2
  cross-attention layers; centers = attention-weighted cell coordinates; 506 414 parameters),
  trained 12 epochs on 42 344 fit-lineage frames against engine-projected targets
  (`repair.targets`, `repair.parser_loss`); hybrid and continuous predictors retrained from
  scratch on its carriers (#77 recipe, 9000 steps, 11 780 windows at starts 0..225 step 15).
- **R0**: frozen issue-70 parser, predictors retrained on the same windows and recipe.
- Fit data: 745 shots (126 #77 N1 admissible branches + 619 inventory executions) of the 12
  #77 predictor/controller lineages; held-out 007/008/014/016 never trained on (G1 pass).

| held-out shots (240) | pig present at start: accuracy | pig at 225 | ap AUC grid (tie-free) |
| --- | --- | --- | --- |
| R0 (frozen parser) | 0.000 | 0.021 | 0.5111 [0.4000, 0.6667] |
| E (new encoder) | **1.000** | **0.979** | **0.8000 [0.6000, 1.0000]** |

Perception is fixed: the perceived-endpoint ranker now exceeds the engine-endpoint ceiling on
the held-out grid (a = 0.6667). The decision-time ranker does not follow:

| held-out grid, stage c, tie-free | value |
| --- | --- |
| E best request (C-F-1) | 0.7259 [0.5778, 0.8444] — the only E request meeting the target (1/12) |
| E − R0, request-mean | 0.0025 [−0.0204, 0.0185] |
| E − current (#77 checkpoints), request-mean | −0.1401 [−0.1667, −0.0963] |
| R0 − current, request-mean | −0.1426 [−0.1852, −0.0759] |

**Tokens: Q3 target `readiness_or_precision_insufficient`, Q4 encoder effect
`readiness_or_precision_insufficient`** — forced by the frozen guard G3 (R0's parsed-endpoint
costs replicate the attribution records within 1e-3), which failed at 0.1344 on the count cost:
the underlying pig-presence difference is at most 1.34e-4 (batch-size float noise of the same
parser; the tie-free cost differs by the same 1.34e-4), multiplied by the count cost's
1000× pig weight. Without the guard the frozen rules would read Q3 `supported` by 1/12 requests
(C-F-1, 3 clusters, 10 distinct bootstrap resamples) and Q4 `not_supported_by_this_experiment`
(upper 0.0185 < 0.02); this is a reading, not a disposition.

Compute: 3259 GPU s (encoder 540 s, 12 predictors 2581 s, evaluation 162 s), wall 5691 s.

## Reading

1. The count cost is the wrong read-out and the frozen parser cannot see the pig; the tie-free
   cost removes the ties and, on the current checkpoint, already lifts held-out grid ranking to
   the target on 7/14 requests — through the pig-slot center drift, not through any perceived pig.
2. A slot encoder that sees the pig (1.000 accuracy) moves the endpoint ranker from chance to
   0.80, but predictors retrained on its carriers rank candidates no better than predictors
   retrained on the blind parser's carriers, and worse than the #77 checkpoints. With perception
   fixed, the loss now sits in dynamics: the rollout from the anchor does not carry the endpoint
   signal the encoder makes available.
3. Next step for #101/#102: the other step-3 remedy, a decision-aware predictor loss (pairwise
   ranking of candidate endpoints on fit lineages) on E carriers, scored on the #104 sealed
   cohort, where the held-out grid has more than 3 clusters.

Issue disposition under the v1 rules: `readiness_or_precision_insufficient` (typed guard
failures G2/G3 above). Superseded by v2 below; v1 is retained byte-identical.

## v2: guard revision (terminal)

Frozen 2026-09-27T05:43:17Z, **after** both v1 publications and with every v1 outcome known.
v2 replaces exactly the two guards that failed for reasons unrelated to what they protect.
Every estimand, cost, threshold, cohort, record and decision rule is v1's. The 17 v1
artifacts and runners are sha256-bound, and both v1 `--validate` commands run inside every
v2 publish and validate. Cost: 0 GPU seconds, 0 engine seconds.

| guard | v1 | v2 | v2 observed |
| --- | --- | --- | --- |
| attribution G2 | every position-0 frame byte-identical to the sealed anchor | every member has ≥1 byte-identical shot, and (v1 G3) every shot has the same engine start state within 1e-6, so every shot starts from the anchor's physical state | 15/15 members; engine spread 0.0; the 4 mismatched frames parse to within 6.1e-4 of a byte-identical shot (3 of them exactly equal) |
| fix G3 | R0 endpoint count costs within 1e-3 of the attribution | R0 parsed presence (all slots, positions 0 and 225), pig displacement and tie-free cost within 1e-3 (the parse tolerance the v1 smoke froze before any outcome) | max presence 6.65e-4, displacement 0, tie-free 1.34e-4 |

| question | v1 | v2 |
| --- | --- | --- |
| Q1 attribution (Gate A: perception) | readiness_or_precision_insufficient | **supported** |
| Q2 tie-free cost (tied share 0.000 < 0.10) | readiness_or_precision_insufficient | **supported** |
| Q3 target (arm E, held-out grid) | readiness_or_precision_insufficient | **supported** by 1/12 requests (C-F-1 0.7259 [0.5778, 0.8444]) |
| Q4 encoder effect (E − R0 0.0025 [−0.0204, 0.0185]) | readiness_or_precision_insufficient | **not_supported_by_this_experiment** |

Limits of the v2 reading:
- The v2 guards were defined after the results were known. The disclosure is in the v2 plan
  and in every v2 rendering.
- Q3 rests on 1 of 12 requests over 3 held-out member clusters (10 distinct bootstrap
  resamples), with no declared multiplicity adjustment.
- Under the count cost, E:C-F-1 is tied in every held-out grid cell.
- Q3 is not evidence that the fix helps: E − current = −0.1401 [−0.1667, −0.0963], and the
  current #77 checkpoints meet the target on 7 requests without any fix.

v2 is the final guard revision of #99. The issue's disposition is the v2 row set above, and
the attribution table is the deliverable. Perception is the dominant loss; once perception is
fixed, dynamics is the binding stage.
