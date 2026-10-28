## Decision P-follow-up (2026-10-29): distance-stratum cut points (frozen before training)

**Trigger.** Memorisation safeguard 2 needs the tertile cut points of the policy-to-fit pixel distance, frozen before training and computed from policy and fit data only.

**Options considered.** (a) All 120 policy anchors: cut points 0.3311 / 0.5050. (b) The 94 policy grid mixed cells: the selection population, which matches the 311 sealed grid mixed cells the strata are read on.

**Choice: (b).** A level's distance uses #128's NN-px metric: the mean absolute difference in grey levels (uint8, over 3 × 240 × 320) between the level's anchor frame 0 and the nearest of the 480 fit anchor frames (`run_issue_128_direct_predictors.pixel_orders`, exact integer sums). The cut points are `numpy.quantile(d, [1/3, 2/3])` (linear) over the 94 policy cells: **c₁ = 0.3246440972222222, c₂ = 0.4763107638888889** (the range of d is 0.0750–1.1117; computed on the CPU on 2026-10-29 before any F1 / F1-D training). Strata: near if d ≤ c₁, mid if c₁ < d ≤ c₂, far if d > c₂.

**Application.** `--score` (after the readout freeze) computes the same distance for each sealed evaluation level from its Gate-B anchor frame (`eb.read_evaluation_anchors`). It writes the distance and the stratum to `strata/evaluation-distances.json`, which is a records-manifest pattern. Strata are reported on the 311 grid mixed cells for F1, F1-D, U+head, A1, NN-px and NN-car, with U as context. Each stratum gets `eb.ranker_table` blocks (level-clustered bootstrap) and cell counts, with no tokens; no stratum is selected. The cut points are stored in `strata/cut-points.json` (written before the selection runs and pinned in `plan.json`).
