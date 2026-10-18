# Issue #117 item 5: request interference on CLEVRER and Physion-Dominoes

Parent: #108. The shared rules of #85 and the #111 scoring contract apply. The #79 and #84 artefacts are read-only inputs.
Items 1–3 (NovPhy) are in `docs/issue-117-request-interference.md`.

| | |
| --- | --- |
| Runner | `scripts/run_request_interference_benchmarks.py`; tests `tests/test_issue_117_item5_benchmarks.py` |
| Artifacts | `.local-artifacts/issue-117-item5-benchmarks-v1/` (every table is in `findings.md`; also `compute.json`, `comparisons.csv`, `records-manifest.json`) |
| Freeze | Plan frozen 2026-10-17T16:09:11Z in commit 845215d2, before any training update and any score |
| Validation | `python -u -m scripts.run_request_interference_benchmarks --validate` (exit 0); `pytest tests/test_issue_117_item5_benchmarks.py` (7 passed) |

Zero engine seconds. Leave-one-scene-out (LOSO) results are EXPLORATORY. In-sample results are DESCRIPTIVE.

Per scene, the estimator is the log of the ratio of seed-mean errors (the joint's mean over the three seeds divided by
the specialist's), averaged over scenes; the interval is a scene-clustered percentile bootstrap of those scene values
(10 000 draws, PCG64 7201), reported as a geometric ratio. It covers scene sampling, not training-seed variation. The
three per-seed ratios are listed beside every interval in `findings.md`.

## Question

Items 1–2 of #117, repeated on the two out-of-family benchmarks. The question is what sharing every weight of the
six-request hybrid predictor costs each request. "Six requests" means Δ ∈ {1, 5, 15} × {continuous, micro}; neither
benchmark has a macro level. The cost is measured in recursive rollout error against a matched single-request
specialist, and its asymmetry is reported across Δ and description.

## Design

**Carriers.** The existing #79 (CLEVRER) and #84 (Physion-Dominoes) annotation-derived carriers:
- 12 scenes each (CLEVRER: validation scenes 10000–10011; Physion-Dominoes: trials 0–11 of the release prefix);
- 40 × 61-frame training windows per scene;
- the null action throughout.

**Schedule.** Requests run round-robin, `step % 6`, which is the #79 / #84 hybrid schedule.

**Arms.** All are `CNNHybridPredictor` with 1 837 690 parameters:
- **J**: joint, all six requests;
- **D1 / D5 / D15**: per-Δ, two requests each;
- **S-\***: six single-request specialists.

**Budgets.**
- **request**: exactly the joint run's 1500 updates per request, each at its joint step;
- **full**: 9000 updates.

**Recipes.** Both were declared before training.
- **B**: the #77 recipe, exactly as #79 / #84 ran it. Four local transitions plus their recursion and the carrier bound;
  0.1 × the symbolic losses for micro requests; batch 64; AdamW 1e-4 / 1e-4; clip 1.
- **LC**: #112 v3 LC, adapted to the 120-frame endpoint.
  - It is B plus the long-horizon recursive loss (weight 1, 64 rows).
  - The curriculum runs from 30 to 120 frames, +15 every 900 updates. It caps at update 5400, the same 0.6 of training
    as NovPhy's 60 → 225, +15 every 500.
  - Each scene has 8 long-window starts, evenly spread over [0, n − 121], so every target is a real frame.
  - Back-propagation is truncated every 45 transitions, and nonfinite updates are skipped (patience 10).
  - The decision-contrastive ranking term is dropped, because neither benchmark has engine-verdicted candidates.

**Folds.** 12 LOSO folds plus fold 'all', × 3 seeds (20260908–10). That gives 39 members per unit, 76 units and
2964 members.

**Scoring.**
- Each request is rolled out recursively with a fixed pair, from the true carrier at contexts 0–3.
- The metric is position MSE from the #79 `field_errors`, the ICLR `tab:res:rollout` metric. It is taken at the #79 /
  #84 endpoints 15, 30, 60 and 120, and the declared endpoint is 120.
- Fold-s members are scored on scene s (the LOSO cohort, primary).
- Fold-'all' members are scored on all 12 scenes. This is the in-sample cohort, the setting of #79 / #84 and of the
  ICLR rows quoted in #117.

**Smoke** (before the freeze):
- The J / B arm reproduces the #79 / #84 hybrid training loop: max parameter delta 1.7e-6 / 4.8e-7 after 12 updates.
- The batched scorer reproduces the published #79 / #84 evaluation records: max relative delta 2e-5.
- The projection was 8.1 GPU-h, 2.0 h on 4 GPUs.

**Pre-declared 1.5× disposition.** The claim, per benchmark, recipe and budget, is "sharing all weights costs no more
than 1.5× per-request rollout error at t = 120 against a matched specialist". The endpoint is t = 120, the frozen
benchmark-specific endpoint of the #79 / #84 clips and metric. The ticket's literal t = 225 is not delivered on these
benchmarks; this is a declared adaptation, not t = 225 compliance.
- Per request, on LOSO:
  - `supported` iff the joint / specialist upper bound < 1.5;
  - `not_supported_by_this_experiment` iff the lower bound > 1.5;
  - otherwise `readiness_or_precision_insufficient`.
- The overall token is `supported` iff every request is supported.
- The headline per benchmark and recipe is the per-request budget, the negative-transfer contrast, as on NovPhy.
- A failed guard makes every affected token `readiness_or_precision_insufficient`: the overall token and every
  per-request token of the scope. The global guards include the GPU-seconds cap.

## Results

**Guards.** G1 (fold pools exclude their scene), G2 (complete schedules), G3 (#111 policy; re-scored records equal the
retained ones), G4 (no LOSO cell without a finite t = 120 error), chronology and caps all pass.

**Training.** No member retired or skipped an update; no update had a nonfinite loss or gradient norm.

### Item 1. 1.5× disposition and joint / specialist ratio at t = 120 (LOSO)

| benchmark | recipe | per-request budget: token | request-mean ratio | largest upper bound | full budget: token | request-mean ratio | largest upper bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLEVRER | B | **supported** | 0.04 [0.03, 0.04] | 0.66 | readiness_or_precision_insufficient | 0.91 [0.73, 1.12] | 2.39 (F-1-micro 1.41 [0.79, 2.39]) |
| CLEVRER | LC | **supported** | 0.37 [0.33, 0.42] | 0.87 | supported | 0.72 [0.58, 0.87] | 1.20 |
| Physion-Dominoes | B | **supported** | 0.08 [0.06, 0.10] | 0.56 | not_supported_by_this_experiment | 3.78 [3.03, 4.71] | 102 (F-1-micro 77.8 [58.5, 102.3]) |
| Physion-Dominoes | LC | **readiness_or_precision_insufficient** | 0.13 [0.08, 0.22] | 3.08 (F-15-continuous 1.58 [0.79, 3.08]) | not_supported_by_this_experiment | 0.85 [0.46, 1.47] | 9.86 (F-15-continuous 4.81 [2.32, 9.86]) |

Per-request ratios at t = 120 (LOSO):

| benchmark / recipe / budget | F-1-cont | F-1-micro | F-5-cont | F-5-micro | F-15-cont | F-15-micro |
| --- | --- | --- | --- | --- | --- | --- |
| CLEVRER B / request | 0.01 [0.01, 0.02] | 0.00 [0.00, 0.00] | 0.02 [0.02, 0.04] | 0.01 [0.01, 0.02] | 0.54 [0.45, 0.66] | 0.39 [0.31, 0.47] |
| CLEVRER B / full | 0.32 [0.16, 0.67] | 1.41 [0.79, 2.39] | 1.09 [0.94, 1.26] | 1.10 [0.89, 1.33] | 1.10 [0.88, 1.34] | 0.91 [0.73, 1.14] |
| CLEVRER LC / request | 0.10 [0.07, 0.15] | 0.12 [0.07, 0.19] | 0.64 [0.52, 0.80] | 0.67 [0.57, 0.83] | 0.72 [0.60, 0.87] | 0.71 [0.63, 0.79] |
| CLEVRER LC / full | 0.61 [0.40, 0.86] | 0.49 [0.27, 0.85] | 0.49 [0.18, 1.06] | 1.04 [0.87, 1.20] | 0.92 [0.73, 1.16] | 0.98 [0.83, 1.19] |
| Physion B / request | 0.22 [0.19, 0.26] | 0.07 [0.06, 0.10] | 0.03 [0.02, 0.06] | 0.01 [0.00, 0.01] | 0.32 [0.17, 0.56] | 0.22 [0.14, 0.33] |
| Physion B / full | 4.90 [2.73, 8.72] | 77.8 [58.5, 102.3] | 1.25 [0.79, 1.98] | 1.98 [1.60, 2.45] | 1.92 [1.09, 3.35] | 1.62 [1.25, 2.06] |
| Physion LC / request | 0.01 [0.00, 0.01] | 0.00 [0.00, 0.01] | 0.40 [0.25, 0.74] | 0.39 [0.24, 0.70] | 1.58 [0.79, 3.08] | 0.97 [0.49, 1.85] |
| Physion LC / full | 0.31 [0.04, 1.17] | 0.06 [0.02, 0.15] | 2.33 [1.43, 3.97] | 0.86 [0.62, 1.27] | 4.81 [2.32, 9.86] | 2.20 [1.54, 3.45] |

**Per-request budget.** At the per-request budget, the joint model's error at t = 120 is below every specialist's on
every request, for three of the four benchmark × recipe cells. The exception is Physion-Dominoes LC at Δ = 15: F-15-continuous is
1.58 [0.79, 3.08] (per seed 1.52, 1.58, 2.09) and F-15-micro is 0.97 [0.49, 1.85]. Their intervals straddle 1.5, so
the token reads `readiness_or_precision_insufficient`.

**Full budget.** Here the specialist makes 6× the joint's updates of its request.
- CLEVRER LC stays `supported`.
- CLEVRER B is insufficient on F-1-micro only.
- On Physion-Dominoes the 9000-update specialists beat the joint:
  - B at Δ = 1: the S-1-micro error at 120 is 2.5 against the joint's 205.
  - LC at Δ = 5 / 15.

**Every endpoint.** `findings.md` gives J / S at t = 15, 30, 60 and 120 for every request, and the per-request rollout
error of J, D and S at both budgets. Two patterns stand out:
- At the per-request budget the joint's advantage grows with the endpoint at Δ = 1 and 5. Example: CLEVRER B F-1-cont
  goes from 1.00 at t = 15 to 0.01 at t = 120.
- Under LC on Physion-Dominoes the joint is 8–21× worse than the Δ = 5 / 15 specialists at t = 15. In absolute terms
  that is about 1e-3 against 1e-4. The joint's error settles at about 0.01 for every request at t = 120, so the gap
  closes by then.

### Item 2. Asymmetry (joint / specialist at t = 120, LOSO)

| benchmark | recipe / budget | Δ = 1 | Δ = 5 | Δ = 15 | continuous | micro | Δ1 / Δ15 | continuous / micro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLEVRER | B / request | 0.01 | 0.02 | 0.46 | 0.05 | 0.02 | 0.01 [0.01, 0.02] | 2.06 [1.76, 2.39] |
| CLEVRER | B / full | 0.68 | 1.10 | 1.00 | 0.73 | 1.12 | 0.67 [0.38, 1.16] | 0.65 [0.51, 0.83] |
| CLEVRER | LC / request | 0.11 | 0.65 | 0.71 | 0.36 | 0.39 | 0.15 [0.11, 0.21] | 0.93 [0.77, 1.12] |
| CLEVRER | LC / full | 0.55 | 0.72 | 0.95 | 0.65 | 0.79 | 0.57 [0.33, 0.93] | 0.82 [0.56, 1.17] |
| Physion-Dominoes | B / request | 0.13 | 0.01 | 0.26 | 0.13 | 0.04 | 0.48 [0.28, 0.87] | 2.97 [2.20, 4.04] |
| Physion-Dominoes | B / full | 19.5 | 1.57 | 1.77 | 2.27 | 6.30 | 11.1 [8.0, 15.1] | 0.36 [0.20, 0.62] |
| Physion-Dominoes | LC / request | 0.00 | 0.40 | 1.24 | 0.15 | 0.11 | 0.00 [0.00, 0.01] | 1.42 [1.13, 1.85] |
| Physion-Dominoes | LC / full | 0.13 | 1.42 | 3.26 | 1.51 | 0.47 | 0.04 [0.02, 0.09] | 3.19 [1.28, 6.87] |

- **Δ = 1 does not degrade more than Δ = 15 at the per-request budget**, on either benchmark under either recipe. In
  all four cells the Δ1 / Δ15 upper bound is below 1, and Δ = 1 gains the most from sharing, as on NovPhy. The ICLR
  CLEVRER Δ = 15 hybrid / continuous-only ratio of 1.47 is a "shared six versus shared three" contrast. At matched
  updates, the joint Δ = 15 here beats its specialist on CLEVRER (0.46 B, 0.71 LC).
- **Where sharing costs.**
  - *Physion-Dominoes LC at Δ = 15*: 1.24 at the request budget, 3.26 at full. Splitting the comparison at the per-Δ
    model shows the cost comes from sharing across Δ: J / D15 is 1.67 [0.87, 3.10] (request) and 2.95 [1.84, 4.85]
    (full). Sharing across descriptions costs nothing: D15 / S is 0.74 [0.61, 0.95] and 1.10 [1.00, 1.24].
  - *Physion-Dominoes B at Δ = 1, full budget*: 19.5. A Δ = 1 specialist given 9000 updates beats the joint's 1500.
- **By description.**
  - At the per-request budget, micro gains more than continuous under B (continuous / micro 2.06 CLEVRER, 2.97
    Physion-Dominoes).
  - Under LC the two descriptions gain about equally on CLEVRER (0.93 [0.77, 1.12]). On Physion-Dominoes continuous
    gains less (1.42 [1.13, 1.85]).

### In-sample (the ICLR setting, DESCRIPTIVE)

**Replication.** Fold-'all' J under B retrains the #79 / #84 hybrid on the same 12 scenes and draws, and reproduces it
(position MSE at 120):

| | Δ = 5 / 15 (here vs published) | Δ = 1 continuous (here vs published) | published per-seed range at Δ = 1 |
| --- | --- | --- | --- |
| CLEVRER | 0.047 / 0.017 vs 0.044 / 0.017 | 3.73 vs 2.98 | 2.51–3.64 (3.73 lies slightly above it) |
| Physion-Dominoes | 0.016 / 0.001 vs 0.016 / 0.001 | 255 vs 167 | 91–274 (255 lies inside it) |

**Ratios.** The in-sample request-mean joint / specialist ratio at t = 120 is below 1 at the per-request budget:
- CLEVRER: B 0.04, LC 0.33;
- Physion-Dominoes: B 0.12, LC 0.12.

At the full budget it is:
- CLEVRER: B 0.87, LC 2.28 [1.77, 2.83];
- Physion-Dominoes: B 6.74, LC 0.70.

On CLEVRER LC, specialists that make 6× the joint's updates fit the 12 training scenes better than they generalise to
a left-out scene: the LOSO full-budget ratio is 0.72.

### Ranking AUC

`unsupported`. Neither benchmark has an engine-verdicted candidate inventory. CLEVRER and the Physion-Dominoes prefix
ship passive trials, and the #79 / #84 carriers hold the null action only.

## Reading

Sharing all weights costs no more than 1.5× per-request rollout error at t = 120 against a matched specialist on:
- CLEVRER, under both recipes (`supported`);
- Physion-Dominoes, under B (`supported`).

Under the adapted LC recipe on Physion-Dominoes the token is `readiness_or_precision_insufficient`. Those Δ = 15
intervals straddle 1.5. The cost comes from sharing across Δ (J / D15 1.67 [0.87, 3.10]), not across descriptions
(D15 / S 0.74 [0.61, 0.95]).

Together with NovPhy (`supported` for B, LC1 and LC3 at the per-request budget), the "Δ = 1 is hurt by sharing"
reading does not hold on any benchmark at matched per-request updates. Δ = 1 is the request that gains most.

At the full budget, specialists that make 6× the joint's updates of their request can beat the joint. This is an
update-allocation comparison, consistent with what #116 also found; it does not isolate a cause. At matched updates per
request, the 1.5× criterion is supported in three of the four benchmark × recipe cells. Physion-Dominoes LC remains
unresolved at Δ = 15: F-15-continuous is 1.58 [0.79, 3.08], so a cost above 1.5× is not excluded there. Even where the
criterion is supported, it bounds the cost below 1.5×; it does not show zero negative transfer.

## Disclosures

- **Hardware.** 4× RTX 5090, one process per GPU, against #85's single RTX 3090. The joint arm is retrained here, not
  reused from #79 / #84.
- **Launch incident.**
  - The training PIDs first recorded, and posted on #117, were the nohup wrapper shells. The real workers were
    2013294 / 2013343 / 2013392 / 2013441.
  - I took the wrappers' exit to mean the workers had died and started a second launch. It ran for about 4 minutes on
    the same GPUs, and I killed it.
  - None of its units reached a checkpoint or finished. The claim files kept every unit to a single trainer, and the
    first launch picked up its stale claims.
  - Its logs are in `logs/duplicate-launch-killed/`, and the correction is posted on #117. Its about 16 GPU-minutes
    are not in the ledger.
  - Scoring was launched once (workers 2052693 / 2052695 / 2052697 / 2052699), and every record was scored once.
- **LC adaptation.** It was declared in the frozen plan before training. The ranking term is absent, so LC here is the
  #112 long-horizon recipe with the v3 stability changes, not the full Gate-B objective.
- **Cohorts.** The 12 scenes per benchmark are the only existing carriers. LOSO is therefore the primary cohort, and no
  held-out scenes exist. The in-sample cohort repeats the ICLR setting and is descriptive.
- **Item 3** (training-dynamics probes) is not part of item 5. Every update's loss and pre-clip norm are logged
  (`updates/`, local), and none was nonfinite.
- **Post-publication runner edit (phase-B/C review).** Commit `5dbd04df` changed
  `scripts/run_request_interference_benchmarks.py` for guard propagation and reporting only. The frozen caps guard now
  enters `global_pass`, `guards_pass` and the disposition stop condition (before, a publish-time cap breach would have
  left `guards_pass: true`; training and scoring already raised on a breach). When a scope's guard fails, every
  per-request token of that scope is now the stop token too, not only the overall token. The validation log now says
  that `compute.json` is compared by value. Every retained guard passes, so no evidence, number or token changed:
  `--validate` regenerates every published file and exits 0. Tests force each global guard (the cap included) and one
  scope guard false on the published `compute.json`. The runner is listed in `scripts/post_freeze_amendments.py`.
- **Wording corrections (phase-B/C review).** The interval description now names the estimator: per scene, the log
  ratio of seed-mean errors. "At matched updates per request there is no negative transfer" is replaced by the
  qualified reading above. The #117 result comment's "Δ = 1 lands inside the published seed spread: CLEVRER 3.73 vs
  2.51–3.64" was wrong: 3.73 lies slightly above that range. Physion's 255 lies inside 91–274.

## Compute

| | |
| --- | --- |
| Training | 24 956 GPU s (LC 20 099, B 4 857); about 1 h 45 min wall on 4 GPUs |
| Scoring | 317 s |
| Ledger total | 25 272 GPU s (projection 28 986) |
| Engine | 0 s |
| Peak memory | 14.3 GB per 39-member group |
| Stays local | Checkpoints (21 GB), update logs (311 MB), records (65 MB) |
| Committed | Plan, smoke, ledger, training reports, worker logs, `compute.json`, `findings.md`, `comparisons.csv`, `summary.json` and the records manifest (sha256 of every record, training report and update log) |
