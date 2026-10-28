# #140 decisions

## Decision P-adapter (2026-10-29): U-native recipe

**Trigger.** #127 budget safeguard fired for A4 (#128, disposition 1: A4 − U `supported`). A4 trains for 2 434.73 GPU-s per seed (mean of #128's published `training_gpu_seconds` 2 424.78 / 2 429.17 / 2 450.24). U's B1 training compute is 1 294.19 GPU-s per seed (`eb._reference_training("unit")`: the #112 v3 retrain training-group reports `LC--hybrid--group{0,1,2}.json`, 1 290.29 / 1 299.88 / 1 292.38 GPU-s, 9 000 updates each). The ratio is 1.88. This child trains U to A4's budget and scores it the way U's Gate-B records were scored. This comment fixes the recipe before any U-native training code runs.

**U's recipe, unchanged except for the update count and the step-indexed schedules.**
- **Unit.** The Gate-B unit F-15-macro: arm `LC`, family `hybrid`, fold `all`. Architecture `s99.new_predictor("hybrid")`, 1 837 690 trainable parameters. Initialisation `torch.manual_seed(seed)`.
- **Code.** The frozen v3 `train_group` (`run_decision_relevant_dynamics.train_group`), reached through a fresh copy of `rt.implementation()`:
  - same arm definitions, loss constants, `SkippingEnsemble` (patience 10), truncated recursion (45 transitions) and logging ensemble;
  - `torch.compile`'d vmapped transition, as in the retrain;
  - only the output, identity, schemas, step count and the three schedule constants below differ.
- **Data.** The #109 fit split exactly as the #112 v3 retrain consumed it: `.local-artifacts/issue-112-v3-retrain-v1/data/training-data.pt` (sha256-pinned), through `rt.IndexedTrainData(payload, "hybrid")`. That is 13 919 complete shots, 220 286 #99 windows, 69 595 long windows and 680 ranking cells. No policy-split or evaluation data.
- **Objective.** The #77 local loss, plus long-horizon supervision (weight 1.0, batch 64), plus the decision-contrastive ranking loss (weight 0.1 × ramp, 2 cells per step, temperature 0.1, tie-free cost). Truncated back-propagation every 45 transitions. Skip-on-nonfinite.
- **Optimiser.** AdamW over stacked parameters at constant lr 1e-4 (no LR schedule exists in this recipe), weight decay 1e-4, per-member gradient clip 1.0. Batch: 64 windows per update.
- **Request schedule.** The hybrid request of update k is `t96.FIXED_ARMS[k % 9]`. This is a round-robin, not a step-indexed schedule, so it is kept as is.
- **Generators.** Windows `seed`, long windows `seed + 112001`, ranking cells `seed + 112002`.
- **Seeds.** 20260908, 20260909, 20260910. One member per process.

**Budget scaling.** Let f = 2435.0 / 1294.19 = 1.881486.
- **Updates.** S′ = 9 × round(1000 × f) = 9 × 1881 = **16 929** (U: 9 000). I keep whole 9-request cycles so that every request trains equally often, as in U (1 000 cycles → 1 881 cycles). Realised update ratio: 16 929 / 9 000 = 1.881.
- **Stretched schedule constants.** Every step-indexed constant c becomes round(c × f):
  - Long-horizon curriculum `every`: 500 → **941**. The horizon is min(225, 60 + 15 × ⌊k / 941⌋), so it reaches 225 at update 10 351. That is 61.1 % of training, as U reaches it at 5 500 / 9 000.
  - Ranking ramp `start`: 1 000 → **1 881**.
  - Ranking ramp `length`: 1 000 → **1 881**. The weight is 0 before update 1 881 and reaches 1 at update 3 761 (U: 1 000 → 1 999).
- **Logging and checkpointing are not training schedules and stay as they are.** These are `TRACE_EVERY` 900 and `PROGRESS_EVERY` 1 000. Under SIGTERM a progress checkpoint is also written after the current update, then the process exits 75. Resume uses `train_group`'s own progress state (ensemble, AdamW, the three generators, trace, GPU-s).
- **End-state guards: U's own rules.**
  - The #117 excursion rule, reached through `rt.excursion` (ratios over [0.6 n, 0.95 n); last 3 updates per request; spike window 500), applied to the full 16 929-update log.
  - A member is *usable* iff it is complete (not retired) and shows no end-state excursion. This is the retrain handoff rule; otherwise the member is a typed failure.
  - G2 (skip count) is reported with the cap scaled the same way: round(45 × f) = 85.

**Measurement.** Each seed's training GPU-s is `train_group`'s own `gpu_seconds`, the same metric as U's 1 294.19 (compile warm-up included, carried across pauses). It will not equal A4's 2 434.73 exactly. I will publish:
- the realised ratios per seed and as means: U-native / U (1 294.19) and U-native / A4 (2 434.73);
- the projection from U's measured rate, 0.1438 s per update × 16 929 ≈ 2 434 s per seed (≈ 0.68 GPU-h; 3 seeds ≈ 2.03 GPU-h).

**Selection.** None. U's configuration is fixed by Gate-B; only the budget changes. Nothing is tuned, re-run or re-selected after any outcome is seen.

**Scoring.** U's own Gate-B path: `gb.score_record` (`gb.request_block` → `m99.rollout_endpoints` with `t96.arm_spec("F-15-macro")`, `harness.EndpointCosts(wlc.load_objective())` rows against the anchor, tie-free primary).
- Every mixed cell of every inventory is scored from its Gate-B anchor.
- Wrong-anchor rows come from the 480 fit anchors.
- The run is one deterministic pass on GPU 0, after the readouts freeze.
- Validation follows `gb.spot_checks`: `harness.endpoint_replication`.

**Guards.**
- Arm guards: G-rank (480 fit anchors × 16 grid launches, tie-free), G-action, G-nonfinite, G-trained (usable per the rule above).
- Extra guard: G-port. The U-native code reproduces U's B1 retrain code bit-exactly for the first 18 updates on fixed CPU inputs (the real fit payload, eager), with identical parameters, AdamW moments and per-update losses and gradient norms. The schedule constants first differ at update 500, so the first 18 updates must match exactly.
- Global guards: G-isolation, G-chronology, G-scoring-policy, G-records.

**Contrasts.** The standard U-native − U / − C\* / − P / − J_dec rows. **A4 − U-native** in both directions, on #128's published A4 records (hash-checked against #128's records manifest), guarded by both arms' guards and both children's global guards. U-native − U is also reported in a descriptive block ("what the extra budget buys").

**Reading** (#140 body). If A4 − U-native is still `supported`, A4's budget qualification is dropped. Otherwise A4's wording stays budget-qualified. The result is reported beside #128's table either way.

Posted: https://github.com/Sino-Huang/NovPhy/issues/140#issuecomment-6479886189
