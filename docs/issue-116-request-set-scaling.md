# Issue #116: request-set scaling (denser and continuous Δ, larger |R|, DP-labelling and policy cost)

Parent: #108 (pass branch). The shared rules of #85 apply, and the #111 scoring contract is adopted. The #101, #104,
#109, #112, #115, #117 and Gate-B records are read-only inputs. Nothing here edits the ICLR 2026 submission.

| | |
| --- | --- |
| Runner | `scripts/run_request_set_scaling.py` |
| Module | `world_model/training/request_sets.py`: the declared request set R (indexing, predictor class, schedules, budgets, curriculum clock, local windows, the endpoint rule, the v3 training loop, the DP over R, the batched DAgger, the rollout over R, the policy heads). #119 and #121 can reuse it. |
| Tests | `tests/test_issue_116_request_set_scaling.py` (33 tests): the R_9 replication guard, request indexing, the endpoint rule, the vectorized / chunked / pruned DP, the heads, the disposition rules |
| Artifacts | `.local-artifacts/issue-116-request-set-scaling-v1/`: `findings.md` (every table), `compute.json`, `summary.json`, `records-manifest.json`, `plan.json`, `smoke.json`, `training/`, `labels/*.json`, `logs/`, `ledger/` |
| Freeze | Plan frozen 2026-10-17T03:02:25Z, commit 730a1e4e, after a passed smoke and before any training update, label or score |
| Validation | `python -u -m scripts.run_request_set_scaling --validate` (exit 0); `pytest tests/test_issue_116_request_set_scaling.py` |

## Results (published 2026-10-17)

Scored once on the #104 sealed #109 evaluation split (Gate-B passed, so the ranking comparison across sets is
confirmatory). Tie-free cost, every rollout, label and score under `deterministic_scoring()`. Intervals are
level-clustered percentile bootstraps (10 000 draws, PCG64 7201), DESCRIPTIVE; tokens follow the pre-declared δ rule
(δ = 0.02: `supported` iff mean ≥ δ and lower bound > 0). Every table, including the per-inventory AUCs, the hindsight
fixed requests and the training table, is in `findings.md`; the numbers below are copied from it.

- **Guards G1–G7: all pass.** G5 re-checked 13 824 R_9 fixed-request blocks against the Gate-B and #101 records (equal).
  G7: the J-flat grid nonfinite share is 0 for every configuration.
- **Replication guard (Work 1):** passed bit for bit in the smoke (training, Gate-B / #101 fixed rows, #101 J, #101 DP
  rows; see Design) and again in scoring through G5.
- **Records.** 42 records (7 configurations × 3 seeds × {fixed, policies}), 21 label reports and 18 training reports.
  `records-manifest.json` holds the SHA-256 of all 101 raw files (records, label reports, round-0 label files, update
  logs). Every scored model is kept.
- **Typed failure (retained, not retried).** R_18/request seed 20260908 is unusable: `final_checkpoint_excursion` (the
  #117 end-of-training rule; first spike at update 17 864 of 18 000, request F-3-macro, pre-clip norm 40 373; 0 skipped
  updates). Its label report and both records are typed-failure records; R_18/request is scored on 2 seeds. The other
  17 new members are usable with 0 skipped updates.
- **Validation.** `--validate` exits 0 (130 s): it re-derives `compute.json`, `summary.json`, `findings.md` and the
  manifest byte for byte, and re-scores R_cont/total J-flat (seed 20260908) and R_18/request F-2-macro (seed 20260909)
  on every grid cell with exact rows. `pytest tests/test_issue_116_request_set_scaling.py`: 33 passed.

### Dispositions

1. **"A larger or continuous Δ set improves ranking over R_9": `readiness_or_precision_insufficient`.** No set is
   supported under both budgets, and not every (set, budget) is `not_supported_by_this_experiment`.

   | set / budget | J-flat grid AUC | minus R_9 J-flat | token |
   | --- | --- | --- | --- |
   | R_9 (Gate-B joint, both budgets) | 0.678 [0.656, 0.700] | – | – |
   | R_18/total | 0.615 [0.603, 0.627] | −0.063 [−0.077, −0.049] | not_supported_by_this_experiment |
   | R_18/request | 0.638 [0.618, 0.659] | −0.042 [−0.056, −0.030] | not_supported_by_this_experiment |
   | R_30/total | 0.543 [0.527, 0.559] | −0.135 [−0.154, −0.116] | not_supported_by_this_experiment |
   | R_30/request | 0.749 [0.737, 0.761] | +0.071 [+0.049, +0.092] | supported |
   | R_cont/total | 0.699 [0.687, 0.711] | +0.021 [−0.002, +0.042] | readiness_or_precision_insufficient |
   | R_cont/request | 0.947 [0.935, 0.958] | +0.269 [+0.245, +0.293] | supported |

2. **"The DP labeller stays tractable at |R| = 225": `supported`.** Every R_cont predictor's exact round-0 pass
   (N = 3480 windows, L = 225 frames, 225 requests) takes 17.8–18.3 GPU s (cap 3600) with a peak of 1.23–1.26 GB, at
   0.72–0.74 × R_9's seconds per carrier evaluation (cap 2 ×). All six checks pass.
3. **Sub-frame Δ: `unsupported`.** One carrier frame is 50 native engine steps, the #104 / #109 capture stride.

### Scaling table (grid; per set and budget)

| set / budget | updates | DP round-0 GPU s | peak GB | label entropy (bits / max) | J-flat label accuracy (eval) | J-flat rollout error t = 225 | J-flat grid AUC | J-flat MACs / candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R_9 | 9000 | 1.2 | 1.16 | 2.12 / 3.17 | 0.528 [0.523, 0.534] | 0.0272 [0.0262, 0.0283] | 0.678 [0.656, 0.700] | 374.9M |
| R_18/total | 9000 | 2.0 | 1.16 | 3.14 / 4.17 | 0.399 [0.390, 0.407] | 0.0389 [0.0378, 0.0400] | 0.615 [0.603, 0.627] | 290.7M |
| R_18/request | 18 000 | 2.0 | 1.16 | 2.65 / 4.17 | 0.484 [0.476, 0.492] | 0.0881 [0.0852, 0.0911] | 0.638 [0.618, 0.659] | 336.7M |
| R_30/total | 9000 | 3.0 | 1.17 | 3.38 / 4.91 | 0.311 [0.308, 0.315] | 0.2746 [0.2694, 0.2798] | 0.543 [0.527, 0.559] | 295.2M |
| R_30/request | 30 000 | 2.9 | 1.17 | 3.81 / 4.91 | 0.220 [0.218, 0.222] | 0.1129 [0.1047, 0.1213] | 0.749 [0.737, 0.761] | 286.4M |
| R_cont/total | 9000 | 18.1 | 1.25 | 6.91 / 7.81 | 0.136 [0.134, 0.138] | 0.2158 [0.2029, 0.2292] | 0.699 [0.687, 0.711] | 105.5M |
| R_cont/request | 225 000 | 17.9 | 1.23 | 6.11 / 7.81 | 0.149 [0.146, 0.151] | 0.0842 [0.0785, 0.0899] | 0.947 [0.935, 0.958] | 220.5M |

J-flat grid / offset / angle / power AUC (the full per-inventory table, with every head and the fixed-request means,
is in `findings.md`):

| ranker | grid | offset | angle | power |
| --- | --- | --- | --- | --- |
| R_9 | 0.678 | 0.689 | 0.752 | 0.780 |
| R_18/total | 0.615 | 0.580 | 0.570 | 0.720 |
| R_18/request | 0.638 | 0.722 | 0.725 | 0.764 |
| R_30/total | 0.543 | 0.589 | 0.542 | 0.680 |
| R_30/request | 0.749 | 0.869 | 0.876 | 0.829 |
| R_cont/total | 0.699 | 0.733 | 0.716 | 0.720 |
| R_cont/request | 0.947 | 0.904 | 0.954 | 0.777 |
| no-model action prior | 0.720 | 0.500 | 0.812 | 0.500 |

**Contrasts against R_9 (grid, paired by level and seed).** The same predictors' fixed R_9-request mean moves with
J-flat: R_18/total −0.035 [−0.044, −0.027], R_18/request −0.027 [−0.040, −0.015], R_30/total −0.128 [−0.138, −0.119],
R_30/request +0.100 [+0.081, +0.119], R_cont/total +0.069 [+0.056, +0.082], R_cont/request +0.237 [+0.217, +0.256].
J-flat rollout error at t = 225 is higher than R_9's for every set and budget (+0.0117 for R_18/total to +0.2474 for
R_30/total).

### Policy-head comparison

| set / budget | head | parameters | label accuracy exact / Δ / α (eval) | grid AUC | minus flat | rollout error t = 225 | MACs / candidate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R_9 | flat | 32 265 | 0.528 / 0.546 / 0.859 | 0.678 | – | 0.0272 | 374.9M |
| R_9 | factorised | 32 376 | 0.505 / 0.558 / 0.858 | 0.676 | −0.002 [−0.004, +0.000] | 0.0272 | 379.9M |
| R_18/total | flat | 33 426 | 0.399 / 0.465 / 0.779 | 0.615 | – | 0.0389 | 290.7M |
| R_18/total | factorised | 33 525 | 0.393 / 0.479 / 0.790 | 0.601 | −0.014 [−0.019, −0.008] | 0.0367 | 347.9M |
| R_18/request | flat | 33 426 | 0.484 / 0.508 / 0.905 | 0.638 | – | 0.0881 | 336.7M |
| R_18/request | factorised | 33 525 | 0.466 / 0.494 / 0.903 | 0.623 | −0.015 [−0.019, −0.011] | 0.0885 | 347.8M |
| R_30/total | flat | 34 974 | 0.311 / 0.334 / 0.872 | 0.543 | – | 0.2746 | 295.2M |
| R_30/total | factorised | 35 085 | 0.304 / 0.341 / 0.869 | 0.564 | +0.021 [+0.008, +0.035] | 0.2676 | 249.3M |
| R_30/request | flat | 34 974 | 0.220 / 0.242 / 0.734 | 0.749 | – | 0.1129 | 286.4M |
| R_30/request | factorised | 35 085 | 0.195 / 0.247 / 0.748 | 0.739 | −0.010 [−0.016, −0.004] | 0.1568 | 333.5M |
| R_cont/total | flat | 60 129 | 0.136 / 0.169 / 0.719 | 0.699 | – | 0.2158 | 105.5M |
| R_cont/total | factorised | 60 105 | 0.145 / 0.192 / 0.780 | 0.678 | −0.021 [−0.037, −0.004] | 0.2316 | 135.0M |
| R_cont/total | ordinal | 60 099 | 0.072 / 0.101 / 0.793 | 0.694 | −0.004 [−0.018, +0.009] | 0.1487 | 74.8M |
| R_cont/request | flat | 60 129 | 0.149 / 0.166 / 0.894 | 0.947 | – | 0.0842 | 220.5M |
| R_cont/request | factorised | 60 105 | 0.143 / 0.172 / 0.892 | 0.926 | −0.022 [−0.027, −0.017] | 0.0788 | 199.7M |
| R_cont/request | ordinal | 60 099 | 0.077 / 0.087 / 0.890 | 0.925 | −0.022 [−0.030, −0.015] | 0.0685 | 122.6M |

Only R_30/total factorised − flat meets the δ rule (`supported`, +0.021); every other head contrast is
`not_supported_by_this_experiment`. Every head's chosen-Δ median is 1 (10–90 %: 1–1) in every configuration.
Predictor MACs per transition are Δ-independent (continuous 1.4M, micro 1.8M, macro 1.5M).

### Controls (every ranking table)

| set / budget | J-flat − prior (grid) | token | right − wrong anchor | token |
| --- | --- | --- | --- | --- |
| R_9 | −0.042 [−0.069, −0.016] | not_supported_by_this_experiment | +0.003 [+0.001, +0.004] | not_supported_by_this_experiment |
| R_18/total | −0.105 [−0.131, −0.079] | not_supported_by_this_experiment | +0.030 [+0.023, +0.036] | supported |
| R_18/request | −0.082 [−0.115, −0.049] | not_supported_by_this_experiment | −0.001 [−0.004, +0.001] | not_supported_by_this_experiment |
| R_30/total | −0.177 [−0.209, −0.145] | not_supported_by_this_experiment | −0.006 [−0.014, +0.001] | not_supported_by_this_experiment |
| R_30/request | +0.029 [+0.001, +0.056] | supported | +0.162 [+0.150, +0.174] | supported |
| R_cont/total | −0.022 [−0.049, +0.006] | not_supported_by_this_experiment | +0.142 [+0.128, +0.156] | supported |
| R_cont/request | +0.227 [+0.202, +0.251] | supported | +0.384 [+0.371, +0.397] | supported |

### DP labelling cost

| set / budget | requests | round-0 s (L = 225) | L = 60 s | s per 10^6 evaluations | ratio to R_9 |
| --- | --- | --- | --- | --- | --- |
| R_9 | 9 | 1.2 | 0.3 | 0.168 | 1.00 |
| R_18 (total / request) | 18 | 2.0 / 2.0 | 0.5 / 0.5 | 0.144 / 0.146 | 0.86 / 0.87 |
| R_30 (total / request) | 30 | 3.0 / 2.9 | 0.6 / 0.6 | 0.137 / 0.136 | 0.82 / 0.81 |
| R_cont (total / request) | 225 | 18.1 / 17.9 | 2.6 / 2.5 | 0.123 / 0.122 | 0.73 / 0.73 |

The measured cost follows the O(N · L · |R|) evaluation count: a linear fit over all 40 exact passes gives 0.121 s per
10^6 carrier evaluations + 0.2 s (R² 0.999). The round-0 seconds grow with |R| at a log-log slope of 0.87, because a
request makes F − Δ evaluations per window and larger Δ makes fewer.

**Exact vs pruned DP (R_cont).** The pruned DP uses candidate Δ ∈ {1, 2, 3, 5, 7, 11, 16, 23, 34, 51, 75} × α (33
requests, 3.1–3.3 GPU s). Its label disagreement with the exact DP is 0.79–0.85, almost all of it in Δ (Δ disagreement
equals label disagreement; α disagreement 0.02–0.25). 20–29 % of the exact labels fall on the pruned grid. The mean
relative value gap is +2.7 % to +3.1 % (R_cont/total) and +5.3 % to +7.3 % (R_cont/request).

### Reading (descriptive; it does not change the tokens)

- **The update-allocation confound is visible.** The two `supported` cells are the equal-per-request budgets of R_30
  (3.3 × R_9's updates) and R_cont (25 ×). In the same predictors, the fixed R_9-request mean gains +0.100 and +0.237
  over R_9, comparable to J-flat's +0.071 and +0.269: the gain is already in the predictor's fixed R_9 requests, not
  only in the larger action set. At equal total updates, R_18 and R_30 rank worse than R_9, and R_cont/total's
  +0.021 [−0.002, +0.042] has a lower bound below 0. This is the confound the both-budgets rule guards against, and the
  rule returns `readiness_or_precision_insufficient`.
- **The heads hardly use the larger set.** Every head's median chosen Δ is 1. In each enlarged set, J-flat ranks below
  its own fixed-request means (for example, R_cont/request 0.947 against 0.967 for the all-request mean).
- **Ranking and rollout error decouple.** R_cont/request has the highest grid AUC, but its rollout error at t = 225
  (0.0842) is about 3 × R_9's.
- **Exact label accuracy falls with |R|** (0.528 at R_9 to 0.136–0.149 at R_cont). α accuracy stays at 0.72–0.91.
  At matched parameters, the factorised and ordinal heads do not improve on flat, except R_30/total factorised.
- **The DP is cheap at every size.** The R_cont exact pass is 18 GPU s. The pruned grid saves about 15 s, at a 3–7 %
  value gap, and changes about 80 % of the labels.

### Compute

156 486 GPU s in total: training 134 453, labelling 1619, scoring 20 414. Zero engine seconds. Hardware: 4 × RTX 5090
(not the #85 RTX 3090). As in the earlier runners, GPU seconds are the wall seconds of each phase on its own GPU.
Elapsed time: training 03:02:57Z–13:57:56Z (10.9 h), labelling 14:00:44Z–14:09:20Z (9 min), scoring
14:09:41Z–15:45:42Z (1.6 h). Each stage was launched once; worker PIDs are in `logs/{train,label,score}-pids.txt`.

### Disclosures

- **Duplicate training launch.** The first launch at 03:02:57Z was alive; its logs were empty only because the workers
  were still importing. It was misread as dead, and a second launch at 03:03:23Z ran concurrently for about 40 s while
  loading the training payload. It was killed at 03:04:10Z, before its first update, and wrote nothing but directories.
  Its shell redirection truncated `train-worker{0..3}.log`, so those logs start mid-stream. The per-update logs in
  `updates/` (hashed in the manifest) and the training reports come from the first launch only, which is the only
  training launch. `logs/train-pids.txt` records both launches.
- **Training time against the projection.** The smoke projected 54 025 GPU s of training (about 3.8 h per worker) and
  7.1 h elapsed for the whole run. Training took 134 453 GPU s (2.5 ×) and 10.9 h elapsed. The overrun is in R_cont:
  each R_cont/request member took about 33.8k s. Training is CPU-bound, dominated by dynamo dispatch over the 225
  compiled pair graphs, with GPU utilisation under 10 %. The first R_cont member on a worker also paid the per-request
  compile: R_cont/total seed 20260909 took 6240 s, 1927 s of it in first updates, against about 1080 s and 15 s for its
  siblings. Labelling (1619 GPU s against 7359 projected) and scoring (20 414 against 38 191) came in under projection.
  The whole run stayed well within the 48 h limit.
- **First smoke failure.** The first smoke run failed one check: the statistics rehearsal passed every inventory's
  prior statistics into `view_rows` instead of one inventory's, so the no-model prior did not reproduce Gate B. The bug
  was fixed before the freeze and the second smoke passed every check. Neither smoke run trained or scored a new model
  on the evaluation split. The frozen runner (commit 730a1e4e) contains the fix and was not edited afterwards.

## Design (frozen)

### Work 1: the generalised request set and the replication guard

`cnn_hybrid.py`, `request_policy.py` and the #112 runners are hash-pinned, so the generalisation lives in a sibling
module. `request_sets.RequestSet` declares a Δ grid × the three descriptions in Δ-major order. R_9's order is
`cnn_hybrid.PAIRS`. The module generalises the three hard-coded places:
- **Pairs.** `predictor_class(R)` subclasses `CNNHybridPredictor` with a `carrier` that accepts exactly R. The
  parameters (1 837 690), the state dict and the ops are unchanged; `PairConditioner` already encodes any integer Δ.
- **Controller output layer.** Every head indexes R. `RequestPolicy(R)` is the flat head.
- **DP cost tensor.** `prediction_error_costs` has |R| columns, and `backward_dp` vectorises the backward recursion
  over R.

**Replication guard (smoke, before the freeze; every check bit for bit):**
- **Training.** R_9 through `request_sets.train_member` equals the retrain's `train_group`, the path that produced the
  #112 LC / Gate-B checkpoints. Over 12 updates, at the real schedule and at full cost, the parameter delta is 0 and
  the logged local losses and pre-clip norms are identical.
- **Fixed-request rows.** The R_9 predictor class loaded with the Gate-B checkpoints, through the generalised rollout,
  reproduces:
  - the Gate-B unit records (F-15-macro, 3 seeds, every inventory, 480 wrong-anchor blocks);
  - the #101 fixed-request records of the other eight requests (grid).
- **Policy rollout.** The #101 J (pe, λ = 1e-4) reproduces its grid rows and schedules.
- **DP labels.** On four policy levels, the DP labels with the batched DAgger reproduce
  `request_policy.window_rows` in both rounds.

Guard G5 re-checks every R_9 fixed-request row of the scoring stage against the Gate-B and #101 records.

### Work 2: request sets and budgets

| set | Δ grid | \|R\| | updates, equal total | updates, equal per request |
| --- | --- | --- | --- | --- |
| R_9 | 1, 5, 15 | 9 | 9000 (Gate-B joint) | 9000 (the same model) |
| R_18 | 1, 2, 3, 5, 10, 15 | 18 | 9000 | 18 000 |
| R_30 | 1, 2, 3, 5, 8, 10, 15, 25, 45, 75 | 30 | 9000 | 30 000 |
| R_cont | 1 … 75 | 225 | 9000 | 225 000 |

- **Recipe.** The #112 v3 LC recipe (the Gate-B recipe) on the retrain's #109 fit payload (hash-checked), fold 'all',
  seeds 20260908/09/10. Requests are trained round robin in declared order.
- **Curriculum clock.** Update u of an n-update run reads the horizon curriculum and the ranking ramp at
  u × 9000 // n. Each request's k-th update therefore meets the curriculum of R_9's k-th update of it.
- **Δ > 15.**
  - The local term reads a window of max(60, 4Δ) transitions at the drawn start (same index draw), so every Δ keeps
    four local transitions where the shot allows.
  - The long term uses floor(horizon / Δ) transitions; there is none while Δ > horizon.
- **Endpoint rule.** Δ ∤ 225 in R_18, R_30 and R_cont. A fixed request makes floor(225 / Δ) transitions, then the
  #101 same-α fallback (the largest Δ' ∈ R with Δ' ≤ remaining) until exactly 225. Examples: R_18 Δ = 2 → 112 × 2
  + 1; Δ = 10 → 22 × 10 + 5; R_cont Δ = 74 → 3 × 74 + 3. The rule applies to the ranking term in training and to
  every fixed-request scoring rollout. Policies mask Δ > remaining.
- **Usability.** The retrain rule: complete, at most 45 skipped updates per 9000 (scaled), every update logged, and no
  #117 end-of-training excursion. An unusable member is a typed terminal failure.

### Work 3: DP labelling

- **Labels.** The ICLR prediction-error DP (#101 pe):
  - cost c(t, i) = Δ_i × one-step carrier MSE + λ × work(pair_i, flat policy of R) / work(15, continuous);
  - λ = 1e-4.
- **Pool and windows.** The #109 policy split as built by #101 (3480 shots, 120 levels). Each shot gives one start-0
  window of **225 frames**, the deployment horizon (#101 used 60), so every Δ ≤ 75 is feasible.
- **MAC term.** The set's flat policy is used for every head, so round-0 labels depend on (predictor, R) only.
- **Cost check.**
  - Measured passes: round 0 at L = 225, a timed round 0 at L = 60, every head's round 1, and the pruned pass.
  - Each pass is measured against the carrier-evaluation count N × Σ_i (F − Δ_i), which is O(N L |R|).
  - Chunk: 256 windows. The smoke measured 99.76 % label agreement with one-window labelling.
- **Pruned DP (R_cont).** Candidate Δ ∈ {1, 2, 3, 5, 7, 11, 16, 23, 34, 51, 75} × α. This grid is the round-half-up of
  12 geometric points from 1 to 75. The pruned labels are compared with the exact labels on every labelled
  position.

### Work 4: policy heads

- **Flat.** `RequestPolicy(R, 128, clamp60)`, the current design. Its parameters grow with |R|.
- **Factorised.** A trunk, then a Δ head × an α head. The scores are the joint log-probabilities.
- **Ordinal (R_cont).** A trunk, then a cumulative-logit Δ head (one score, ordered thresholds, renormalised over the
  feasible Δ) × an α head.
- **Matched parameters.** The factorised and ordinal widths are set so their parameter counts are closest to the set's
  flat head:

  | set | flat | factorised | ordinal |
  | --- | --- | --- | --- |
  | R_9 | 32 265 (w 128) | 32 376 (w 130) | – |
  | R_18 | 33 426 (w 128) | 33 525 (w 133) | – |
  | R_30 | 34 974 (w 128) | 35 085 (w 137) | – |
  | R_cont | 60 129 (w 128) | 60 105 (w 187) | 60 099 (w 243) |

- **Recipe.** The #101 pe recipe:
  - initialisation with seed + 7700;
  - round 0 on observed-context labels;
  - round 1 on the head's own DAgger contexts;
  - 2 × 1800 AdamW steps, batch 128, unit = policy level.

  The loss is cross-entropy on the joint scores (for the factorised and ordinal heads, the sum of the factor NLLs).
  Policy seed s uses predictor seed s.

### Evaluation (scored once, in processes that did not train)

- **Cells.** The Gate-B mixed cells (grid 311, offset 76, angle 78, power 47 levels), the Gate-B E99 anchors, and the
  tie-free cost. Cells are (level, seed).
- **Arms.** Per configuration and seed:
  - every fixed request of R (endpoint rule);
  - every head's greedy policy J;
  - the wrong-anchor control of J-flat (16 grid candidates from the 480 fit anchors).
- **Rollout error at t = 225.** The #100 `engine_position_mse` of every mixed grid candidate's endpoint against its own
  branch, from the #115 engine-truth store (hash-bound; the smoke reproduces #115's values).
- **Label accuracy.** On the 4976 evaluation grid branches (E99 parses 0..min(225, last) from the same store): the
  exact DP label of the head's predictor against the head's greedy choice, at every position.
- **MACs.** Linear MACs per candidate rollout (predictor + policy), and policy MACs per call.
- **Controls in every ranking table.** The no-model action prior (Gate-B rule) and the wrong-anchor control.

### Dispositions (pre-declared)

1. **"A larger or continuous Δ set improves ranking over R_9."**
   - Per set and budget: grid J-flat(R, b) − J-flat(R_9), paired by (level, seed), with the δ superiority token
     (mean ≥ δ and lower bound > 0).
   - `supported` iff some set is supported under **both** budgets; `not_supported_by_this_experiment` iff every
     (set, budget) is; otherwise `readiness_or_precision_insufficient`.
2. **"The DP labeller stays tractable at |R| = 225."**
   - `supported` iff every usable R_cont predictor's exact round-0 pass over the pool (N = 3480, L = 225) takes at most
     3600 GPU s on one RTX 5090, and its seconds per carrier evaluation are at most 2 × the R_9 mean;
   - otherwise `not_supported_by_this_experiment`.
3. **Sub-frame Δ (Work 6): `unsupported`.** One carrier frame is 50 native engine steps, the #104 / #109 capture stride.
   A shorter Δ has no observed target in any existing capture. A finer-stride capture is outside the frozen #109 /
   #104 plan, and changing it is an owner decision.

A failed guard turns every token into `readiness_or_precision_insufficient`.

### Guards

- **G1 isolation.** The fit payload, the policy pool and the evaluation levels are disjoint; the wrong anchors are fit
  levels.
- **G2 scoring policy.** Every record and label report carries the #111 scoring policy.
- **G3 chronology.** The freeze precedes every training, and every training and labelling precedes the first score.
- **G4 binding.** Predictor, policy, payload, pool-window and truth-store hashes match.
- **G5 R_9 replication.** Every R_9 fixed row equals the Gate-B and #101 records.
- **G6 schedule integrity.** Every member's schedule is its frozen schedule.
- **G7 nonfinite.** The J-flat grid nonfinite share is ≤ 0.10.
