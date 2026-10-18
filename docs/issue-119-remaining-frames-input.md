# Issue #119: remaining-frames input saturation

Parent: #108. The binding shared rules of #85 apply, and #119 adopts the #111 scoring contract. Gate B (#108), #101, #104, #109 and #112 records are read-only inputs. Nothing here edits the ICLR 2026 submission.

| | |
| --- | --- |
| Runner | `scripts/run_remaining_frames_input.py` |
| Module | `world_model/training/remaining_input.py`: the input variants, the elapsed-channel policy, window rows with an elapsed column, distillation, label accuracy, request mix by κ. `request_policy.py` is unchanged, because it is hash-pinned by later plans. |
| Artifacts | `.local-artifacts/issue-119-remaining-frames-v1/`: `plan.json`, `smoke.json`, `selection.json`, `findings.md`, `summary.json`, `compute.json`, `records-manifest.json`, `models/*/seed-*/report.json`, `records/`, `accuracy/` |
| Freeze | The plan was frozen at commit 3d5c039a after a passed smoke and before any training or score. |
| Validation | `python -u -m scripts.run_remaining_frames_input --validate` exits 0. `pytest tests/test_issue_119_remaining_frames_input.py` passes (15 tests). |

Intervals are percentile bootstraps clustered by level (member): 10 000 draws, PCG64 7201, DESCRIPTIVE. The margin is δ = 0.02. Tokens follow the δ rule only:
- `supported` iff the mean is ≥ δ and the lower bound is > 0;
- `not_supported_by_this_experiment` iff the upper bound is < δ;
- otherwise `readiness_or_precision_insufficient`.

Engine time is zero seconds. The reading is confirmatory because Gate B passed: one score on the #104 sealed #109 evaluation split (Gate-B cells: grid 311, offset 76, angle 78, power 47; 3 seeds). Grid is primary; offset, angle and power are DESCRIPTIVE.

## Result: primary `not_supported_by_this_experiment`

Claim: "the κ cap costs decision value: an unsaturated remaining-frames input improves J − F_hindsight by ≥ δ on the primary inventory".

| Contrast (grid, 933 cells, 311 levels) | Estimate | Token |
| --- | --- | --- |
| (J(v\*) − F_hindsight) − (J − F_hindsight) = J(b, L60) − J(a, L60) | **+0.0042 [−0.0005, +0.0088]** | `not_supported_by_this_experiment` |

v\* = (b) min(κ, 225)/225, selected on the policy split before any score. Its seed-mean policy-split AUC was 0.738, against 0.736 for (c), 0.735 for (d) and 0.721 for control (a). F_hindsight is F-15-continuous (grid AUC 0.762). All seven guards pass, including the bit-for-bit replication of the #101 J by variant (a).

### Per-variant table (grid)

Label accuracy is policy-split argmax agreement with the DP labels; this is training data (see "Label accuracy" below). AUC differences are paired by level and seed.

| arm | labels | label acc. L60 / L225 / own | AUC | − F_hindsight | − J | − prior | MACs / candidate | policy MACs / call | right − wrong anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| J(a, L60) = J | pe, L60 | 47.2 / 37.4 / 48.9 % | 0.706 [0.685, 0.727] | −0.057 [−0.068, −0.046] | — | −0.015 | 184.0M | 32 128 | +0.034 [0.030, 0.037] |
| J(b, L60) | pe, L60 | 47.3 / 38.0 / 47.6 % | 0.710 [0.689, 0.730] | −0.052 [−0.062, −0.043] | +0.004 [−0.001, +0.009] | −0.010 | 164.9M | 32 128 | +0.033 [0.029, 0.037] |
| J(c, L60) | pe, L60 | 47.5 / 43.8 / 49.9 % | 0.709 [0.688, 0.731] | −0.053 [−0.063, −0.043] | +0.004 [−0.001, +0.008] | −0.011 | 227.5M | 32 128 | +0.032 [0.028, 0.036] |
| J(d, L60) | pe, L60 | 47.4 / 41.2 / 49.1 % | 0.705 [0.684, 0.726] | −0.057 [−0.067, −0.047] | −0.001 [−0.006, +0.005] | −0.015 | 193.4M | 32 256 | +0.034 [0.028, 0.039] |
| J(a, L225) | pe, L225 | 47.0 / 53.5 / 54.5 % | 0.680 [0.658, 0.702] | −0.082 [−0.094, −0.071] | −0.026 [−0.032, −0.019] | −0.040 | 379.2M | 32 128 | +0.001 [0.001, 0.002] |
| J(b, L225) | pe, L225 | 47.5 / 53.7 / 54.9 % | 0.683 [0.661, 0.705] | −0.079 [−0.091, −0.067] | −0.023 [−0.028, −0.017] | −0.037 | 376.6M | 32 128 | +0.005 [0.003, 0.006] |
| J_dec(b) | decision, λ 1e-1 | 9.1 / 6.6 / 77.7 % | 0.742 [0.724, 0.760] | −0.020 [−0.025, −0.016] | +0.036 [0.027, 0.046] | +0.022 | 22.7M | 32 128 | +0.125 [0.115, 0.135] |
| J_dec(a) (#101 J_dec) | decision, λ 1e-1 | 9.1 / 5.2 / 78.2 % | 0.746 [0.728, 0.764] | −0.016 [−0.020, −0.012] | +0.040 [0.030, 0.052] | +0.026 | 22.7M | 32 128 | +0.128 [0.117, 0.139] (#101) |

### Contrasts per inventory

| Contrast | grid | offset | angle | power |
| --- | --- | --- | --- | --- |
| J(b, L60) − J (primary) | **+0.004 [−0.001, +0.009]** | +0.033 [+0.014, +0.052] supported | +0.032 [+0.019, +0.045] supported | −0.030 [−0.043, −0.018] |
| J(c, L60) − J | +0.004 [−0.001, +0.008] | +0.018 [−0.008, +0.044] insufficient | +0.028 [+0.008, +0.047] supported | −0.034 [−0.048, −0.021] |
| J(d, L60) − J | −0.001 [−0.006, +0.005] | +0.006 [−0.019, +0.031] insufficient | +0.013 [−0.011, +0.037] insufficient | −0.017 [−0.029, −0.007] |
| J(a, L225) − J (labels only) | −0.026 [−0.032, −0.019] | −0.011 [−0.031, +0.009] | −0.007 [−0.023, +0.010] | −0.014 [−0.025, −0.005] |
| J(b, L225) − J(b, L60) | −0.027 [−0.031, −0.023] | −0.064 [−0.090, −0.037] | −0.043 [−0.060, −0.027] | +0.025 [+0.013, +0.037] supported |
| J(b, L225) − J (input and labels matched) | −0.023 [−0.028, −0.017] | −0.031 [−0.046, −0.015] | −0.011 [−0.028, +0.007] | −0.005 [−0.010, 0.000] |
| J(b, L225) − J(a, L225) | +0.003 [+0.001, +0.005] | −0.020 [−0.035, −0.004] | −0.004 [−0.021, +0.012] | +0.009 [0.000, +0.019] |
| interaction (b225 − a225) − (b60 − a60) | −0.001 [−0.007, +0.005] | −0.053 [−0.080, −0.026] | −0.036 [−0.052, −0.021] | +0.039 [+0.020, +0.061] supported |
| J_dec(b) − J_dec(a) | −0.004 [−0.008, −0.001] | +0.017 [+0.004, +0.029] insufficient | +0.008 [−0.005, +0.021] insufficient | −0.000 [−0.011, +0.013] |
| J_dec(b) − J(b, L60) | +0.032 [+0.023, +0.041] supported | −0.078 [−0.123, −0.033] | −0.102 [−0.142, −0.061] | +0.017 [−0.000, +0.034] insufficient |
| J_dec(b) − F_hindsight | −0.020 [−0.025, −0.016] | −0.135 [−0.161, −0.111] | −0.149 [−0.173, −0.126] | −0.020 [−0.030, −0.009] |

Every unmarked cell is `not_supported_by_this_experiment`. "insufficient" means `readiness_or_precision_insufficient`. Offset, angle and power are DESCRIPTIVE. F_hindsight is F-5-micro on offset, angle and power. `findings.md` has every contrast, the AUC of every arm on every inventory, and the − F_hindsight and − prior rows.

### Request mix against the true remaining frames κ (grid)

Share of policy steps, pooled over cells, candidates and seeds. The full five-bin tables are in `findings.md`.

| arm | κ 166–225: Δ=15 / continuous / macro | κ 1–30: Δ=15 / continuous / macro | late − early (κ 1–60 minus κ 166–225), percentage points: Δ=1 / Δ=15 / macro |
| --- | --- | --- | --- |
| J(a, L60) = J | 9.2 / 13.9 / 71.2 % | 1.5 / 10.2 / 85.3 % | +5.3 / −6.5 / +14.2 |
| J(b, L60) | 16.6 / 48.0 / 29.2 % | 1.5 / 16.5 / 78.4 % | +12.8 / −14.0 / +50.1 |
| J(c, L60) | 4.7 / 14.2 / 76.5 % | 1.5 / 19.1 / 75.9 % | +1.5 / −2.4 / +1.7 |
| J(d, L60) | 7.2 / 11.0 / 75.3 % | 2.2 / 9.9 / 85.1 % | +3.7 / −3.9 / +10.9 |
| J(a, L225) | 0.2 / 11.1 / 34.5 % (Δ = 1 on 99.8 %) | 0.0 / 0.0 / 41.3 % | +0.2 / −0.2 / +4.4 |
| J(b, L225) | 0.2 / 12.6 / 34.3 % | 0.0 / 0.0 / 46.1 % | +0.2 / −0.2 / +10.6 |
| J_dec(b) | 99.9 / 55.7 / 44.3 % | 99.7 / 42.4 / 24.8 % | +0.0 / −0.1 / −16.8 |
| J_dec(a) | 100.0 / 53.9 / 46.1 % | 99.9 / 43.5 / 22.6 % | +0.0 / −0.0 / −18.7 |

### Label accuracy by κ (L225 round-0 labels, seed means)

| arm | κ 166–225 | 106–165 | 61–105 | 31–60 | 1–30 |
| --- | --- | --- | --- | --- | --- |
| J(a, L60) = J | 33.5 % | 34.4 % | 34.6 % | 41.4 % | 51.1 % |
| J(b, L60) | 25.9 % | 37.6 % | 41.8 % | 43.3 % | 51.2 % |
| J(c, L60) | 41.6 % | 42.2 % | 43.3 % | 44.3 % | 51.4 % |
| J(d, L60) | 37.9 % | 40.5 % | 41.7 % | 41.4 % | 48.4 % |
| J(a, L225) | 58.0 % | 54.0 % | 52.0 % | 48.0 % | 51.4 % |
| J(b, L225) | 58.2 % | 54.1 % | 52.3 % | 48.4 % | 51.7 % |

### Reading

- **An unsaturated time input does not close J − F_hindsight on grid.**
  - J(b) − J is +0.004 on grid. The upper bound, +0.009, is below δ, so the primary token is `not_supported_by_this_experiment`. (c) gives +0.004 and (d) gives −0.001; neither is supported.
  - J − F_hindsight moves from −0.057 to −0.052.
  - On the descriptive inventories, (b) is above J on offset (+0.033) and angle (+0.032), and (c) is above J on angle (+0.028); all three are `supported` by the δ rule. On power, both are below J (−0.030 and −0.034). These are descriptive inventories, so they do not change the disposition.
- **With the unsaturated input, the policy uses κ.**
  - With (b), the early part of the rollout (κ 166–225) is longer and finer: Δ=15 on 16.6 % of steps against 9.2 % for J, continuous on 48 % against 14 %, macro on 29 % against 71 %.
  - Late in the rollout the policy moves to macro: the late − early shift is +50 points, against +14 for J. It also moves to Δ = 1: Δ=15 drops by 14.0 points.
  - So, with κ visible, the late shift becomes sharper toward coarse α but toward short Δ, not long. Δ = 15 is masked once κ < 15, which accounts for part of the drop. This changes the schedule without changing grid ranking beyond δ.
- **DP labels that plan to the 225-frame endpoint make the ICLR prediction-error labels worse, not better.**
  - J(a, L225) − J is −0.026, J(b, L225) − J(b, L60) is −0.027, and J(b, L225) − J is −0.023. None of these is supported, and all have upper bounds below 0 on grid.
  - L225-trained policies request Δ = 1 on 99.8–100 % of steps. Their MACs double (377–379M per candidate against 184M), and their schedules barely depend on the anchor: right − wrong is +0.001 / +0.005, with rank stability 0.998.
  - Over a 225-frame window, the prediction-error DP (Δ × one-step MSE + λ·MAC, λ = 1e-4) prefers Δ = 1 almost everywhere. The longer label horizon adds the same short-Δ preference that already makes J trail the fixed Δ = 15 requests.
  - The input × window interaction is −0.001 on grid. Input range and label range matching adds nothing beyond the label effect.
- **Under the decision-valued labels, the unsaturated input adds nothing.**
  - J_dec(b) − J_dec(a) is −0.004 [−0.008, −0.001] on grid, and J_dec(b) − F_hindsight is −0.020. J_dec(b) stays a Δ = 15 policy that varies α (99.9 % Δ = 15).
  - J_dec(b) − J(b, L60) is +0.032, `supported` on grid. This repeats the #101 finding that the decision labels, not the time input, are what move J toward F_hindsight.
- **Label accuracy.**
  - The unsaturated inputs fit the endpoint-aware L225 labels only somewhat better than the capped input: 38.0–43.8 % against 37.4 %.
  - (c) and (d) gain most at κ > 60. (b) trained on L60 is worse than (a) at κ 166–225 (25.9 % against 33.5 %), because its input there is outside the training range (κ ≤ 60, input ≤ 0.27).
  - L225-trained policies reach 53.5–53.7 % on L225 labels. All prediction-error policies are near 47 % on L60 labels.
- **For #101.** The κ cap is not where J − F_hindsight comes from. #101's decision-valued labels need not be re-measured against an unsaturated input: J_dec(b) ≈ J_dec(a).

## Protocol

### Inputs, recipe and pool

- **Pool.** The #101 pool, read-only and hash-bound in the plan: the #109 policy split, 120 levels, 3480 shots. Unit = policy level.
- **Predictors.** The LC hybrid Gate-B checkpoints, seeds 20260908/09/10. Policy seed s uses predictor seed s.
- **Recipe.** The #101 J recipe:
  - prediction-error DP with λ = 1e-4;
  - 2 DAgger rounds × 1800 steps, batch 128, AdamW 1e-3;
  - CPU sampler `Generator(seed + 7700 + round)`;
  - init `torch.manual_seed(seed + 7700)`;
  - request set PAIRS.

### Input variants (frozen; everything else identical)

| variant | time input | policy |
| --- | --- | --- |
| a | min(κ, 60)/60 (ICLR, control) | `RequestPolicy(PAIRS, 128, "clamp60")` = 242 → 128 → 9 |
| b | min(κ, 225)/225 | `request_policy` `over225` (κ/225; κ ≤ 225 on every path) |
| c | log1p(κ)/log1p(225) | `request_policy` `log1p` |
| d | min(κ, 60)/60 and elapsed min(k, 225)/225 | `remaining_input.ElapsedRequestPolicy`: 243 → 128 → 9 |

- **Variant (d)'s elapsed input.** In training windows, k is the number of frames since the window start, which is shot frame 0; the DAgger rollout passes it explicitly. In scoring and continuation rollouts from the anchor, k = 225 − κ.
- **Variant (d)'s MAC term.** It counts the 243-input policy, which has 128 more MACs per call.

### DP label windows (separate pre-declared 2 × 2)

- **L60.** The #101 first 60-frame window of every pool shot.
- **L225.** The start-0 window of every pool shot reaching the 225-frame endpoint: `rp.first_window(..., 225)`, length min(225, shot end), with terminal absorption.
  - 3326 of the 3480 shots reach frame 225. The mean length is 222.9 and the minimum is 140.
  - That gives 775 520 round-0 label rows per group, against 208 800 for L60.
  - The #112 long-horizon supervision windows are fit-split predictor-training windows. The fit split stays predictor-only, so the policy-split counterpart is this window on the same E99 carrier stream, as in #116 Work 3.
- **Factor.** Input {a, b} × labels {L60, L225}. The cells (a, L60) and (b, L60) are shared with the input arms.

### v\* and the decision-label arm

- **Selection.** v\* ∈ {b, c, d} is the L60 variant with the largest seed-mean policy-split pooled AUC. This is the #101 selection metric and uses training data only. Ties go in the order b, c, d.
  - It was selected after every phase-1 group was terminal and before any score.
  - Selected: b (0.7383, against 0.7361 for c and 0.7349 for d).
- **J_dec(v\*).** One pre-declared arm, not a search: the #101 decision-valued labels.
  - λ = 1e-1, the #101 selected J_dec λ.
  - Continuation: F-15-macro in round 0, then the round-0 policy.
  - Softplus pairwise ranking loss, T = 0.1, on the #101 mixed policy cells.
  - Same recipe. The decision rows carry elapsed = rollout position.
  - Reference: the #101 J_dec, which is variant (a) under the same labels.

### Scoring and statistics

- **New records.** Six prediction-error policies and J_dec(b), × 3 seeds. They were scored once, under `deterministic_scoring()`, in processes that did not train. The scoring path is `request_policy.rollout_endpoints`, joint argmax.
- **Wrong-anchor control.** For every new policy, on 480 fit anchors × grid candidates.
- **Reused #101 records.** The nine fixed requests, the J replication target and J_dec(a). Each is checked against the #101 records manifest.
- **Cost and AUC.** Tie-free cost (`EndpointCosts.row`) and `m99.view` cell AUC.
- **F_hindsight.** The in-inventory best fixed request by member-mean AUC. It is the outcome-selected contrast target, so every − F_hindsight interval is conditional on that selection.
- **No-model action prior.** The Gate-B rule.
- **Label accuracy.** Argmax agreement on the policy split, which is training data.
  - Two reference sets: the round-0 (observed-context) labels of the L60 and L225 windows, from the variant-(a) groups of the same seed.
  - The policy's own aggregated training rows: both rounds; in-sample.
  - Per κ bin of the L225 rows. The policy-split windows are training data for every policy, so these are fit measures, not held-out accuracies.
- **Request mix.** For each scored policy step, the true κ = 225 − position, binned 166–225 / 106–165 / 61–105 / 31–60 / 1–30, with Δ and α shares. Late − early is κ 1–60 minus κ 166–225.
- **Guards** (all pass):
  - G1 isolation;
  - G2 scoring policy;
  - G3 chronology: freeze 2026-10-17T23:50:44Z ≤ phase-1 training ≤ selection ≤ J_dec training; all training ≤ first score;
  - G4 checkpoint hashes;
  - G5 replication of the #101 J;
  - G6 reused #101 records match the #101 manifest;
  - G7 selection re-derived.

  A failed guard turns every token into `readiness_or_precision_insufficient`.

### Replication guard (G5)

Variant (a) on L60 reproduces the #101 J (pe, λ = 1e-4) bit for bit, for all three seeds:
- both label rounds (`z`, `action`, `remaining`, `labels` of every unit);
- both round checkpoints;
- the final policy state dict;
- every one of the 1536 scored blocks (4 inventories × 3 seeds), including rows, schedules, transition and policy calls, and MACs.

The pre-freeze smoke had already shown this for seed 20260908, through the same training path: labels, checkpoints, per-step losses, policy and policy-split evaluation.

### Smoke (pre-freeze)

The smoke read training data and #101 artefacts only, and no evaluation-split model output. It checked:
- the replication above;
- the feature values of a–d, including that (b) equals the min form;
- that (a)'s initialization equals `request_policy.new_policy`;
- (d)'s default elapsed;
- that the elapsed input reaches the policy as 225 − κ in the scoring rollout and as the window frame in the DAgger rollout;
- that a–c rows equal `request_policy.window_rows`;
- label timing on four policy levels for both windows;
- pool and Gate-B bindings;
- the #101 record manifest.

The projection was 0.75 h on 4 GPUs.

## Module API (`world_model/training/remaining_input.py`)

| Item | What it is |
| --- | --- |
| `VARIANTS` | `a` / `b` / `c` / `d` → `remaining_feature` (a `request_policy.REMAINING_FEATURES` key) and whether there is an elapsed channel |
| `ElapsedRequestPolicy(pairs, width=128, horizon=225)` | `[z \| a \| min(κ,60)/60 \| min(k,225)/225]` → 128 → \|R\|; Δ > κ masked; `elapsed=None` means k = horizon − κ |
| `new_policy(variant, pairs, seed, *, width, device)` / `build_policy(variant, pairs, width)` | Variants a–c are `request_policy.new_policy` / `RequestPolicy` exactly |
| `call(policy, z, action, remaining, elapsed)` | Logits; elapsed reaches only an elapsed-channel policy |
| `window_rows(model, policy, windows, *, aggregated, compute_weight, device)` | Variants a–c: `request_policy.window_rows` plus an `elapsed` column. Variant d: the same DP, with the DAgger rollout fed the elapsed frames (`dagger_contexts`) |
| `aggregate_rounds(history)` / `distil_round(...)` | DAgger aggregation keeping `elapsed`. Variants a–c: `request_policy.distil_round`. Variant d: the same loop reading `elapsed` |
| `decision_elapsed(rows)` | Adds elapsed = position to `request_policy.decision_rows` output |
| `label_accuracy(policy, rows)` / `kappa_mix(sequences, pairs)` / `KAPPA_BINS` | The diagnostics above |

## Decisions under delegation (posted on #119)

1. **Reading.** Confirmatory on the sealed split, because Gate B passed. The exploratory leave-one-lineage-out mode is not run.
2. **Pool, predictor and recipe.** The #101 pool, the Gate-B LC hybrid and the #101 J recipe.
3. **Variants.** As in the table above. They live in a sibling module, because `request_policy.py` is pinned.
4. **Label-window factor.** A separate 2 × 2, {a, b} × {L60, L225}. (b) is the unsaturated input in the 2 × 2 because it is the linear one. L225 uses the policy-split start-0 windows to frame 225.
5. **v\* and J_dec.** v\* is selected on the policy split, and J_dec(v\*) is one pre-declared arm.
6. **Primary.** J(v\*, L60) − J(a, L60) on grid, which equals the change in J − F_hindsight, under the δ rule.
7. **Replication guard and reused records.** As in G5 and G6.
8. **Controls, label accuracy and request mix.** As defined above.
9. **Compute cap.** 12 GPU-h. A training error is a typed terminal failure, never retried. No typed failure occurred.

## Compute and disclosures

- **Observed GPU time.** 7889 s in total (2.19 GPU-h). Process wall time equals GPU time.

  | Phase | GPU s |
  | --- | --- |
  | Smoke | 216 |
  | Assemble | 0 (6 s CPU) |
  | Training, 18 prediction-error groups on 4 GPUs | 4356 |
  | Training, 3 J_dec groups | 122 |
  | Label accuracy | 84 |
  | Scoring, 42 records on 4 GPUs | 3111 |

  Elapsed time was about 21 min for training, 1 min for J_dec, 1 min for accuracy and 14 min for scoring. Engine time is 0 s.
- **Hardware.** RTX 5090 instead of the #85 RTX 3090.
- **Known before the freeze.** Gate B; the #101 results (J 0.706, J_dec 0.746, F-15-continuous 0.762); #114; and every #96, #100 and #112 result. No policy with input b, c or d had been scored before the freeze.
- **The (a, L225) cell was largely known before the freeze.** #116's R_9 J-flat is the #101 prediction-error recipe on the same start-0 225-frame policy windows, with the Gate-B joint predictor and the clamp60 input. It scored grid 0.678 [0.656, 0.700] at 374.9M MACs, with median Δ = 1. The frozen plan's disclosure names #116's 225-frame DP but not this policy result. J(a, L225) here (0.680, 379.2M) is a near-replication of it. #116 labelled with a batched DP (99.76 % label agreement with one-window labelling) and trained on its own path, so the two are not bit-identical. New information in the label-window factor is therefore mainly the (b, L225) cell, the interaction and the per-κ diagnostics.
- **Selection caveat.** The policy-split AUCs used for v\* are in-sample, because the labels used those levels. The spread between b, c and d was 0.003.
- **Label accuracy is in-sample.** The policy-split windows are training data for every policy. The κ-binned accuracy compares policies on the same rows; it is not a held-out estimate.
- **Retained artefacts.** The policy checkpoints and the label files (14 GB) stay local in `models/` and are bound by sha256 in the training reports. The 42 raw records (21 scored and 21 wrong-anchor) and the 24 label-accuracy records stay local in `records/` and `accuracy/`, and the committed `records-manifest.json` binds them by size and sha256. `--validate`:
  - re-derives `compute.json`, `summary.json`, `findings.md` and the manifest, and byte-compares them;
  - re-derives the selection (G7);
  - re-scores J(b, L60) seed 20260908 on all 311 grid cells (rows and schedules), and recomputes its label-accuracy record exactly.
- **Post-publication runner edit (phase-B/C review).** Commit `6b08b9af` changed `scripts/run_remaining_frames_input.py` for validation hardening only. `windows-225.pt` is assembled from the #101 pool shots, which the plan does not pin (as in #101), and embeds the sha256 of each of its 3 480 shots. Nothing re-verified those hashes. `compute_tables` now recomputes every shot's sha256 and the shot order before any table, and refuses the publication on any difference; only the (a, b) × L225 cells depend on these files. All 3 480 hashes match, so no evidence, number or token changed: `--validate` regenerates every published file byte for byte and exits 0. A test checks the retained file and a tampered copy. The runner is listed in `scripts/post_freeze_amendments.py`.
