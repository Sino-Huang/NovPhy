# Issue #115: predicate-feedback robustness (corrupted, removed, oracle, learned and label-free description levels)

Parent: #108 (pass branch). The shared rules of #85 apply, and the #111 scoring contract is adopted. The #99, #100, #101,
#104, #109, #112, #113, #114 and Gate-B records are read-only inputs. Nothing here edits the ICLR 2026 submission.

| | |
| --- | --- |
| Runner | `scripts/run_predicate_feedback_robustness.py` |
| Module | `world_model/training/predicate_feedback.py`. It holds the feedback interventions, the carrier step with a given adapter input, the predictor variants, the label-use spec of the training loss, the corruptions and the readout counts |
| Tests | `tests/test_issue_115_predicate_feedback.py` (16 tests): corruption calibration, oracle alignment, the bottleneck, the loss spec, temporal history |
| Artifacts | `.local-artifacts/issue-115-predicate-feedback-v1/`. `findings.md` has every table. Also `compute.json`, `summary.json`, `dose-response.svg`, `records-manifest.json`, `handoff_106.json`, `plan.json`, `smoke.json`, `logs/`, `ledger/`, `calibration/`, training reports |
| Freeze | Plan frozen 2026-10-16T17:34:11Z, commit c7289d67. The freeze followed a passed smoke and came before any training update, evaluation byte or score |
| Validation | `python -u -m scripts.run_predicate_feedback_robustness --validate` (exit 0); `pytest tests/test_issue_115_predicate_feedback.py` (16 passed) |
| #106 handoff | `.local-artifacts/issue-115-predicate-feedback-v1/handoff_106.json`: the micro label-free bottleneck `mi-free`, with the qualification under "Dispositions" |

Engine seconds: 0. Intervals are level-clustered percentile bootstraps (10 000 draws, PCG64 7201). They are DESCRIPTIVE.
The margin is δ = 0.02.
- Grid is primary.
- Offset, angle, power, the rollout error and the readouts are DESCRIPTIVE.
- All eight guards pass.

## Dispositions (pre-declared, grid)

Claim: "the description axis keeps its value without engine-labelled predicates". It compares the label-free bottleneck
(item 3 iii) with the engine-labelled reference.

| description | token | C1 = (J−T)_free − (J−T)_ref | C2 = F-mean_free − F-mean_ref | J_free − J_ref | reference J − T |
| --- | --- | --- | --- | --- | --- |
| micro (`mi-free` vs Gate-B) | **`supported`** | −0.003 [−0.015, +0.008] | +0.020 [+0.007, +0.033] | +0.008 [−0.000, +0.016] | +0.003 [−0.007, +0.014] (not supported) |
| macro (`ma-free` vs `ma-ref`) | **`readiness_or_precision_insufficient`** | +0.052 [+0.041, +0.063] | −0.004 [−0.025, +0.017] | +0.017 [+0.000, +0.033] | −0.008 [−0.012, −0.003] (not supported) |

**Rule (frozen):**
- `supported` iff both lower bounds are > −δ (non-inferiority);
- `not_supported_by_this_experiment` iff either upper bound is < −δ;
- otherwise `readiness_or_precision_insufficient`;
- a failed guard turns every token into `readiness_or_precision_insufficient`.

**Micro: `supported`, with two qualifications.**
1. **There is little value to keep.** Under the δ rule, the engine-labelled reference has no description-axis value
   over its horizon-only restriction on grid: J − T is +0.003 [−0.007, +0.014]. The token therefore establishes that the
   label-free bottleneck is non-inferior at δ. It does not show that a demonstrated ≥ δ value survives.
2. **Grid only.** The rule is fixed on grid. On the DESCRIPTIVE inventories, C1 stays within ±0.01, but the fixed micro
   requests of the label-free predictor rank far worse than the reference's:
   - C2 offset −0.145 [−0.160, −0.129];
   - C2 angle −0.106 [−0.119, −0.093];
   - C2 power −0.000.

   The handoff to #106 carries this qualification.

**Macro: `readiness_or_precision_insufficient`.** C1 is non-inferior (+0.052; the label-free J − T is higher), but C2's
lower bound −0.025 does not exclude a loss of δ. The macro reference's own head reads nothing: delivered and parsed macro
F1 are 0.000, as on Gate-B (#100, #113). So every macro comparison here concerns a head that carries no readable
predicate.

## Design

### Predictors

All predictors are `CNNHybridPredictor`s on E99 carriers with 1 837 690 parameters. The Δ-indexed macro variants have
1 838 722 (+0.06 %).
- **Training.** Each predictor was trained under the Gate-B recipe on the #112 v3 retrain payload (#109 fit split,
  hash-checked): #112 v3 LC, 9000 updates, fold 'all', one member per group, seeds 20260908/09/10.
- **Training path check.** Under the default label spec, the path equals the retrain's `train_group` bit for bit. In
  the smoke, after 12 full-cost updates, the parameter delta was 0 and the logged local losses were identical.
- **Usability.** The retrain rule applies: complete, at most 45 skipped updates, and no #117 end-of-training excursion.
  An unusable member is a typed terminal failure: it is not replaced, and its cells drop.

| predictor | description | labels of that description | class |
| --- | --- | --- | --- |
| `gateb` | micro reference | engine (#99): the Gate-B handoff checkpoints | `CNNHybridPredictor` |
| `mi-flipXX` | micro | engine labels with symmetric flips at p = XX/100 on labelled entries (contact per unordered pair), one draw per (arm, seed) | same |
| `mi-low10`, `mi-low25` | micro | engine labels on 10 % (48) or 25 % (120) of fit levels only; elsewhere neither the label term nor the availability term | same |
| `mi-learned` | micro | readings of a micro perception head trained on the 25 % levels and frozen | same |
| `mi-free` | micro | none: binary straight-through codes over slot pairs, no micro symbolic loss | `BinaryMicroPredictor` |
| `ma-ref` | macro reference | the #113 frozen observable predicates of the request's Δ | `DeltaMacroPredictor` |
| `ma-flipXX`, `ma-low10/25`, `ma-learned` | macro | as the micro arms, on the #113 predicates. The perception head is the #113 temporal-head recipe (K4-mix50) on the 25 % levels | same |
| `ma-free` | macro | none: binary codes, no macro symbolic loss | `BinaryDeltaMacroPredictor` |

- **The other description keeps the reference labels.** In the micro arms, the macro head keeps the Gate-B (old) macro
  labels. In the macro arms, the micro head keeps the engine labels.
- **Unchanged.** Adapter dimensions and loss weights (0.1, pos_weight 5) are the same in every arm.
- **Low-label subsets.** They are nested and family-stratified, drawn with PCG64(11501).
- **Perception-head thresholds.** These are the in-sample best F1 on the labelled subset over the #113 threshold grid.
  The heads' readings replace the engine labels at every payload position. For the micro head, the mask is "both slots
  present in the carrier" combined with the head's channel availability. For the macro head, the availability is
  positional.

**J for every predictor.** Every trained predictor gets its own ICLR request policy J: the #101 pe DP at λ = 1e-4 on
the #109 policy pool, with two DAgger rounds and distillation. The smoke reproduced #101's round-0 labels exactly. The
Gate-B predictor uses the #101 J itself.

### Arms and controls

- **J:** the predictor's own J.
- **T:** the horizon-only restriction of J (T-continuous, #114). It makes no description request and reads no feedback.
- **D:** the description-only restriction of J at J's modal Δ on the policy split (#96 rule).
- **Fixed requests:** F-{1,5,15}-α for the predictor's description α.
- **Controls, in every table:**
  - the no-model action prior (Gate-B rule);
  - the wrong-anchor control: J from the 480 fit-split anchors over the 16 grid candidates, default feedback, every
    predictor.

### Item 1: inference-time interventions

The interventions run on the frozen micro reference (`gateb`, micro requests) and the frozen macro reference (`ma-ref`,
macro requests). Requests of the other description keep the model's own feedback.

| intervention | adapter input at a description request |
| --- | --- |
| default | the model's heads (`symbolic_features` of the current carrier) |
| oracle | engine labels of the executed candidate's branch at the matching frame (clamped to its last frame), through the same adapter. Micro: #99 relations and relation_mask, channel availability = any labelled pair. Macro: the #113 predicate of the request's Δ, with positional availability |
| flipXX | the oracle with symmetric flips at p = XX/100, drawn per call |
| gaussXX | the model's logits + σ N(0, 1), contact noise symmetric. σ is calibrated on fit-split cells to XX % of the default delivered F1 |
| removed | zero adapter input |
| shuffled | the model's feedback of the next candidate (cyclic ordinal order) of the same anchor, at the same iteration |
| temporal (macro) | the #113 temporal head K4-mix50 (seed-paired) on the row's own stride-Δ rollout history, binarised at #113's selected predicted-source thresholds |

**Delivered-feedback F1** is the dose-response x-axis. It applies the #100 readout to the adapter input actually fed at
every description request of the arm's rollouts:
- a gated value ≥ 0.5 counts as positive;
- micro contact is scored on the upper triangle and supports as directed pairs;
- macro is scored per predicate on available frames.

Each reading is compared with the executed branch's engine labels at the matching frame. The counts are pooled, and the
reported F1 is the mean over the two channels.

**Gaussian calibration.** This was a fit-split stage, run after training and before any score:
- the fixed α requests from the first 24 fit grid cells;
- bisection in log10 σ on [1e-2, 1e2] with 12 halvings;
- micro fit labels from the retrain shot cache, macro fit labels from the #113 fit labels.

| | default F1 (fit) | σ at 0.8 / 0.6 / 0.4 / 0.2 |
| --- | --- | --- |
| Gate-B seeds 08 / 09 / 10 | 0.790 / 0.625 / 0.887 | 3.40 / 4.95 / 7.32 / 16.8; 4.10 / 5.80 / 9.04 / 34.0; 3.07 / 4.60 / 6.75 / 14.8 (every target met within 0.001) |
| `ma-ref` | 0.000 | 1e-2 for all (degenerate: the target is 0, so the Gaussian arms equal default within 0.001) |

### Evaluation

- **Ranking.** Cells are the Gate-B mixed cells (grid 311 levels, offset 76, angle 78, power 47, × 3 seeds). Rollouts
  start from the Gate-B E99 anchors and use the tie-free cost and `m99.view`. Cells are (level, seed), and predictor seed
  s pairs with reference seed s.
- **Rollout error at t = 225.** The #100 `engine_position_mse` of every mixed grid candidate's endpoint, against its own
  branch at min(225, last frame).
- **Readout F1 of the heads.** The #100 audit form, on the 4 976 grid branches:
  - parsed: the head on the E99 parse of each frame 0..min(225, last);
  - predicted: the head after one Δ-step prediction from the parsed frame f − Δ;
  - macro: read on the #113 predicate of the request's Δ column.

  The parses are #113's deterministic E99 parses. The engine micro labels of all 7 766 mixed-cell branches come from
  `s99.shot_labels` and were built after the freeze.
- **Label-free arms.** The readout compares binary codes with engine predicates by channel identity. It is an
  interpretation aid only.

## Results

### Item 1: micro dose-response (Gate-B, grid)

| feedback | delivered F1 | J | D | F-1 | F-5 | F-15 | rollout J | rollout F-1 / F-5 / F-15 | J − default | F mean − default |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | 1.000 | 0.625 | 0.588 | 0.287 | 0.370 | 0.463 | 0.0318 | 0.0318 / 0.0386 / 0.0422 | −0.081 [−0.091, −0.071] | −0.345 [−0.364, −0.326] |
| default | 0.853 [0.843, 0.863] | 0.706 | 0.718 | 0.650 | 0.745 | 0.760 | 0.0317 | 0.0303 / 0.0375 / 0.0419 | — | — |
| shuffled | 0.851 | 0.707 | 0.714 | 0.646 | 0.712 | 0.739 | 0.0318 | 0.0306 / 0.0379 / 0.0420 | +0.001 [−0.001, +0.003] | −0.019 [−0.023, −0.014] |
| gauss80 | 0.661 | 0.700 | 0.708 | 0.621 | 0.693 | 0.702 | 0.0319 | 0.0323 / 0.0394 / 0.0437 | −0.005 [−0.010, −0.001] | −0.046 [−0.053, −0.040] |
| flip05 | 0.610 | 0.636 | 0.581 | 0.321 | 0.364 | 0.445 | 0.0323 | 0.0364 / 0.0416 / 0.0458 | −0.070 [−0.080, −0.060] | −0.341 [−0.359, −0.324] |
| gauss60 | 0.499 | 0.691 | 0.698 | 0.601 | 0.654 | 0.659 | 0.0323 | 0.0362 / 0.0429 / 0.0468 | −0.014 [−0.019, −0.010] | −0.080 [−0.090, −0.070] |
| flip10 | 0.431 | 0.639 | 0.585 | 0.336 | 0.372 | 0.446 | 0.0328 | 0.0411 / 0.0454 / 0.0499 | −0.067 [−0.076, −0.057] | −0.334 [−0.350, −0.317] |
| gauss40 | 0.340 | 0.685 | 0.687 | 0.558 | 0.613 | 0.621 | 0.0330 | 0.0442 / 0.0501 / 0.0529 | −0.021 [−0.029, −0.013] | −0.121 [−0.134, −0.108] |
| flip20 | 0.256 | 0.647 | 0.588 | 0.367 | 0.398 | 0.435 | 0.0341 | 0.0517 / 0.0546 / 0.0596 | −0.059 [−0.068, −0.049] | −0.318 [−0.333, −0.304] |
| gauss20 | 0.176 | 0.676 | 0.678 | 0.546 | 0.579 | 0.579 | 0.0350 | 0.0675 / 0.0714 / 0.0696 | −0.030 [−0.038, −0.021] | −0.150 [−0.164, −0.136] |
| flip40 | 0.116 | 0.646 | 0.606 | 0.427 | 0.447 | 0.462 | 0.0381 | 0.0783 / 0.0806 / 0.0861 | −0.060 [−0.070, −0.050] | −0.273 [−0.286, −0.260] |
| removed | 0.000 | 0.708 | 0.716 | 0.680 | 0.740 | 0.757 | 0.0324 | 0.0318 / 0.0371 / 0.0416 | +0.002 [−0.000, +0.005] | +0.007 [+0.004, +0.011] |

T (no feedback) is 0.702 for every row. The prior is 0.720, and wrong-anchor J is 0.672 (right − wrong +0.034). Figure:
`dose-response.svg`. The per-inventory AUCs and the J − T and J − D of every row are in `findings.md`.

- **Ranking is not monotone in feedback F1.**
  - At the ends: removing the feedback (F1 0) changes no arm by δ (J +0.002, F mean +0.007). Engine truth (F1 1) costs
    J −0.081 and the fixed micro requests −0.345.
  - The adapter has learned to use the distribution of its own head's soft outputs, not predicate semantics. Hard 0/1
    engine labels, and the flipped oracle at every rate, are inputs the adapter never saw. With them, the fixed micro
    requests fall below the prior.
- **Noise on the model's own logits degrades ranking monotonically with the delivered F1.** F mean falls by −0.046 at F1
  0.66 and by −0.150 at F1 0.18; J falls by −0.005 to −0.030.
- **Rollout error does not track ranking.**
  - The oracle leaves the t = 225 error unchanged (J 0.0318 vs 0.0317).
  - Corrupting the feedback raises the error monotonically with the corruption: F-1-micro rises from 0.030 to 0.078 at
    flip40 and to 0.068 at gauss20.
  - Removing the feedback barely changes it: +0.0007 on J and at most +0.0015 on the fixed requests.
- **Shuffled feedback is nearly as accurate as default** (F1 0.851). Candidates of one anchor share most relations before
  the shot lands, so shuffling is a weak corruption on this split.
- **The rollouts of the fixed micro requests are sensitive to corrupted feedback, but the predictor does not need the
  feedback.** The "removed" row is the clearest reading of the micro description level's runtime dependence.

### Item 1: macro dose-response (`ma-ref`, grid)

| feedback | delivered F1 | J | D | F-1 | F-5 | F-15 | rollout J | rollout F-1 / F-5 / F-15 | J − default | F mean − default |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | 1.000 | 0.664 | 0.686 | 0.498 | 0.553 | 0.394 | 0.0546 | 0.0655 / 0.0866 / 0.0725 | −0.014 [−0.020, −0.008] | −0.212 [−0.234, −0.190] |
| flip05 | 0.119 | 0.645 | 0.682 | 0.496 | 0.520 | 0.441 | 0.0549 | 0.0666 / 0.0853 / 0.0729 | −0.033 [−0.043, −0.024] | −0.208 [−0.228, −0.188] |
| flip10 | 0.064 | 0.639 | 0.682 | 0.484 | 0.522 | 0.474 | 0.0555 | 0.0695 / 0.0859 / 0.0741 | −0.040 [−0.049, −0.030] | −0.200 [−0.218, −0.182] |
| flip20 | 0.031 | 0.640 | 0.681 | 0.506 | 0.510 | 0.510 | 0.0574 | 0.0825 / 0.0933 / 0.0837 | −0.038 [−0.050, −0.026] | −0.185 [−0.201, −0.169] |
| flip40 | 0.012 | 0.624 | 0.671 | 0.538 | 0.560 | 0.588 | 0.0680 | 0.1359 / 0.1350 / 0.1230 | −0.054 [−0.068, −0.040] | −0.131 [−0.146, −0.116] |
| temporal | 0.009 | 0.651 | 0.678 | 0.583 | 0.655 | 0.661 | 0.0631 | 0.1080 / 0.1184 / 0.1552 | −0.027 [−0.037, −0.017] | −0.060 [−0.070, −0.050] |
| default | 0.000 | 0.678 | 0.708 | 0.631 | 0.707 | 0.741 | 0.0545 | 0.0654 / 0.0828 / 0.0667 | — | — |
| gauss80–20 | 0.000 | 0.678 | 0.708 | 0.631 | 0.707 | 0.740 | 0.0545 | = default | ±0.000 | ±0.000 |
| shuffled | 0.000 | 0.680 | 0.709 | 0.631 | 0.707 | 0.740 | 0.0545 | = default | +0.002 [+0.001, +0.003] | −0.000 |
| removed | 0.000 | 0.674 | 0.710 | 0.614 | 0.670 | 0.749 | 0.0539 | 0.0650 / 0.0853 / 0.0701 | −0.004 [−0.008, +0.000] | −0.015 [−0.019, −0.012] |

T is 0.686, and wrong-anchor J is 0.626 (right − wrong +0.052).
- **The macro head reads nothing.** Its delivered F1 is 0, and its parsed and predicted readout is 0.000.
- **Settle never occurs in the rollout window.** No grid branch carries a settle frame in positions 0..224, so macro F1
  here is structure-change F1.
- **Readable macro feedback hurts.** The oracle, the flipped oracle and the #113 temporal head make J and the fixed macro
  requests worse. The rollout error rises with flips and with the temporal head.
- **Removing the macro feedback costs at most −0.015** (F mean).

### Item 2: training-label noise (grid)

| predictor | label F1 | readout parsed / predicted | delivered F1 | J | T | D | F-1 / F-5 / F-15 | rollout J | J − ref | F mean − ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gate-B (ref) | 1 | 0.965 / 0.965 | 0.853 | 0.706 | 0.702 | 0.718 | 0.650 / 0.745 / 0.760 | 0.0317 | — | — |
| mi-flip05 | 0.609 | 0.956 / 0.954 | 0.823 | 0.654 | 0.610 | 0.671 | 0.603 / 0.659 / 0.768 | 0.0512 | −0.052 [−0.061, −0.042] | −0.041 [−0.046, −0.037] |
| mi-flip10 | 0.430 | 0.930 / 0.940 | 0.844 | 0.707 | 0.650 | 0.691 | 0.636 / 0.729 / 0.768 | 0.0477 | +0.001 [−0.004, +0.006] | −0.007 [−0.016, +0.002] |
| mi-flip20 | 0.255 | 0.116 / 0.134 | 0.315 | 0.633 | 0.611 | 0.630 | 0.649 / 0.699 / 0.775 | 0.0280 | −0.072 [−0.080, −0.065] | −0.011 [−0.024, +0.002] |
| mi-flip40 | 0.116 | 0.112 / 0.112 | 0.151 | 0.661 | 0.658 | 0.662 | 0.652 / 0.681 / 0.747 | 0.0656 | −0.045 [−0.055, −0.036] | −0.025 [−0.040, −0.010] |
| ma-ref (ref) | 1 | 0.000 / 0.000 | 0.000 | 0.678 | 0.686 | 0.708 | 0.631 / 0.707 / 0.741 | 0.0545 | — | — |
| ma-flip05 | 0.175 | 0.000 / 0.000 | 0.000 | 0.744 | 0.736 | 0.788 | 0.623 / 0.739 / 0.781 | 0.0469 | +0.066 [+0.058, +0.074] | +0.021 [+0.019, +0.024] |
| ma-flip10 | 0.106 | 0.000 / 0.000 | 0.000 | 0.725 | 0.708 | 0.736 | 0.641 / 0.717 / 0.785 | 0.0494 | +0.047 [+0.038, +0.055] | +0.021 [+0.012, +0.030] |
| ma-flip20 | 0.055 | 0.019 / 0.019 | 0.008 | 0.598 | 0.605 | 0.642 | 0.599 / 0.565 / 0.629 | 0.0767 | −0.080 [−0.090, −0.071] | −0.096 [−0.103, −0.088] |
| ma-flip40 | 0.023 | 0.019 / 0.019 | 0.008 | 0.676 | 0.677 | 0.676 | 0.671 / 0.750 / 0.753 | 0.0407 | −0.002 [−0.011, +0.005] | +0.032 [+0.027, +0.036] |

J − T per predictor, grid (the δ token is in brackets):

| | Gate-B | mi-flip05 | mi-flip10 | mi-flip20 | mi-flip40 | ma-ref | ma-flip05 | ma-flip10 | ma-flip20 | ma-flip40 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| J − T | +0.003 (no) | +0.044 (sup.) | +0.057 (sup.) | +0.023 (sup.) | +0.002 (no) | −0.008 (no) | +0.008 (no) | +0.017 (insuff.) | −0.007 (no) | −0.001 (no) |
| J − D | −0.013 | −0.017 | +0.015 | +0.003 | −0.001 | −0.030 | −0.045 | −0.011 | −0.044 | −0.000 |

- **Micro readout.** The micro head denoises symmetric flips up to p = 0.1 (parsed F1 0.93–0.96) and collapses at
  p ≥ 0.2 (0.11–0.12).
- **Ranking and rollout are not monotone in the label noise.** J − ref is −0.052, +0.001, −0.072 and −0.045 for the
  four micro rates. For macro it is +0.066, +0.047, −0.080 and −0.002, while the macro head reads nothing at any rate.
- **Retrain-to-retrain variation exceeds δ.** It shows in T, which never reads predicates and still ranges from 0.605 to
  0.736 across these retrains. The noisy-label retrains with J − T ≥ δ owe it to a weaker T, not to a stronger J.
  Training-label noise has no consistent ≥ δ effect at these rates that can be separated from the variation between
  retrains.

### Item 3: low-label, learned and label-free predicates (grid)

| predictor | label F1 | readout parsed / predicted | delivered F1 | J | T | D | F-1 / F-5 / F-15 | rollout J | J − ref | F mean − ref | (J−T) − ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mi-low10 | — | 0.955 / 0.956 | 0.831 | 0.641 | 0.605 | 0.627 | 0.603 / 0.682 / 0.735 | 0.0570 | −0.064 [−0.075, −0.054] | −0.045 [−0.050, −0.040] | +0.033 |
| mi-low25 | — | 0.963 / 0.962 | 0.849 | 0.663 | 0.681 | 0.656 | 0.639 / 0.708 / 0.743 | 0.0521 | −0.043 [−0.058, −0.028] | −0.022 [−0.026, −0.018] | −0.021 |
| mi-learned | 0.982 | 0.965 / 0.965 | 0.829 | 0.690 | 0.664 | 0.693 | 0.662 / 0.661 / 0.783 | 0.0424 | −0.015 [−0.025, −0.005] | −0.016 [−0.028, −0.004] | +0.024 |
| mi-free | — | 0.000 / 0.000 | 0.000 | 0.714 | 0.714 | 0.709 | 0.671 / 0.745 / 0.798 | 0.0402 | +0.008 [−0.000, +0.016] | +0.020 [+0.007, +0.033] | −0.003 |
| ma-low10 | — | 0.000 / 0.000 | 0.000 | 0.677 | 0.622 | 0.676 | 0.648 / 0.742 / 0.769 | 0.0546 | −0.001 [−0.012, +0.009] | +0.027 [+0.022, +0.031] | +0.062 |
| ma-low25 (2 seeds) | — | 0.000 / 0.000 | 0.000 | 0.699 | 0.670 | 0.709 | 0.647 / 0.687 / 0.743 | 0.0516 | −0.007 [−0.013, −0.002] | −0.011 [−0.017, −0.005] | +0.037 |
| ma-learned | 0.088 | 0.000 / 0.000 | 0.000 | 0.728 | 0.703 | 0.750 | 0.608 / 0.720 / 0.787 | 0.0583 | +0.050 [+0.043, +0.057] | +0.012 [+0.009, +0.016] | +0.033 |
| ma-free | — | 0.004 / 0.002 | 0.005 | 0.695 | 0.651 | 0.675 | 0.623 / 0.699 / 0.745 | 0.0449 | +0.017 [+0.000, +0.033] | −0.004 [−0.025, +0.017] | +0.052 |

- **Micro heads need few labels.** With 10 % of fit levels labelled, the head reads 0.955; a perception head trained on
  25 % reads 0.982 on its own labelled rows and 0.982 on the grid branches (`findings.md`, perception heads).
- **The label-free codes carry no engine predicate by channel identity** (F1 0.000). Yet `mi-free` ranks at least as
  well as the reference on grid: J 0.714, fixed requests +0.020. Its rollout error is higher (J 0.040 vs 0.032).
- **No engine-labelled micro arm beats the label-free one on grid.** Low-label and learned-head training cost −0.015 to
  −0.064 in J.
- **The macro perception head reads poorly.** On the grid branches it reaches macro F1 0.084 (parsed); its training
  labels agree with the engine at F1 0.088.

### Controls

- **No-model action prior:** grid 0.720 [0.697, 0.744], angle 0.812. Offset and power are 0.500: the #109 fit split
  executes no offset or power candidate, so the prior is constant there (the Gate-B rule).
  - J − prior on grid is −0.015 [−0.039, +0.009] for Gate-B and −0.006 for `mi-free`, both `not_supported`.
  - No predictor's J clears the prior by δ on grid. The best is `ma-flip05`, +0.023 [−0.005, +0.051] (`insufficient`).
- **Wrong anchor (J, grid).** Right − wrong is positive for every predictor. It is `supported` for 14 of 18.
  - Exceptions: `mi-flip20` +0.014, `ma-flip20` +0.008 and `ma-flip40` +0.006 are `not_supported`, and `mi-low25` +0.017
    is `insufficient`.
  - For those predictors, J barely uses the anchor.
  - Gate-B +0.034 reproduces #114's +0.0336.

### Training

- 51 members (17 arms × 3 seeds); 0 retired; 0 skipped updates.
- One typed terminal failure: `ma-low25` seed 20260910, `final_checkpoint_excursion` (end-state local-loss ratio 19.1).
  It was not replaced and has no policy. Its 8 records are typed failures, and the `ma-low25` contrasts use 2 seeds.
- The realized flip rates of the training labels equal p to 4 decimals. Label F1 against clean labels:
  - micro: 0.609 / 0.430 / 0.255 / 0.116;
  - macro: 0.175 / 0.106 / 0.055 / 0.023.

## Guards

All eight pass:

| guard | result |
| --- | --- |
| G1 isolation | pass |
| G2 scoring policy | pass, on all 783 records and 6 calibrations |
| G3 chronology | pass: freeze 17:34:11Z ≤ heads, training 18:09–20:54Z ≤ calibration (last 20:10Z) ≤ first score 20:58Z; truth built 18:38Z |
| G4 checkpoint binding | pass |
| G5 replication | pass: 9 216 Gate-B default-feedback blocks (J, F-*-micro, T-continuous, D-Δ_D rows and schedules) equal the #101 and #114 records exactly |
| G6 intervention integrity | pass: oracle fp = fn = 0; removed delivers zeros; realized flip rates 0.04999–0.40003 |
| G7 nonfinite | pass: 0.0 |
| G8 label integrity | pass |

`--validate`:
- re-derives `compute.json`, `summary.json`, `findings.md`, `dose-response.svg`, `records-manifest.json` (sha256 of all
  783 records and 51 update logs) and `handoff_106.json`, and byte-compares them;
- re-scores on every grid cell `mi-free` J, the Gate-B oracle and flip20 F-5-micro, and the `ma-ref` temporal
  F-15-macro, first seed, with exact rows and feedback counts;
- re-scores one readout record.

It exits 0 (1 166 s).

## Compute

| phase | GPU s |
| --- | --- |
| training (51 members, 2 processes per GPU) | 72 313 |
| request policies J (50) | 15 995 |
| scoring (783 records, 2 processes per GPU) | 119 603 |
| Gaussian calibration | 5 119 |
| perception heads | 214 |
| smoke (frozen run) | 295 |
| **total** | **213 539** |

- **Engine truth:** 3 855 s wall on 40 CPU processes (0 GPU s).
- **Elapsed:** about 8 h from the freeze to the last score (17:34Z–01:15Z), then 19 min to publish.
- **Engine seconds:** 0.
- **Hardware:** 4 × RTX 5090.
- **Accounting:** GPU s equals the summed process wall, and contention between two processes per GPU is included. The
  frozen cap was 80 GPU-h, and 59.3 h were used. The projection was 10.7 h elapsed (153.8k GPU s). Scoring took 2× its
  projection because of the two-per-GPU contention.

## Disclosures

- **Hardware.** RTX 5090, not the #85 RTX 3090.
- **Known before the freeze:**
  - Gate B (unit 0.7444, prior 0.7203);
  - #101 (J grid 0.706, 79 % (1, macro));
  - #114 (J − T-continuous +0.003);
  - #100 (E99 micro edge F1 0.989, macro 0);
  - #113 (temporal head held-out macro F1 0.157 engine / 0.113 predicted / 0.015 rollout; current head 0.000).
- **Pre-freeze work:**
  - The analysis and rendering code was rehearsed on synthetic random rows over fit-split cells posing as the
    evaluation set. There was no model output and no evaluation capture.
  - Three smokes passed. The first two ran on runner revisions that differed only in the findings layout, the
    guard-function split and the plan's disclosure text. The third ran on the frozen bytes.
- **Macro reference.** The Gate-B macro head reads nothing on the #113 predicates. Macro items 1–3 therefore use
  `ma-ref`, an LC retrain whose Δ-indexed macro head is supervised by them. It also reads nothing (delivered, parsed and
  predicted macro F1 0.000), so the macro Gaussian arms are degenerate by the frozen calibration rule.
- **Execution.**
  - Each stage was launched once (PIDs in `logs/*-pids.txt`). The heads worker 2 finished in 21 s, before its PID was
    recorded (see its log).
  - Nothing was relaunched or retried. The typed failure was not replaced. No runner or module byte changed after the
    freeze.
- **Interpretation limits.**
  - The label-free readout is by channel identity.
  - "Shuffled" is a weak corruption on this split.
  - The dispositions concern one discrete bottleneck on NovPhy, not every unsupervised predicate learner.
