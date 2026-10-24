## Decision P-other (2026-10-24): B1 selection configurations, native schedule, native head, budget projection

**Trigger.** The #129 runner (`scripts/run_issue_129_slotformer.py`) is about to launch the selection runs. This
fixes, outcome-blind and before any B1 training: the ≤ 3 configurations, the native schedule, the matched
secondary, the native success head with its **U + head** control, and the budget projection. Priority
(P-other): validity, then fairness to the baseline (native budget, the equal tuning allowance), then cost.

**Native schedule (every configuration).** SlotFormer's PHYRE config fixes the schedule in epochs:
- `max_epochs = 50`, `train_batch_size = 64`, nerv loader with `drop_last=True`;
- the step count (`~300k`) is a comment that holds only for its own data size, about 384 k videos per epoch.

Scaled to our 13 919 fit sequences (one per shot, from the launch frame, as PHYRE), 50 epochs is
⌊13 919 / 64⌋ = 217 steps × 50 = **10 850 steps × 64**.
- Optimizer and scheduler as in the adapter decision: 5 % warm-up, cosine to lr/100.
- fp16 autocast with the nerv `GradScaler` step, as the official `--fp16` training.
- Epoch *e* is a fresh permutation of the fit shots drawn from `eb.batch_rng(seed, e)`. Batch *b* is slice
  *b mod 217* of epoch ⌊b / 217⌋'s permutation. This is deterministic and resumable at any step.
- A nonfinite loss is a typed terminal failure.

**Selection configurations** (seed 20260908, each at its full budget; the selected run is reused as main seed 20260908):

| name | stride (rollout steps) | lr | epochs (steps) |
| --- | --- | --- | --- |
| `C1-s15` | 15 (15) | 2e-4 (official) | 50 (10 850) |
| `C2-s5` | 5 (45) | 2e-4 | 50 (10 850) |
| `C3-s15-long` | 15 (15) | 2e-4 | 250 (54 250) |

Why the third axis is schedule length, not LR:
- The fit split is about 28 × smaller than SlotFormer's PHYRE training set, so the epoch-scaled native schedule
  gives the model about 3.6 % of the official run's updates.
- The most likely reviewer objection to a re-trained baseline is under-training. A 5 × longer schedule at the
  official LR answers it directly, at an affordable cost.
- The official LR and warm-up are the authors' tuned values, so an LR alternative would test a less likely failure.
- The full official step count (~300 k steps, 1 380 epochs) was not chosen: projected at roughly 10–25 GPU-h per
  seed, it would approach the 20 GPU-h cap for an unmeasured gain.

The rule is #127's: the largest policy-split pooled grid AUC of the primary score (tie-free cost of the predicted
frame-225 carrier), with ties at 4 d.p. going to fewer parameters (all equal here), then declaration order.

**Matched secondary (B1-m).**
- Model: 1 793 445 parameters (d_model 136, 8 layers, 8 heads, ffn 544).
- Schedule: U's B1 recipe, 9 000 steps × 64, with 5 % warm-up and cosine.
- Every other hyperparameter (stride, lr) is the selected configuration's.
- Seeds 20260908, 20260909, 20260910.

**Native success head (secondary, exploratory) = SlotFormer's PHYRE readout.**
- The official `PHYREReadout` (`slotformer/phyre_planning/models/readout.py`) is vendored verbatim, with its nerv
  `build_pos_enc`. It is a CLS token plus a pre-norm Transformer over the slots at two selected timesteps
  (`sel_slots`), sine temporal P.E., an MLP to a logit, and BCE.
- Official training recipe: Adam, lr 1e-3, 10 % warm-up, cosine to lr/100, 50 epochs, batch 256. On 13 919 fit
  shots that is ⌊13 919 / 256⌋ = 54 steps × 50 = 2 700 steps.
- Inputs, matching the official recipe: it trains on the dynamics model's **rollout** slots of the training
  actions, labelled by the engine verdict.
- The official `sel_slots = [0, 3]` pairs the initial frame with an intermediate rolled frame (3 of 10 at 1 fps).
  Per #127 the head reads the **predicted end state**: `sel_slots` = [anchor, predicted frame 225].
  - Both are packed as `pack_slots(carrier, launch5)`: the 18 carrier slots plus the launch token.
  - The construction from carriers is identical for B1 and for U.
- Width: the official `d_model = slot_size` (128) cannot hold 8 heads at our slot width 13. The head keeps the
  official absolute width: d_model 128, 4 layers, 8 heads, ffn 512, with `in_proj` 13 → 128.
- Data: all 13 919 fit shots, with unweighted BCE as the official loss. 838 of them (6.0 %) are positive.
  PHYRE's `pos_ratio = 0.2` is a subsampling rule over the 100 k action cache; applying it here would discard
  about 9 700 negatives.
- Score: the head's logit, ranked high-first (`eb.score_rows`).
- One head per arm-seed; the head seed equals the dynamics seed.

**U + head control.** Gate-B unit U (`gb.load_model("unit", seed)`, seeds 20260908/09/10) is rolled the way the
Gate-B scoring and `eb.reference_compute` roll it:
- start: each fit shot's level anchor carrier (`eb.fit_anchors`), under the shot's executed launch (`launch5`);
- the request's fixed pair, to frame 225.
The same head architecture and recipe is then trained on U's predicted fit endpoints with the same inputs and
labels (fit data only). The U + head score minus U's cost score, and B1's head score minus its cost score, are the
exploratory "native head" contrasts.

**Budget projection** (estimated; the smoke replaces it with measured GPU throughput, and P-cap applies if over).
Estimated cost per step from FLOPs at batch 64: native s15 about 0.15 s, native s5 about 0.5 s, matched about
0.3 × native.

| item | GPU-h |
| --- | --- |
| selection (the three runs above) | ≈ 0.45 + 1.5 + 2.3 ≈ 4.3 |
| two more seeds of the selected configuration | ≤ 4.6 |
| matched, 3 seeds × 9 000 steps | ≤ 1.2 |
| heads and U + head (U rollouts of 13 919 shots × 3 seeds, 6 heads of 2 700 steps) | ≈ 1 |
| readouts (G-rank / G-action, 480 × 16; policy AUCs) and one scoring pass with position error and wrong anchors | ≈ 2–4 |
| **total** | **≈ 13–16** |

This is well inside the child share (≈ 48–88 GPU-h) and the 20 GPU-h/seed cap, even at 3 × the estimated step
cost. Over the cap, the P-cap order applies: cut the matched secondary first; never the primary's native budget
or its three seeds.
