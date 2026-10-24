## Decision P-license (2026-10-24): vendor SlotFormer's slot Transformer, loss and LR schedule (MIT)

**Trigger.** Phase 0 vendoring for B1 (#127 rule P-license).

**License check.**
- Upstream [pairlab/SlotFormer](https://github.com/pairlab/SlotFormer) at commit `10a6d9b7bc05ef2397ce2586da3fc0212f8c47aa` (2023-09-20, the merge of PR #8). Its `LICENSE` is MIT, "Copyright (c) 2022 Ziyi Wu", sha256 `1ddc76632bc1d6daecf1240e457e69d424f6251e8667b3d8fb69a7b98fdc2317`.
- SlotFormer's model base class and LR scheduler come from the author's toolbox [Wuziyi616/nerv](https://github.com/Wuziyi616/nerv) at tag `v0.1.0` (commit `e83ac66c6ce30e1ca3d0a287df9d3699ed9ec499`), which SlotFormer's `docs/install.md` pins. It is also MIT, "Copyright (c) 2022 Ziyi Wu", with a byte-identical `LICENSE`.
- Released weights (`pretrained.zip`, Google Drive folder linked from `docs/benchmark.md`) are used only for the Tier-E fidelity check. They are never redistributed or committed, and they stay under `data/external/slotformer/` (gitignored).

**What is vendored** into `world_model/baselines/slotformer/vendor/`: verbatim copies, sha256-pinned in `VENDOR.json` (schema `issue_127_vendor_v1`).
- `slotformer/video_prediction/models/slotformer.py`: `SlotRollouter`, the sine temporal P.E. and `SlotFormer`, including the official loss `calc_train_loss` (per-slot MSE, the `vid_len` truncation and the RPIN temporal weighting).
- `slotformer/video_prediction/models/single_step_slotformer.py`: `SingleStepSlotRollouter` / `SingleStepSlotFormer`, the PHYRE burn-in-1 model with iterative overlapping and a `cond_len` window.
- nerv `nerv/training/model.py` (`BaseModel`) and `nerv/training/lr.py` (`CosineAnnealingWarmupRestarts`, the official scheduler).
- Both `LICENSE` files.

**Edits.**
- The `nerv` import in `slotformer.py` becomes package-relative.
- `slotformer.py`'s `from slotformer.base_slots.models import StoSAVi` is replaced by `StoSAVi = None`. StoSAVi is used only by `SlotFormer._build_decoder` / `decode`, i.e. the frozen SAVi pixel decoder for the optional image loss. The NovPhy adapter has no image decoder: E99 slots are the state. The adapter overrides both methods, so vendoring SAVi plus its nerv module tree would add unused code.
- Every edit is listed in `VENDOR.json` with its reason.

**Not vendored.**
- Optimizer construction lives in `SlotBaseMethod._configure_optimizers` / nerv `BaseMethod`, which pull in the wandb trainer. The adapter re-states it (Adam on the trainable parameters, weight decay 0, no clipping; `CosineAnnealingWarmupRestarts(total, max_lr=lr, min_lr=lr/100, warmup_steps=0.05 total)`, stepped per step). G-port holds it to the official methods, which run in the reference env.
- SAVi / evaluation code: the Tier-E check runs the official `test_vp.py` from the pinned clone in its own env (`b127-slotformer`). Nothing from it is imported at NovPhy run time.

**Options considered.**
- (a) Vendor SAVi and the nerv model tree verbatim: more unused code, no gain in fidelity.
- (b) Re-implement the slot Transformer from the paper: rejected. The license permits vendoring, and an official copy is the faithful choice.
- **Choice: (c)**, the minimal verbatim vendor above.

## Decision P-other (2026-10-24): B1 adapter — SlotFormer's PHYRE model on E99 slots

**Trigger.** #129 needs SlotFormer's slot Transformer on the E99 carrier with the launch injected at t0. Every choice that the vendored code (Decision P-license above) does not fix is listed here, before any training. Priority, per P-other: validity, then fairness to the baseline, then cost.

**Model.** The official `SingleStepSlotFormer` is used unchanged; it is the model SlotFormer uses for PHYRE planning.
- Burn-in 1, with the iterative-overlapping rollout over a `cond_len` = 6 window.
- Sine temporal P.E., no slot P.E.
- Pre-norm `nn.TransformerEncoder` with dropout 0.1 (the torch default the official code keeps).
- `in_proj` / `out_proj` linear maps between the slot width and `d_model`.

The adapter (`world_model/baselines/slotformer/novphy.py`, class `SlotFormerE99`) subclasses it only to drop the SAVi pixel decoder: there is no image loss, and the E99 slots are the state.

**Choices.**

1. **Slots = the 18 E99 vocabulary slots × 13 `ENTITY_FEATURES`** (carrier columns 2..235), raw, with no z-scoring.
   - SlotFormer feeds SAVi slots as they come, and #123 also fed carriers without z-scoring.
   - The 18 slots form a fixed vocabulary; all 18 are declared in every fit carrier.
   - An absent object is a slot with presence ≈ 0. It is kept as a token, like SAVi's empty/background slots. Nothing is masked.
   - No slot P.E., as in the official config. Slot identity is still visible through `kind_index_normalized` / `expected_kind_probability`.
2. **The header (has-prior flag, elapsed seconds) is not a token.** It is parse-schedule metadata, not an object.
   - A predicted carrier gets `ROLLED_HEADER = (1, 0.02)`, the header that ≥ 99.8 % of non-anchor fit carriers have. The rest are rare short-interval frames.
   - `EndpointCosts` reads no header column.
3. **Launch = one extra token at t0.** The 5-d launch vector `[drag_x/480, drag_y/480, release/1000, tap/1000, 1]` (`eb.launch5`) is zero-padded to the slot width, `[launch5, 0 × 8]`, and appended to the anchor's 18 slots, giving 19 tokens.
   - This is SlotFormer's PHYRE recipe: the intervention is inserted into the slot set, then the scene is rolled out from the first frame. It adds no new module.
   - The token is rolled forward and fed back like every slot. Its target at every step is the same token: the injected object persists. This lets the official loss apply verbatim to all 19 slots, and it keeps the launch inside the 6-frame `cond_len` window once frame 0 has left it.
   - The token is distinguishable from carrier slots: `slot_declared` = 0, `presence_available` ≠ 1.
4. **Frames:** the anchor (shot frame 0) plus every 15th frame to 225, so 1 burn-in frame + 15 rollout steps. Stride 5 (45 steps) is one of the three pre-declared selection configurations. Training sequences start at the launch frame 0, as in PHYRE: one sequence per fit shot.
5. **Shots ending before frame 225** (601 of the 13 919 fit shots; shortest end 138): the official PHYRE path.
   - `fix_video_len` pads with the last real strided frame.
   - `vid_len` (real strided frames) truncates the loss in the official `calc_train_loss`.
   - Option rejected: training on the padded frames as targets (U's #112 terminal padding), because it changes the official loss. The padded fraction is under 1.5 % of target frames at stride 15.
6. **Loss:** the official slot reconstruction `F.mse_loss` over the rolled slots (teacher signal = the true strided carriers; the rollout is autoregressive from the burn-in frame, as in the official training).
   - `slot_recon_loss_w` = 1.
   - No image loss.
   - No RPIN temporal loss decay (`use_loss_decay = False` in the PHYRE config).
7. **Optimizer:** the official recipe, with batch 64 (the PHYRE config's `train_batch_size`).
   - Adam, lr 2e-4, weight decay 0, no gradient clipping.
   - nerv `CosineAnnealingWarmupRestarts` from lr/100, with linear warm-up over the first 5 % of steps and cosine decay to lr/100, stepped every optimizer step.
8. **Sizes:**
   - **native** (primary): the PHYRE config's encoder, d_model 256 × 8 layers, 8 heads, ffn 1024. 6 325 005 trainable parameters.
   - **matched:** U's budget at the same depth, heads and ffn ratio (4 d), with d_model 136 and ffn 544. 1 793 445 parameters, −2.41 % of U's 1 837 690, inside ±5 %. Trained with U's B1 recipe, 9 000 updates × 64.
   - There are no heads: the model has no trainable module outside the rollouter.
   - The native step count (the PHYRE config's 50 epochs ≈ 300 k steps on its data) is fixed in the #129 plan before selection, within the 20 GPU-h/seed cap.
9. **Score:** the frame-225 predicted carrier = `ROLLED_HEADER` + the 18 predicted carrier slots (the launch token is dropped). `EndpointCosts` (tie-free) applies with no readout.

**G-port** (CPU, float32; official code from the pinned clone in env `b127-slotformer`, torch 2.7.1, against the adapter in novphy, torch 2.13). Four arms were checked, each with forward, loss, gradients and two official optimizer steps: step 1 at the step-0 LR 2e-6, step 2 at 2e-4.
- `official-phyre`: the official PHYRE config, 8 × 128 slots.
- `b1-native-s15`, `b1-matched-s15` and `b1-native-s5`: built from carriers, with two of the four shots short, so the `vid_len` path runs.

Result: **all pass.**
- Losses are bit-equal.
- Gradient tensors match within 4.4e-4 relative L2.
- Parameter counts, optimizer and scheduler are identical.
- The adapter's packed batches equal the official `fix_video_len` batches exactly.
- Parameter tolerance follows #123. A first all-elements allclose run failed only on Adam-amplified elements, which the record lists.
  **[Corrected 2026-10-24 (issuecomment-6383196400, "Decision P-fid (2026-10-24): correction" below): this was not #123's rule but a share-based variant of it, with per-element isclose(rtol 1e-5, atol 1e-6) instead of |Δ| ≤ 1e-7. #123's exact rule fails for b1-native-s5 step 2 (share 0.99202).]**

## Decision P-route (2026-10-24): run the official SlotFormer code unmodified on torch 2.7.1 + CUDA 12.8 (env `b127-slotformer`)

**Trigger.** SlotFormer's `docs/install.md` pins Python 3.8.8, PyTorch 1.10.1 + CUDA 11.3, nerv v0.1.0, `phyre==0.2.2` and `einops==0.3.2`. CUDA 11.3 builds have no kernels for the RTX 5090 (sm_120). The `phyre` 0.2.2 wheels exist only for Python 3.6, yet SlotFormer's dataset package imports `phyre` unconditionally.

**Options considered.**
- (a) Port the code to the novphy env (torch 2.13). Rejected: it would mix the reference stack with ours.
- (b) Build PHYRE from source in a Python 3.6 / torch 1.10 env. Rejected: it cannot use the GPU, and PHYRE is unused by the CLEVRER evaluation and by G-port.
- (c) **Chosen:** a separate env, `b127-slotformer`, with no edit to any official file.

**The env** (Python 3.10; package list in the artefact `envs/b127-slotformer.txt`, sha256 `ea85388def7e982cbff9d51f2ff13229a065f86d9651464d49a220600135682a`):
- torch 2.7.1 + cu128 and torchvision 0.22.1 (the oldest torch line with sm_120 kernels);
- nerv v0.1.0 (commit e83ac66), installed `--no-deps` with only the packages its imported modules need. open3d and pytorch-lightning are listed in its setup.py but never imported;
- numpy 1.26.4, scikit-image 0.25.2, lpips 0.1.4, pycocotools, einops 0.3.2 and pandas.

Both the evaluation and the G-port reference run the official files from the pinned clone `SlotFormer@10a6d9b`.

**Run-time measures.** None of these touches official code.
1. A `phyre` shim module is placed on `PYTHONPATH`. Any attribute access raises, so a PHYRE call would fail loudly. Only the import is satisfied.
2. The evaluation runs from a work directory (`fidelity/run`) with the relative paths the official config expects: `pretrained/` and `data/CLEVRER/`. Both are symlinks to `data/external/slotformer/`.
3. The released checkpoints load under torch ≥ 2.6's `weights_only=True` default (checked on CPU), so no load flag is changed.

**Data assembled for the official CLEVRER evaluation.** The official CLEVRER layout is built under `data/external/slotformer/CLEVRER/` (MANIFEST.json there; CLEVRER is CC0):
- the #106 validation videos (symlinked);
- the validation annotations (unzipped from the #106 copy);
- the `derender_proposals` object masks (downloaded, 396 807 825 B, sha256 `68ea4785…b7f3`);
- the released `clevrer_slots.pkl` (13 581 152 270 B, sha256 `05bbab80…ef27`).

**Check.** A CPU dry run of the official pipeline ran on 8 validation clips with the released weights: dataset build (2 314 filter_enter clips), slots, masks, forward and metrics, with no error. This is a pipeline check only; the Tier-E statistic comes from the full GPU run.

## Decision P-fid (2026-10-24): frozen Tier-E band for B1, CLEVRER mIoU 49.42 ± 0.5

**Trigger.** The B1 Tier-E G-fid run, one evaluation run of the released weights, is about to be enqueued on the night-window scheduler as job `129-gfid-tier-e` (GPU 1). The band is frozen before that run.

**Target.** CLEVRER video-prediction mIoU **49.42**: README of pairlab/SlotFormer @ 10a6d9b, update 2023-09-20.
- The README says "The updated mIoU of SlotFormer on CLEVRER is **49.42** (using the provided pre-trained weight)".
- That number follows the mIoU fix in PR #7 (commit fb1c39b: union = |A| + |B| − |A ∩ B| in `vp_utils.hungarian_miou`) and was posted in PR #8.
- The paper's pre-fix Table 2 value is not the target.

**Metric.**
- The official `slotformer/video_prediction/test_vp.py` prints `miou: {v.mean():.4f}`; ours is that value × 100.
- `v` holds the 42 per-step means (6 burn-in + 42 rollout frames, `frame_offset` 2, 64 × 64) of the per-clip Hungarian-matched IoU between the ground-truth foreground masks (CLEVRER `derender_proposals`) and SlotFormer's post-processed slot masks.
- The clips are the CLEVRER validation videos under the official `filter_enter` rule, with burn-in slots from the released `clevrer_slots.pkl`.
- Weights: `slotformer_clevrer_params/model_80.pth` and the frozen SAVi decoder `stosavi_clevrer_params/model_12.pth`.

**Band:** [48.92, 49.92] (#127 / #129: 49.42 ± 0.5), from one evaluation run.
- Pass iff ours lies inside the band.
- A fail follows P-fid: at most two logged repair attempts, each fixing a diagnosed port defect, never the band; then the arm is dropped with `fidelity_not_reproduced`.
- The optional Tier T (PHYRE planning AUCCESS 82.0 ± 1.1) is not run in Phase 0.

`fidelity/band.json` sha256: **`02ff5b00e06530e63927b3ff26f5a13c6c3213695c3eba40f33f1f4a437868de`**. `scripts/b127_slotformer_fidelity.py` refuses to run unless this hash is in `decisions.md`.

**Job.** `129-gfid-tier-e`, GPU 1, non-resumable, `est_minutes` 60.
- The CPU dry run measured about 1.1 s per clip of single-process data loading. There are 2 314 clips; the loader uses 8 workers.
- Loading the dataset and its annotations took 106 s, and the slot pickle is 13.6 GB.
- On SIGTERM the run exits 75 and records no number.
- Command: `/srv/home/shua0171/miniconda3/envs/b127-slotformer/bin/python -u scripts/b127_slotformer_fidelity.py`, which writes `fidelity/gfid.json`.

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

## Decision P-crash (2026-10-24): selection runs relaunched once after a co-scheduling OOM

**Trigger.** The three selection jobs (`129-select-c1/c2/c3`) were launched together on GPU 1 at 20:40 as
`shares_gpu` jobs. All three died within their first steps with CUDA out-of-memory (the logs show about 8.3 GB per
stride-15 run and over 14.8 GB for the stride-5 run, so they could not share the 32 GB GPU). No optimizer state,
checkpoint or model was written. This is an infrastructure crash caused by my memory estimate, not a typed
failure of any configuration.

**Choice.** Relaunch once (P-crash), changing only the co-scheduling, never a configuration:
- `129-select-c1` and `129-select-c3` (both stride 15) run together on GPU 1, about 17 GB;
- `129-select-c2` (stride 5) runs alone on GPU 1 after both.

The runs restart from step 0 with the same seed and batch stream. The relaunch started at 20:50, and both
stride-15 runs passed step 200 with finite, decreasing loss. Disclosed in the child doc.

## Decision P-fid (2026-10-24): correction — the B1 G-port parameter criterion is a declared variant of #123's rule

**Trigger.** The pre-freeze review (Review129, P2) found an error in my earlier comments on #129: the adapter
decision (issuecomment-6382359850) and `decisions.md` said that the G-port parameter tolerance "follows #123". It
does not:
- **#123's rule** (`run_hjepa_comparison.GPORT_TOLERANCE`): the share of elements with |Δparam| ≤ 1e-7 (absolute)
  must be ≥ 0.999, and max |Δ| ≤ 2·lr_max + 1e-7.
- **The B1 compare** gated on the share of elements with `isclose(rtol 1e-5, atol 1e-6)`.

**What changed.** `gport/gport.json` now records both, for every arm and step:
- `param_close_share_abs`, `pass_123_exact` and `G_port_123_exact`: #123's exact rule;
- `param_close_share_isclose` and `G_port`: the variant.

Both use #123's max-|Δ| bound, with lr_max taken after the step as in #123. Every other check (loss, gradient
tensors, grad norm, Adam moments, LRs, optimizer, scheduler, parameter counts, batches, predictions) is the same
under both rules.

**Result.**
- **#123's exact rule fails** for one arm-step: `b1-native-s5`, step 2 (the positive-LR step at lr 2e-4), with
  an absolute share of 0.99202 (< 0.999). Its isclose share is 0.99984, and max |Δ| is 3.1e-4 ≤ 2·lr = 4e-4.
- The other 11 arm-steps pass #123's exact rule: shares 0.99922–0.99999 (official PHYRE config, native and
  matched stride 15, both heads).
- **The variant passes** for all six arms.

So **`G_port` = PASS rests on a declared variant**: a share-based version of #123's rule, loosened per element from
|Δ| ≤ 1e-7 to isclose(rtol 1e-5, atol 1e-6). The loosening is needed because Adam normalises every element, which
amplifies round-off-level gradient differences between torch 2.7.1 (official side) and torch 2.13 (adapter).
#123's positive-LR step ran at 5e-5; ours runs at the official 2e-4.

**Evidence that this is float noise, not a port defect.**
- **Losses** are bit-equal in all 12 arm-steps; the one exception is head-b1 step 2, with a relative delta of 8e-8.
- **Gradients:** tensor relative L2 ≤ 6.6e-5 everywhere except `b1-native-s5` step 2, at 4.3e-4. That run has 45
  autoregressive steps, and it starts step 2 from parameters that already differ by the amplified step-1 round-off.
- **Step 2 rerun from the official step-1 state** (non-gating diagnostic `step2_from_official_step1_state`; the
  official step-1 parameters and Adam state are loaded, so step-1 differences cannot propagate):
  - `b1-native-s5` gives an absolute share of 0.99960 and passes #123's rule;
  - `b1-native-s15`, which passes in the chained run, gives 0.99759 here.

  The absolute 1e-7 share at lr 2e-4 therefore moves on either side of 0.999 with the run's float history. It does
  not track a systematic difference.

The orchestrator decides whether the variant is acceptable. Until then this is recorded as a disclosed variant,
not as #123's rule.

## Decision P-fid (2026-10-24): follow-up to the G-port correction — the frozen G-port reference runs on torch 2.13

**Trigger.** The orchestrator decided that B1's G-port must not rest on a loosened criterion when the cause can be
removed. The cause was the torch version: the official reference ran on 2.7.1 and the adapter runs on 2.13. A1, A2
and A4 got Δ = 0 because their official sources ran on the adapter's torch.

**Change.** This removes the version difference; it does not change any tolerance.
- **Official side:** `scripts/b127_slotformer_port_reference.py reference` now runs the official SlotFormer and
  nerv sources from the pinned clones (`SlotFormer@10a6d9b`, `nerv@e83ac66`, on `PYTHONPATH`) on **torch
  2.13.0+cu130, CPU**, the same build as the adapter. It runs in venv `b127-slotformer-ref213`, a
  `--system-site-packages` venv over the novphy env; its package list is in `envs/b127-slotformer-ref213.txt`.
- **Gate:** `G_port` now gates on **#123's exact rule**: share of |Δparam| ≤ 1e-7 ≥ 0.999 and max |Δ| ≤ 2·lr_max
  + 1e-7. `gport.json` records `G_port_rule: issue_123_exact` and the reference torch and Python.
- **Smoke:** `--smoke` requires this rule.
- **Diagnostics, recorded but not gating:**
  - the isclose share, as `G_port_isclose_variant`;
  - the torch-2.7.1 comparison, kept as `gport/torch271/gport.json`. It was referenced from `diagnostic_torch_2_7_1`
    with sha256 `f1e9f999…ded6`, `G_port_123_exact` = false.

**Result: G-port PASS under #123's exact rule, bit-identical.**
- All six arms pass: the official PHYRE config, native stride 15, matched stride 15, native stride 5, the official
  readout and the #129 head.
- In both steps of every arm, loss, gradient and Adam-moment deltas are exactly 0, the absolute parameter share is
  1.000000, and max |Δparam| = 0.
- Predictions are identical, and so is the step-2-from-official-state diagnostic.

