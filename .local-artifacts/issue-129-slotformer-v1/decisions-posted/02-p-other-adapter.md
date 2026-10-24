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
