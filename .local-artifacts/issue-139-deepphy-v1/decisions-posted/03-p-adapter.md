## Decision P-adapter (2026-10-27): T7 adapter (decision frame, DeepPHY prompt, launch → `shoot(angle, power)`, answer-likelihood score, ties), three pre-declared configurations, deterministic pass, and every contract-mode adaptation for a zero-shot scorer

**Trigger.** #139 item 2: the adapter `world_model/baselines/deepphy/novphy.py`, the score and the ≤ 3 configurations must be fixed before selection is implemented and before any NovPhy frame is scored. The contract modes of `round3/child-runner-contract.md` assume a trained model, so their zero-shot adaptations are declared here before they govern anything. Routes: Decision P-route issuecomment-6437058547; vendor @ 0562104 (issuecomment-6437037536).

**Upstream facts** (`deepphy/runner/05_angrybirds_runner.py`, `planner/prompt_strategy_05_angrybirds.py`, `provider/llm/qwen.py`):
- Prompts: `get_llm_action` builds the user prompt from the state (`Current Bird on Slingshot`, `Remaining Birds in Queue`), the shot history (first shot: "This is the first shot of the level. No actions have been taken yet.") and the action instructions (`shoot(angle=X, power=Y)`, `success()`). The system prompt `system_prompt_angry_birds` defines angle as "An integer from 0 to 90", 0 = horizontal right and 90 = up, and power as "A float from 0.0 to 1.0", where 1.0 is the maximum pull.
- Image: the current screenshot, labelled "This is the current game state. Plan your next shot based on this image." The runner first crops 1.10 cm off its top (`crop_image_by_cm`: the macOS iPhone-mirroring window title bar).
- Answer: "Strictly provide only the action code in '[' and ']' brackets". Parsing: strip code fences, take the last `[...]`, then the regex `shoot\s*\(\s*angle=(-?\d+\.?\d*)\s*,\s*power=(\d+\.?\d*)\s*\)`.
- DeepPHY's agent emits one sampled action and never ranks candidates. Any per-candidate score is therefore an adaptation of its interface.

**Choice.**
1. **Frame.** The NovPhy decision frame is frame 0 of the anchor branch (uint8 3 × 240 × 320), the same frames #129, #130 and #137 score:
   - fit: `eb.fit_anchor_frames()`; policy: `eb.policy_anchor_frames()`;
   - evaluation (after the freeze only): `eb.read_evaluation_anchors`; novelty: `eb.read_novelty_anchors`.
   
   The frame is 2× pixel-replicated to 640 × 480, the engine's screenshot size (as #128 A3-img), written as PNG and passed by path, as DeepPHY passes its screenshot. **No crop:** NovPhy frames have no title bar. The Qwen processor resizes to 644 × 476, giving 391 image tokens; the full prompt is 1 202 tokens.
2. **Prompt.** DeepPHY's first-shot prompt, byte for byte: current bird `red`, queue `None`, no history images. Every NovPhy bird is BirdRed, and each verdict asks whether this one launch solves the level (engine pig removal), which is DeepPHY's single-shot decision.
3. **Launch → action code.** angle = atan2(drag_y, −drag_x) in degrees, rounded to an integer as the system prompt asks. power = min(1, pull / 18.491484 px), where the BirdRed drag radius is from #94 and launch speed is proportional to the clamped pull; it is rendered with `round(·, 2)`. Answer = `[shoot(angle=X, power=Y)]`.
   - Every grid candidate is beyond the radius, so power = 1.0. Ordinals 0–15 give angles 7, 37, 54, 63, 4, 21, 35, 45, 2, 14, 25, 34, 2, 11, 19, 27.
   - Ordinals 8 and 12 (1.79° and 2.39°) both render as `angle=2`, so they **tie exactly**: 1 of the 120 pairs per cell.
4. **Score.** s(c) = Σ log p over the answer tokens followed by `<|im_end|>`, teacher-forced after the prompt.
   - Each distinct answer is one forward pass of prompt + answer through the model's own forward, in right-padded batches.
   - The configuration's logits processing is applied in HF's order: repetition penalty over the prefix, then temperature. Truncation warpers (top-k 50) are excluded, because they would put most candidates at −∞.
   - Rows: `eb.score_rows(items, s)` (value −s; the more probable launch ranks first).
   - **Ties:** identical answer strings are scored once, so they are exactly equal. There is no tie-break, and the AUC counts a tie as one half (the #123 pooled-AUC rule). A nonfinite s is a retained excluded row, which feeds G-nonfinite.
5. **Configurations** (declaration order; `eb.select_configuration` on the policy-split pooled grid AUC; ties go to fewer parameters, then declaration order):
   - `q7b`: Qwen2.5-VL-7B, T = 1 (the model's own distribution); 8.29 B parameters.
   - `q7b-t01`: Qwen2.5-VL-7B under DeepPHY's decoding distribution before truncation: repetition penalty 1.05 from `generation_config.json`, then temperature 0.1; 8.29 B parameters.
   - `q3b`: Qwen2.5-VL-3B, T = 1; 3.75 B parameters.
   
   `q7b` and `q7b-t01` share one forward pass.
6. **Seeds.** Teacher-forced scoring has no sampling. There is one deterministic pass under `harness.SCORING_POLICY`'s flags, in fp16 with batch 16. The three `eb.SEEDS` records hold this same pass: it is scored once and written per seed with `replicated_from`. The selection run is that pass on the policy cells.
7. **Contract modes for a zero-shot scorer.**
   - `--selection` (GPU; needs funded.json, G-port and G-fid): scores the 94 policy grid cells under all three configurations, which takes two model loads.
   - `--selection-readouts` (CPU): computes the AUC per configuration and writes `selection.json`.
   - `--train` is a no-op. It writes `models/T7--seed<s>/report.json` with `{"zero_shot": true, "steps": 0, "gpu_seconds": 0}`.
   - `--readouts` (GPU, selected configuration): the 480 fit anchors × 16 grid launches give G-rank (`eb.g_rank` on −s), the fit-anchor nonfinite share, MACs per candidate (`FlopCounterMode`) and wall time per decision; the policy AUC comes from selection.
   - **G-action, G-collapse and G-readout are not applied**, because there is no dynamics model, latent or readout. Arm checks: `G_rank`, `G_nonfinite`, `G_trained`, plus `G_port` and `G_fid`.
   - `--score` covers every mixed cell of every inventory from its Gate-B anchor frame. The wrong-anchor rows come from the 480 fit-anchor frames, each another level's frame, with the 16 grid launches. There is no position error, since no state is predicted.
   - `--score-novelty` scores each mixed novelty level's decision frame. `--freeze-readouts`, `--smoke`, `--prepare`, `--publish` and `--validate` work as the contract says; `--validate` re-scores one record and requires exact row equality.
8. **G-port** (tolerances fixed now).
   - Prompts and messages must be byte-identical to the official `get_llm_action` and `get_qwen_model_response`, executed from the vendored source with GUI calls stubbed, on fixed inputs. The parser must agree with the official parsing lines and `parse_and_execute_action` on fixed answers.
   - Model call: our inputs (`input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`) must equal exactly what the official pipeline passes to `generate`. Teacher-forced log p at the generated tokens must match generate's logits within ≤ 1e-3 nats. Processed scores (rep 1.05, T 0.1) must match generate's finite processed scores within ≤ 1e-3. Batch padding must change no score by more than 1e-4.
   - This runs on (i) a tiny random-init Qwen2.5-VL with the 7B processor (every code path), and (ii) the real Qwen2.5-VL-3B in fp32 on CPU, on a fit-anchor frame. The 7B in fp32 needs about 33 GB, over the shared host's 20 GB CPU budget; the 3B runs the same architecture class and code.

**Reason.** Every input stays DeepPHY's: prompt, image path, action code and inference stack. The only additions are the ones a ranking contract needs: the launch rendered in DeepPHY's own action language, and the likelihood DeepPHY's own agent assigns to emitting it. No choice here depends on any NovPhy result.
