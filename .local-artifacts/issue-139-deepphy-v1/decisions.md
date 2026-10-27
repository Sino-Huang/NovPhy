# #139 decisions

## Decision P-license (2026-10-27): vendor XinrunXu/DeepPHY @ 0562104 (MIT) verbatim; Qwen2.5-VL-7B-Instruct (Apache-2.0) and Qwen2.5-VL-3B-Instruct (Qwen Research Licence, non-commercial research) weights used locally and never redistributed

**Trigger.** #139 Phase 0 needs the official DeepPHY code (the Angry Birds prompt and action interface, its answer parsing and the local Qwen2.5-VL inference path) for the adapter and G-port. It also needs the official PHYRE runner for G-fid, and the open-weight VLM weights DeepPHY reports.

**Facts.**
- Code: `XinrunXu/DeepPHY` `main` @ `056210492bca103ce8aff11aa225c40bf8e7c797` (2025-11-16, "Minor updates for GUI"; HEAD on 2026-10-27). The root `LICENSE` is plain MIT, "Copyright (c) 2025 Xinrun Xu". The repository has no other licence text. The Angry Birds and Cut the Rope apps are third-party iPhone/Mac games and are not part of the repository; T7 uses neither.
- Weights (Hugging Face, the models DeepPHY's local provider `deepphy/provider/llm/qwen.py` loads; paper Table 4):
  - `Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898137f460bfe9f0759e9844b3ce807cfb5`: card licence `apache-2.0`, 16.6 GB bf16 safetensors.
  - `Qwen/Qwen2.5-VL-3B-Instruct` @ `66285546d2b821cf421d4f5eb2576359d3770cd3`: `LICENSE` = **Qwen RESEARCH LICENSE AGREEMENT** (2024-09-19). It grants use "FOR NON-COMMERCIAL PURPOSES ONLY", where Non-Commercial means "for research or evaluation purposes only"; redistribution must carry the agreement and a notice. 7.5 GB.
  - Qwen2.5-VL-32B / 72B (DeepPHY also reports these) are not downloaded; Decision P-route gives the reason.

**Options.** (a) Vendor verbatim, as MIT allows. (b) Re-implement under G-port: not needed. (c) Drop with `license`: there are no grounds.

**Choice.** (a).
1. Vendor @ 0562104 into `world_model/baselines/deepphy/vendor/` with `VENDOR.json` (schema `issue_127_vendor_v1`, per-file upstream and vendored sha256, **no edits**):
   - `LICENSE` and `README.md`.
   - `deepphy/planner/prompt_strategy_05_angrybirds.py`: the adapter imports it by file location; it has no imports.
   - Reference copies whose functions the G-port reference executes from source: `deepphy/runner/05_angrybirds_runner.py` (`get_llm_action`, `parse_and_execute_action`, the parsing lines of `play_level_with_llm`), `deepphy/provider/llm/{qwen,base}.py`, `deepphy/utils/img_utils.py` and `conf/env_config_05_angrybirds.json`.
   - The PHYRE G-fid runs the official `deepphy.runner.01_phyre_runner` from the pinned clone `/mnt/array/sukaih/novphy-baselines-src/DeepPHY@0562104/`.
2. Weights go to `data/external/deepphy/` (gitignored) with `MANIFEST.json` (repo, revision, url, bytes, sha256, downloaded_at, licence). They are used only for this NovPhy research evaluation and are never committed or redistributed. The 3B model's non-commercial terms are satisfied by this use.
3. A dedicated env `b127-deepphy` holds the official stack. Its package list goes to artefact `envs/b127-deepphy.txt`. Decision P-route gives the details.

**Reason.** MIT permits use, modification and redistribution with the notice, which is kept verbatim. Apache-2.0 permits use of the 7B weights. The 3B licence permits non-commercial research evaluation, which is exactly T7's use, provided nothing is redistributed.

Posted: https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6437037536

## Decision P-route (2026-10-27): T7 = DeepPHY's local Qwen2.5-VL path (7B-Instruct primary, 3B-Instruct as a declared configuration), unquantised fp16 under DeepPHY's pinned transformers 4.52.4, run in env `b127-deepphy` by a scoring worker; no 32B / 72B, no API

**Trigger.** #139 asks which open-weight VLMs DeepPHY reports, which of them fit a 32 GB RTX 5090 locally, and the inference interface and decoding. This fixes the env, the adapter, G-port and G-fid. Licences: Decision P-license issuecomment-6437037536.

**Facts** (vendored @ 0562104; paper arXiv 2508.05405 v1):
- DeepPHY evaluates 4 open-weight models: Qwen2.5-VL-3B / 7B / 32B / 72B-Instruct (paper Table 4).
  - Its scripts run 3B and 7B locally (`scripts/01_phyre_eval.sh`: `Qwen2.5-VL-3B-Instruct`, `Qwen2.5-VL-7B-Instruct`) and 72B through an API (`api-qwen2.5-vl-72b-instruct`); 32B is commented out.
  - The local path is `deepphy/provider/llm/qwen.py`: `transformers.pipeline(model=<path>, task="image-text-to-text", model_kwargs={"torch_dtype": torch.float16}, trust_remote_code=True)`, with no device argument (in transformers 4.52.4 that means GPU 0). Each call is `pipe(messages, max_new_tokens=200, do_sample=True, temperature=0.1, return_full_text=False)` with ≤ 3 retries. The messages are a system turn, then per image a text label followed by `{"type": "image", "url": path}`, then the user prompt. Decoding also inherits the checkpoints' `generation_config.json`: `repetition_penalty` 1.05, `do_sample` true, `temperature` overridden to 0.1, and the transformers default `top_k` 50.
- Published numbers for 7B: Angry Birds success rate 20.59 % (Table 3; this needs the iPhone app, so it cannot be run here); PHYRE VLA within 10 attempts 9.62 % (Table 6); I-PHYRE, Kinetix and Pooltool tables. 3B appears in the same tables.
- Sizes (bf16 / fp16 safetensors): 7B 16.6 GB, 3B 7.5 GB, 32B ≈ 65 GB, 72B ≈ 146 GB. DeepPHY never quantises. Only 3B and 7B fit one 32 GB RTX 5090 unquantised.
- DeepPHY's own pins (`requirements_py39.txt`): torch 2.7.1, transformers 4.52.4, tokenizers 0.21.4, safetensors 0.6.1, huggingface-hub 0.34.3, phyre 0.2.2, numpy 1.26.4. The torch 2.7.1 build pinned there is cu126, which has no RTX 5090 (sm_120) kernels.

**Options.** (a) 7B and 3B unquantised, as DeepPHY runs them locally. (b) Add 32B in 4-bit: a model DeepPHY never ran, so it is off-protocol. (c) 72B: does not fit, and DeepPHY used an API. (d) Run in the novphy env (transformers 5.20.1): this changes the inference library away from DeepPHY's pin.

**Choice.** (a), with the following setup.
1. **Models.**
   - `Qwen2.5-VL-7B-Instruct` @ cc59489 (primary).
   - `Qwen2.5-VL-3B-Instruct` @ 6628554, which becomes one of the ≤ 3 selection configurations (Decision P-adapter).
   - Both load as DeepPHY loads them: `Qwen2_5_VLForConditionalGeneration` and `AutoProcessor` from the checkpoint directory, **float16**, GPU 0 of the job's `CUDA_VISIBLE_DEVICES`.
2. **Env `b127-deepphy`.**
   - It is a clone of `b127-phyre`: py3.11, with the phyre 0.2.2 build of #128, which G-fid needs.
   - DeepPHY's pins: **transformers 4.52.4**, tokenizers 0.21.4, safetensors 0.6.1, huggingface-hub 0.34.3, openai 1.99.1, python-dotenv 1.1.1.
   - Deviations, none of which touches the prompt or model math: **torch 2.7.1+cu128** (the same version; the cu128 build carries the 5090 kernels); numpy 2.4.6 (the phyre build's numpy) instead of 1.26.4; python 3.11 instead of 3.9/3.10; accelerate added.
   - Package list: artefact `envs/b127-deepphy.txt`.
3. **Runtime split.** The runner `scripts/run_issue_139_deepphy.py` runs in env novphy: it handles the `eb` data, rows, guards and publication. It hands decision frames and candidate launches to the scoring worker `world_model/baselines/deepphy/novphy.py --worker`, which runs as a script in env `b127-deepphy`. The worker builds DeepPHY's messages, applies the processor exactly as the pipeline's `preprocess` does (`apply_chat_template(..., add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")`), and calls the model's forward. Scores therefore come from DeepPHY's pinned library and weights.
4. **Decoding.** NovPhy scoring is a teacher-forced likelihood, so there is no sampling: one deterministic pass under `deterministic_scoring()`'s flags. The three "seeds" are that one pass, and seeds are not sampling seeds (Decision P-adapter). G-fid runs DeepPHY's official sampled decoding unchanged (temperature 0.1 plus the generation config), with a fixed torch seed per run.

**Reason.** 7B is the strongest open-weight DeepPHY model that runs unquantised on one GPU, and its licence is permissive. 3B is DeepPHY's other locally-run model. Running the worker on DeepPHY's own transformers pin keeps the model call identical to the official path, without a cross-version port. Quantising 32B, or using an API for 72B, would evaluate a system DeepPHY did not run locally, or would cost money (#127: no paid API calls).

Posted: https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6437058547

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

Posted: https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6437241396

## Decision P-fid (2026-10-27): G-fid = DeepPHY's PHYRE VLA success within 10 attempts for Qwen2.5-VL-7B (paper 9.62 %), one seeded run of the official runner on the 1 000 test tasks; band [7.47, 11.77] % frozen; GPU job spec prepared (not CPU-feasible)

**Trigger.** #139 item 4: G-fid must reproduce a published DeepPHY number for one of the chosen open-weight VLMs, on an environment runnable here, with a band frozen before the run. Routes: Decisions P-route issuecomment-6437058547 and P-adapter issuecomment-6437241396.

**Facts.**
- DeepPHY publishes Qwen2.5-VL-7B numbers on six environments:
  - Angry Birds 20.59 % (Table 3): needs the iPhone app, so it cannot be run here.
  - Cut the Rope: Mac app, not runnable.
  - Kinetix (JAX) and Pooltool (3-D): not installed; each would be a new stack.
  - I-PHYRE VLA 20.00 % (Table 10): only 40 games, so the binomial SE of one run is ≈ 6.3 pp.
  - **PHYRE VLA 9.62 %** within 10 attempts (Table 6, column Att. 10, the mean over 3 runs at temperature 0.1; Att. 1 = 0.60 %; MOCK 7.64 %): 1 000 tasks.
- PHYRE is built here (#128, `b127-phyre`; cloned into `b127-deepphy`). The official runner `deepphy/runner/01_phyre_runner.py` works on CPU with DeepPHY's MOCK agent (smoke run on 5 tasks), apart from one environment shim. `phyre/vis.py` calls `np.float`, the alias of the builtin `float` that numpy ≥ 1.24 removed; DeepPHY's own pin, numpy 1.26.4, also lacks it. The fidelity wrapper sets `np.float = float`, which leaves the arithmetic identical: float64 arrays.
- The released `conf/env_config_01_phyre.json` sets an 8 × 8 grid, 8 radius levels, 10 attempts and 5 image-history attempts. The paper's appendix prints these as placeholders, so the released config is what we run.
- Cost: about 9 600 VLM calls (1 000 tasks × about 9.6 attempts, since about 90 % of tasks use all 10). Each call has a prompt of up to about 3 k tokens (≤ 27 PHYRE images at 81 tokens each, plus text) and generates up to 200 tokens. In fp16 on one RTX 5090 that is ≈ 2.1–4.5 GPU-h [INFERENCE]. On 4 CPU threads, the 7B in fp32 alone exceeds the host's 20 GB budget, and in bf16 it would take ≫ 12 CPU-h. **Not CPU-feasible.**

**Choice.**
1. **Quantity:** PHYRE success rate within 10 attempts, Qwen2.5-VL-7B-Instruct, prompt format VLA, published value **9.62 %**.
2. **Protocol:** the official `deepphy.runner.01_phyre_runner` of the pinned clone @ 0562104 is run unchanged via `scripts/b127_deepphy_fidelity.py --run` (env `b127-deepphy`).
   - Arguments: `--model Qwen2.5-VL-7B-Instruct --format VLA --eval_type test`, for `ball_within_template` and `ball_cross_template`, fold 0, 500 + 500 tasks (the paper's combined 1 000). The model loads through the official `initialize_qwen_vlm` in fp16 on GPU 0 of the job; decoding is the official `get_qwen_model_response`.
   - **One run.** torch, `random` and numpy are seeded with 20260908 plus the number of tasks already done, at process start.
   - Instrumentation only: per-task resume through the official `--reuse_log_dir` summary. A SIGTERM stops the run before the next task, with exit 75.
   - `--finish` computes the success rate over the 1 000 official per-task results and writes `fidelity/gfid.json`.
3. **Band** (`fidelity/band.json`, sha256 `ad5bfb652c1ba6151dc52ddff5944e8d5a0fb81067456e958c55784bfaf53c30`): **[7.47, 11.77] %**.
   - It is 9.62 ± 2 SE of the difference between one 1 000-task run and the paper's 3-run mean (binomial: SE₁ = 0.93 pp, SE₃ = 0.54 pp, combined 1.08 pp).
   - G-fid passes iff our value lies in the band. At most two logged repair attempts are allowed, each fixing a diagnosed port or environment defect and never the band; after that T7 is dropped with `fidelity_not_reproduced`.
   - **Disclosed:** the paper's MOCK agent (7.64 %) is inside the band. G-fid tests that the local stack reproduces DeepPHY's published Qwen-7B behaviour, not that the VLM beats random placement.
4. **Environment check** (descriptive, CPU, not a gate): DeepPHY's MOCK agent runs through the same official runner on the same 1 000 tasks (`--mock`, `random` seeded), to compare with the paper's MOCK 7.64 %.
5. **GPU job:** spec `scripts/gpu_window_issue_139_gfid.json` (job `139-gfid`, resumable, est 300 min, `shares_gpu` false, < 22 GB GPU memory, cap 6 GPU-h charged to #139). It is prepared, not enqueued: the orchestrator enqueues it with the funding decision. `--score` and `--selection` refuse until `fidelity/gfid.json` passes.

**Reason.** PHYRE is the only DeepPHY environment with a published Qwen-7B number that is both runnable here and large enough (1 000 tasks) for a tight band. The official runner is used unchanged, so a pass shows that weights, library, prompts and decoding reproduce DeepPHY's agent. The band comes from the paper's own quantity and its sampling spread, fixed before any run.

Posted: https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6437344622

band.json sha256 at posting: sha256:ad5bfb652c1ba6151dc52ddff5944e8d5a0fb81067456e958c55784bfaf53c30

## Decision P-adapter (2026-10-27): addendum. DeepPHY's `temperature=0.1` never reaches `generate` in its pinned stack; the configurations stay as declared; G-port checks the logits processing at the official effective decoding config

**Trigger.** The G-port reference was developing on a tiny random-init Qwen2.5-VL, before any released model ran and before any NovPhy frame was scored. It showed that the official call `pipe(messages, max_new_tokens=200, do_sample=True, temperature=0.1, return_full_text=False)` reaches `generate` with **only** `max_new_tokens`. This contradicts the decoding stated in Decision P-route (issuecomment-6437058547) and in P-adapter item 5 (issuecomment-6437241396).

**Facts** (transformers 4.52.4, DeepPHY's pin, `pipelines/image_text_to_text.py`):
- `_sanitize_parameters` forwards only `max_new_tokens` / `generate_kwargs` to `generate`. Every other keyword, here `do_sample` and `temperature`, goes into the preprocess parameters.
- The chat branch of `preprocess` ignores those parameters.
- `generate` therefore decodes with the checkpoint's `generation_config.json`: `do_sample` true, **`temperature` 1e-6**, `repetition_penalty` 1.05, `top_k` 50 (default), `top_p` 1.0. That is effectively greedy decoding with a repetition penalty.
- Measured on the tiny model: generate's processed scores have magnitude ~10^5 to 10^6 (logits / 1e-6). The adapter's `processed_rows` at (T 1e-6, rep 1.05) matches them with relative error 4.5e-7 in fp32 and 5.9e-4 in fp16.
- The paper (Section 5.1) and the code state temperature 0.1. Whatever DeepPHY's own runs did is reproduced by running its pinned code unchanged, which is what G-fid does.

**Options.** (a) Keep the three configurations as declared. `q7b-t01` stays DeepPHY's *stated* decoding: T 0.1 plus the checkpoint's repetition penalty 1.05. (b) Replace `q7b-t01` by the effective T = 1e-6. A likelihood at that temperature degenerates into a sum of logit gaps to the greedy token, scaled by 10^6, which is a different score family. (c) Patch DeepPHY's call: an edit of the official path.

**Choice.** (a); nothing that is already declared changes.
1. The configurations (`q7b`, `q7b-t01`, `q3b`), the score and the selection rule are unchanged.
2. G-port verifies the adapter's logits processing against generate's processed scores **at the official effective config** (T 1e-6, rep 1.05). The pass criterion is relative error ≤ 1e-3 against max(1, max |score|), at generate's finite (top-50) entries. The same function then computes `q7b-t01` at (T 0.1, rep 1.05). The teacher-forced log p check (≤ 1e-3 nats, T = 1) and the exact-input check are unchanged.
3. G-fid runs the official runner unchanged, so it uses this effective decoding. That is what DeepPHY's pinned code does; the frozen band (issuecomment-6437344622) is unchanged.
4. Correction to P-route: "temperature overridden to 0.1" is wrong under the pinned library. The effective local decoding is the checkpoint's generation config. This is disclosed in the plan.

**Reason.** The finding concerns how DeepPHY's own stack decodes, not how T7 scores. Keeping the declared configurations avoids any post-hoc change. G-port must verify the computation the official path actually performs, and the G-fid run executes exactly that path.

Posted: https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6437519414

## #139 Phase 0 gate results (2026-10-27): G-port passes (incl. the released Qwen2.5-VL-3B); G-fid band frozen, GPU spec prepared; MOCK environment check 6.20 % (paper 7.64 %); CPU projection 5.9–12.3 GPU-h [INFERENCE]

**G-port** (`gport/gport.json`, schema `issue_127_gport_v1`, **pass**; reference `scripts/b127_deepphy_port_reference.py` in env `b127-deepphy`, transformers 4.52.4, torch 2.7.1, CPU):
- **Prompts and parser.** The official `get_llm_action`, `parse_and_execute_action`, the parsing statements of `play_level_with_llm` and `get_qwen_model_response` were executed from the vendored source (GUI calls, crop and model stubbed). On 4 prompt cases, system and user prompts, image lists and chat messages are **byte-identical** to the adapter's. On 26 answers (the 16 grid answers plus code fences, multiple brackets, `success()` and malformed strings), the parser agrees in all 26 cases. The 16 grid answers round-trip.
- **Model call.** On a fit-anchor decision frame (`issue-109-sc-fit-type010103-001`), the adapter's processor outputs (`input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`; 1 202 prompt tokens) **equal exactly** what the official pipeline passes to `generate`. Teacher-forced log p of the generated tokens against generate's logits:
  - released **Qwen2.5-VL-3B** (fp32): official answer `[shoot(angle=45, power=0.8)]` (14 tokens); max |Δ log p| **2.3e-6** nats; raw logits 3.4e-5; argmax agreement 1.0;
  - tiny random-init model run through the verbatim official `initialize_qwen_vlm` (fp16): 4.9e-4;
  - tiny model in fp32: 9.5e-7.
  
  All are within the 1e-3 tolerance.
- **Logits processing** at the official effective decoding config (addendum issuecomment-6437519414): relative error 6.9e-7 (3B). Candidate scores are batch-invariant within 3.8e-6 (tolerance 1e-4), and ordinals 8 and 12 tie exactly.
- Peak RSS of the 3B run was 27.7 GB, briefly over the ~20 GB host guideline: loading the fp32 model. No further CPU runs of the released models are planned.

**G-fid** (Decision P-fid issuecomment-6437344622). Band **[7.47, 11.77] %** (PHYRE VLA Qwen-7B within 10 attempts, paper 9.62 %), `fidelity/band.json` `sha256:ad5bfb65…3c30`. GPU job `139-gfid` (`scripts/gpu_window_issue_139_gfid.json`, resumable, est. 300 min, 2.1–4.5 GPU-h [INFERENCE]); not CPU-feasible; not enqueued.
- **Environment check** (descriptive, CPU): DeepPHY's MOCK agent through the same official runner on the same 1 000 tasks reaches **6.20 %** within 10 attempts (within 5.8 %, cross 6.6 %; `fidelity/mock.json`). The paper's MOCK value is 7.64 %; the gap is 1.4 pp, about 1.5 SE of the difference.
- Environment shim (P-fid): `np.float = float` for `phyre/vis.py`.

**Projection** (`projection.json`, [INFERENCE]):
- Forward FLOPs per decision from the configs (≤ 16 distinct answers × one 1 216-token sequence with 1 564 vision patches): 7B 0.31 PFLOP, 3B 0.16 PFLOP. These are mapped to an RTX 5090 at 100 / 40 sustained fp16 TFLOP/s, with model loads.
- Totals: selection 0.2–0.4, readouts 0.5–1.1, score 0.9–2.2, novelty 0.4–1.0 and validate 0.9–2.2 GPU-h; G-fid 2.1–4.5; timing probe 1.0. **Child total 5.9–12.3 GPU-h**, against the ≈ 73 GPU-h remainder and the 18 GPU-h child cap.
- CPU reference: the released 3B in fp32 on 4 threads scores one grid decision in 538 s.
- A ≤ 1 GPU-h timing probe is prepared (`scripts/gpu_window_issue_139_timing.json`, `shares_gpu` false, est. 20 min), with its decision drafted for the orchestrator.

Nothing has been scored on NovPhy policy or evaluation data, and no GPU time has been used. The evaluation split has not been read.

Posted: https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6437774684


## Decision P-coverage (2026-10-27): a GPU timing probe (≤ 1 GPU-h, exclusive GPU) to measure DeepPHY-VLM's projection before the funding decision

**Trigger.** Program decision #127 issuecomment-6436095504 and #138's `cap` decision (issuecomment-6436855469): T7 DeepPHY (#139), the last coverage arm, is funded only if its measured projection fits the remainder (≈ 73 GPU-h).

#139 Phase 0 status on CPU:
- Licence (issuecomment-6437037536), route (issuecomment-6437058547), adapter (issuecomment-6437241396, with the addendum issuecomment-6437519414), and G-fid design (issuecomment-6437344622): band [7.47, 11.77] % frozen; GPU spec `scripts/gpu_window_issue_139_gfid.json` prepared.
- G-port: `gport/gport.json`. Prompts, parser and model inputs are exact against the official code; the teacher-forced log p matches the official generate path on a tiny model and on the released Qwen2.5-VL-3B (fp32, CPU).
- CPU projection (`projection.json`, [INFERENCE]): forward FLOPs per decision from the configs (7B: 19.5 TFLOP per 1 216-token sequence × ≤ 16 distinct answers ≈ 0.31 PFLOP per decision; 3B: 0.16 PFLOP), mapped to an RTX 5090 at 100 / 40 sustained fp16 TFLOP/s, plus model loads, G-fid (2.1–4.5 GPU-h) and this probe. That gives **5.9–12.3 GPU-h** for the child, inside the remainder and the 18 GPU-h child cap.

**Decision.** Run one probe on an exclusive GPU in the night window: `scripts/run_issue_139_deepphy.py --timing` (job `139-timing`, spec `scripts/gpu_window_issue_139_timing.json`). The spec sets `shares_gpu: false` and estimates 20 min; it runs on GPU 2 tonight, which carries no other job until #128's A4 selection (after the A4 G-fid gate, ~06:00). Per model it makes one scoring-worker pass in env `b127-deepphy` (fp16, batch 16): Qwen2.5-VL-7B for `q7b` + `q7b-t01`, and Qwen2.5-VL-3B for `q3b`. Each pass covers 8 fit-anchor decision frames × the 16 grid launches (fit split only; no policy or evaluation data) and measures:
1. the model load time;
2. the FLOPs of the first (warm-up) decision;
3. the median s/decision of the other 7.

`timing.json` (+ `ledger/timing.json`) then becomes the primary, measured projection in `projection.json` (`projection_measured`). It covers selection (both models on the 94 policy cells), readouts (480 fit anchors), scoring (512 evaluation decisions + 480 wrong anchors), novelty (464), validation (one re-score), G-fid (upper [INFERENCE] value) and this probe. **Cap: 1 GPU-h of device time, charged to #139.** Nothing from the probe is reused later.

The funding decision (`funded.json`, or `not-run.json` with typed reason `cap`) stays separate and follows the measured projection. If T7 is funded, the order is: G-fid (`139-gfid`, then `--finish` on CPU) → selection → selection readouts → smoke → the freeze → readouts → freeze readouts → score → novelty → publish / validate. If the measured projection is over the remainder, T7 is listed as not run (`cap`). P-cap's cut order, fixed now before any selection result, drops `q3b` first, then `q7b-t01`.

## Decision P-coverage (2026-10-28): T7 funded

Funded under P-coverage (program decision https://github.com/Sino-Huang/NovPhy/issues/127#issuecomment-6443206040): the exclusive-GPU timing probe measures **8.8 GPU-h** (G-fid 4.5; selection 0.2; readouts 0.5; scoring 1.05; novelty 0.5; validation 1.05), within the 18 GPU-h child cap and the remainder (≈ 73 GPU-h). `funded.json` written by the orchestrator. The PHYRE G-fid job (≈ 27 GB GPU memory with the 7B) runs first on a GPU without large co-tenants; selection only after G-fid passes.
## Decision P-fid (2026-10-28): G-fid attempt 0 misses the band (6.0 % vs [7.47, 11.77] %); one diagnosed environment deviation in the VLM's input (matplotlib glyph rendering) → logged repair attempt 1 in DeepPHY's pinned rendering stack; band unchanged

**Trigger.** Attempt 0 (`139-gfid`, the official runner, 1.61 GPU-h) scored **6.0 %** success within 10 attempts on the 1 000 PHYRE tasks. That is outside the band [7.47, 11.77] % frozen in issuecomment-6437344622 (`band.json` sha256 `ad5bfb65…3c30`), so the gate failed. The record is kept as `fidelity/gfid-attempt0.json`.
- Our curve by attempt: 0.5 / 1.0 / 2.0 / 2.2 / 3.3 / 3.8 / 4.6 / 4.9 / 5.4 / 6.0 %.
- Paper Table 6: 0.60 / 2.10 / 4.10 / 4.30 / 5.50 / 6.40 / 7.16 / 8.16 / 8.82 / 9.62 %.
- No empty responses (no OOM or call errors) and no invalid-format answers. 3 190 of 9 723 attempts were `INVALID_INPUT`.

**Diagnosis (CPU; official code unedited).**
1. **Environment outcomes: no defect.** Every one of the 9 723 parsed actions was replayed through PHYRE in two environments:
   - DeepPHY's pinned environment (py3.9, the released `phyre==0.2.2` wheel, numpy 1.26.4; env `b127-phyre022`);
   - our build (`b127-deepphy`).
   
   Both give exactly the recorded status for every action (NOT_SOLVED 6 473, INVALID_INPUT 3 190, SOLVED 60), with the same task order. `INVALID_INPUT` is therefore a legitimate agent outcome, the official `simulate_action` verdict for a ball that overlaps an object or leaves the scene. The official runner counts it as a failed attempt, so it belongs to the paper's protocol.
   
   Consequences:
   - Neither the PHYRE build nor the `np.float` shim changes any outcome. The shim is needed in DeepPHY's own pins too: `phyre/vis.py` of 0.2.2 uses `np.float`, which numpy 1.26.4 lacks, so the official runner cannot run unshimmed there.
   - The MOCK check (6.20 % against the paper's 7.64 %) is therefore not an environment offset. It is sampling noise (≈ 1.5 SE).
2. **Decoding: no defect.** In transformers 4.49, 4.50, 4.51, 4.52.4 (DeepPHY's pin), 4.53 and 4.55, the image-text-to-text pipeline does not forward `do_sample` / `temperature` to `generate`. DeepPHY's runs therefore also decoded with the checkpoint's generation config, which is effectively greedy, as ours did (addendum issuecomment-6437519414).
3. **VLM input images: one deviation found.** We rendered the official initial scenes and 8 × 8 gridded images of 20 tasks (every 50th of each setup) twice: in DeepPHY's pinned stack (matplotlib 3.9.4, pillow 11.3.0, numpy 1.26.4) and in ours (matplotlib 3.11.2, pillow 12.3.0).
   - Initial scenes, and therefore the simulation keyframes, which are rendered by the same `plt.imsave`: **pixel-identical**.
   - Gridded images: same size, same lines and scene, but **the cell-number glyphs are rasterised differently** (up to 7 972 px per image, max |Δ| 255; confined to the 64 labels).
   
   Under effectively greedy decoding, these inputs determine the agent's whole 10-attempt trajectory. This is a deviation of our environment from DeepPHY's pins in what the VLM sees.
4. Remaining unavoidable differences: the GPU (RTX 5090) and torch's CUDA build (cu128 instead of cu126, which has no sm_120 kernels).

**Options.** (a) Drop now with `fidelity_not_reproduced`. (b) One logged repair fixing the diagnosed input deviation. (c) Change the band, decoding or protocol: forbidden.

**Choice.** (b), **repair attempt 1** of at most 2.
- Run the same official runner, arguments and seeding (`scripts/b127_deepphy_fidelity.py --run --repair 1`, output `fidelity/runs-r1/`) in env **`b127-deepphy-gfid`**. This is `b127-deepphy` with DeepPHY's pinned **matplotlib 3.9.4 and pillow 11.3.0**. Verified on CPU: its initial and gridded renders of the 20 tasks are **pixel-identical** to the pinned-stack renders (40 / 40 images).
- The T7 scoring env `b127-deepphy` is unchanged.
- `--finish --repair 1` writes `fidelity/gfid.json`, with `repair_attempt` 1 and the sha256 of `gfid-attempt0.json`. The band is unchanged.
- A miss again, with no further diagnosed defect, means T7 is dropped with `fidelity_not_reproduced`.
- Package lists: `envs/b127-deepphy-gfid.txt`, `envs/b127-phyre022.txt`. Cost about 1.6 GPU-h, charged to #139.

**Reason.** The repair removes the only deviation found between our G-fid environment and DeepPHY's in the VLM's inputs, and changes nothing else. Outcomes and decoding were shown to be equivalent. Disclosed: whether the glyph difference caused the miss is not established. Under near-greedy decoding, any input perturbation reshuffles trajectories, so attempt 1 also redraws that variation, which the binomial band does not model.

Posted: https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6450386391

