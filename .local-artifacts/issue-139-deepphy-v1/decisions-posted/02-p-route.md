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
