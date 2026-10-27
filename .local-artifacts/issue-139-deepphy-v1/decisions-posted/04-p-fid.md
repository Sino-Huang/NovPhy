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
