## #137 Phase 0 gate results (2026-10-27): G-port bit-exact; G-fid (Tier E) passes; CPU projection [INFERENCE]

**G-port** (`gport/gport.json`, schema `issue_127_gport_v1`, **pass, exact**). Official code (`Trainer.train_component`, `utils.configure_opt` / `get_lr_sched`, `Denoiser.forward`, `DiffusionSampler.sample`, `WorldModelEnv.step`, the official `Episode` / `make_segment` / `collate_segments_to_batch`; objects instantiated from the vendored `config/*.yaml`; env `b127-diamond`) against the adapter (env `novphy`), CPU float32, torch 2.13.0+cu130 on both sides, three cases: `atari-discrete` (the unchanged official model, 6 discrete actions, full LR 1e-4), `launch` (num_actions 5, the `csgo`-branch real-valued action table; real NovPhy fit windows incl. left padding and the launch frame; warm-up steps at LR 0 and 1e-6; plus a full **45-step rollout** of 2 fit anchors under 2 grid launches) and `launch-ar4` (9-frame windows). Initial state, the official batch, forward loss, both optimizer steps (loss, LR, pre-clip gradient norm, post-clip gradients, parameters), one sampler call and all 45 rollout frames: **all deltas 0**. Injected defects are detected (CPU tests): launch one frame early (batch differs), no warm-up (LR differs), one noise draw reused at every rollout step (rollout differs).

**G-fid** (`fidelity/gfid.json`, schema `issue_127_gfid_v1`, **pass**; band frozen before the run, issuecomment-6426803157, `sha256:a21b44ec…0cca`). Released agents (HF @ 5d4abca), official test env and collector, ε 0, 100 episodes per game, CPU (≈ 17 CPU-min per game), 0 repair attempts:

| game | ours (100 episodes, mean ± SE) | paper 5-seed mean (Table 1) | band (mean ± 2 SD) | pass |
| --- | --- | --- | --- | --- |
| Pong | 20.89 ± 0.05 | 20.36 (20.4) | [18.963, 21.757] | yes |
| Freeway | 32.22 ± 0.08 | 33.32 (33.3) | [32.108, 34.532] | yes |
| Boxing | 90.74 ± 0.49 | 86.90 (86.9) | [63.676, 110.124] | yes |

(Freeway lies 0.11 above its band's lower end; the paper's own seeds span 32.4–33.8.)

**Projection** (`projection.json`, [INFERENCE] CPU micro-benchmark; the measured probe replaces it): FLOPs per official update native 1.17 / ar4 2.92 TFLOP (batch 32), per scoring decision 105 TFLOP (16 candidates × 8 paths × 45 steps × 3 denoising steps + readouts). Mapped to an RTX 5090 at 40 / 10 sustained TFLOP/s: **40.3–161.1 GPU-h** total (selection 3.3–13.3 + 8.3–33.3 + 6.7–26.6; two seeds at ar4 16.6–66.5; readouts 1.3–5.0; scoring 3.9–15.5), serial-ladder wall 21.1–84.5 h. Paper anchor (App. I RTX 4090 timings, unscaled): **109.7 GPU-h**, wall 53.9 h (ar4 25.0 GPU-h per seed). The range straddles the program remainder (≈ 122 GPU-h): a ≤ 1 GPU-h timing probe on an exclusive GPU is drafted for the orchestrator (`draft-decision-timing.md`, spec `scripts/gpu_window_issue_137_timing.json`, not enqueued). Nothing has been trained on a GPU; the evaluation split has not been read.
