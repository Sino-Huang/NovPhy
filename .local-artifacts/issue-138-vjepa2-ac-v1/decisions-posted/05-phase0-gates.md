## #138 Phase 0 gate results (2026-10-27): G-port passes; G-fid band frozen (GPU spec); readiness is a GPU job (spec); CPU projection 2 109–5 271 GPU-h [INFERENCE]

**G-port** (`gport/gport.json`, schema `issue_127_gport_v1`, **pass**). The reference is the official code from the pinned clone in env `b127-vjepa2`: the hub's `_make_vjepa2_ac_model` with the released `vjepa2-ac-vitg.pt`, `VideoTransform`, `init_opt`, `world_model_wrapper.step_predictor`, and `forward_target` / `forward_predictions` / `loss_fn` / `train_step` executed from the source text of `app/vjepa_droid/train.py`. It is compared with the adapter in env `novphy`, CPU float32, torch 2.13.0+cu130 on both sides. Inputs: 2 real fit shots × 4 stride-5 frames, with the launch at step 0.
- Preprocessing, frozen ViT-g features (load: no missing or unexpected keys) and the released AC predictor's forward (strict load, all 300 tensors) are **bit-identical**.
- Training path, cases `native` (auto_steps 2) and `ar3`: initial state equal; loss, gradients and parameters after two official updates **exact**; LR (7.5078e-5, 7.5156e-5) and WD (0.04) equal.
- The 7-step `step_predictor` rollout is within max |Δ| 2.9e-6 (tolerance 1e-5).

**G-fid** (Decision P-fid issuecomment-6436495373). The band (Diving-48 V2 test, ViT-L released probe) is frozen at **[87.54, 90.46] %** (`fidelity/band.json`, `sha256:f45da0b1…e0c3`). The data are prepared on CPU: all 1 970 V2 test videos present and decodable (`fidelity/data.json`). The run is the GPU job `138-gfid` (`scripts/gpu_window_issue_138_gfid.json`, est. 60 min [INFERENCE]); on CPU it would take ≈ 105 CPU-h. Not enqueued.

**Readiness** (C1-dino's R1 procedure; P-adapter item 6). The ceiling has not been computed. Encoding the 277 292 needed frames costs 148 CPU-h on 4 threads (1.92 s per ViT-g frame, measured), beyond the CPU allowance. Spec: `scripts/gpu_window_issue_138_readiness.json` (`138-features-readiness` 75 min → `138-readiness` 180 min, ≈ 2–4 GPU-h [INFERENCE], ~200 GB NVMe). Its decision is drafted for the orchestrator. Not enqueued.

**Projection** (`projection.json`, [INFERENCE]): CPU FlopCounterMode on the official code paths, mapped to an RTX 5090 at 100 / 40 sustained bf16 TFLOP/s, with a 3 GB/s feature-read bound.
- The native budget is 94 500 updates × 256 clips. One update costs 1.09 PFLOP (`native`, `lr-half`) or 1.93 PFLOP (`ar4`).
- Native per seed: **285–713 GPU-h** (ar4: 507–1 268). Scoring and readouts: 0.74 PFLOP, i.e. 7–19 s, per decision.
- **Child total: 2 109–5 271 GPU-h** against the ≈ 73 GPU-h remainder, the 90 GPU-h child cap and the 24 GPU-h seed cap.
- A ≤ 1 GPU-h timing probe is prepared (`scripts/gpu_window_issue_138_timing.json`, decision drafted). A `cap` not-run decision is also drafted for the orchestrator.

Nothing has been trained or scored, and no GPU time has been used. The evaluation split has not been read.
