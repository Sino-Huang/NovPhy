## Decision P-coverage (2026-10-27): a GPU timing probe (≤ 1 GPU-h, exclusive GPU) to measure V-JEPA 2-AC's projection before the funding decision

**Trigger.** Program decision #127 issuecomment-6436095504: V-JEPA 2-AC (#138, the coverage arm after DIAMOND) is funded only if its measured projection fits the remainder (≈ 73 GPU-h).
- #138 Phase 0 status on CPU: vendoring (issuecomment-6436220188), route (issuecomment-6436321490), adapter (issuecomment-6436355731), and G-fid design (issuecomment-6436495373; band frozen, GPU spec).
- G-port passes: `gport/gport.json` is exact on the training path, and the rollout is within 2.9e-6.
- Readiness and G-fid are prepared as GPU specs.
- The CPU projection (`projection.json`, [INFERENCE]) counts FLOPs per official update: native 1.09 PFLOP and ar4 1.93 PFLOP per update of 256 clips, plus 0.74 PFLOP per scoring decision. Mapped to an RTX 5090 at 100 / 40 sustained bf16 TFLOP/s, that gives **2 109–5 271 GPU-h** for the child, with native 285–713 GPU-h per seed against the 24 GPU-h seed cap.

**Decision.** Run one probe on an exclusive GPU in the night window: `scripts/run_issue_138_vjepa2_ac.py --timing` (job `138-timing`, spec `scripts/gpu_window_issue_138_timing.json`). The spec sets `shares_gpu: false`, keeps every other job off that GPU meanwhile, and estimates 30 min; the orchestrator sets the free GPU. It keeps no model and measures:
1. the frozen ViT-g encoder on 512 fit frames (s/frame);
2. 6 timed official updates each of `native` (= `lr-half` per update) and `ar4`: 256 clips as 32 micro-batches of 8, bf16 autocast + GradScaler, activation checkpointing, on clips of those encoded frames held in GPU memory. The feature-cache read bound (1.48 GB per update at 3 GB/s) stays modelled;
3. the scoring path from 4 fit anchors (anchor encode + 45-step rollouts of the 16 grid candidates + R1) under `deterministic_scoring()`;
4. 50 R1 steps.

`timing.json` (+ `ledger/timing.json`) then becomes the primary, measured projection in `projection.json` (`projection_measured`). It covers features, readiness, three selection runs, two more seeds at the most expensive configuration, selection readouts, readouts and scoring of three seeds including novelty, G-fid and this probe. **Cap: 1 GPU-h of device time, charged to #138.** Nothing from the probe is reused later.

The funding decision (`funded.json`, or `not-run.json` with typed reason `cap`) stays separate and follows the measured projection. If that projection is over the remainder, T6 is listed as not run (`cap`). P-cap's cut order, fixed before any selection result, would drop `lr-half` and then `ar4`. Even one native seed is projected at ≥ 285 GPU-h [INFERENCE], so no cut brings T6 within the remainder.
