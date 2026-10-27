## Decision P-coverage (2026-10-27): a GPU timing probe (≤ 1 GPU-h, exclusive GPU) to measure DeepPHY-VLM's projection before the funding decision

**Trigger.** Program decision #127 issuecomment-6436095504 and #138's `cap` decision (issuecomment-6436855469): T7 DeepPHY (#139), the last coverage arm, is funded only if its measured projection fits the remainder (≈ 73 GPU-h).

#139 Phase 0 status on CPU:
- Licence (issuecomment-6437037536), route (issuecomment-6437058547), adapter (issuecomment-6437241396, with the addendum issuecomment-6437519414), and G-fid design (issuecomment-6437344622): band [7.47, 11.77] % frozen; GPU spec `scripts/gpu_window_issue_139_gfid.json` prepared.
- G-port: `gport/gport.json`. Prompts, parser and model inputs are exact against the official code; the teacher-forced log p matches the official generate path on a tiny model and on the released Qwen2.5-VL-3B (fp32, CPU).
- CPU projection (`projection.json`, [INFERENCE]): forward FLOPs per decision from the configs (7B: 19.5 TFLOP per 1 216-token sequence × ≤ 16 distinct answers ≈ 0.31 PFLOP per decision; 3B: 0.16 PFLOP), mapped to an RTX 5090 at 100 / 40 sustained fp16 TFLOP/s, plus model loads, G-fid (2.1–4.5 GPU-h) and this probe. That gives **5.9–12.3 GPU-h** for the child, inside the remainder and the 18 GPU-h child cap.

**Decision.** Run one probe on an exclusive GPU in the night window: `scripts/run_issue_139_deepphy.py --timing` (job `139-timing`, spec `scripts/gpu_window_issue_139_timing.json`). The spec sets `shares_gpu: false` and estimates 20 min; the orchestrator sets the free GPU. Per model it makes one scoring-worker pass in env `b127-deepphy` (fp16, batch 16): Qwen2.5-VL-7B for `q7b` + `q7b-t01`, and Qwen2.5-VL-3B for `q3b`. Each pass covers 8 fit-anchor decision frames × the 16 grid launches (fit split only; no policy or evaluation data) and measures:
1. the model load time;
2. the FLOPs of the first (warm-up) decision;
3. the median s/decision of the other 7.

`timing.json` (+ `ledger/timing.json`) then becomes the primary, measured projection in `projection.json` (`projection_measured`). It covers selection (both models on the 94 policy cells), readouts (480 fit anchors), scoring (512 evaluation decisions + 480 wrong anchors), novelty (464), validation (one re-score), G-fid (upper [INFERENCE] value) and this probe. **Cap: 1 GPU-h of device time, charged to #139.** Nothing from the probe is reused later.

The funding decision (`funded.json`, or `not-run.json` with typed reason `cap`) stays separate and follows the measured projection. If T7 is funded, the order is: G-fid (`139-gfid`, then `--finish` on CPU) → selection → selection readouts → smoke → the freeze → readouts → freeze readouts → score → novelty → publish / validate. If the measured projection is over the remainder, T7 is listed as not run (`cap`). P-cap's cut order, fixed now before any selection result, drops `q3b` first, then `q7b-t01`.
