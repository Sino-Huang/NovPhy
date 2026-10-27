## Decision P-coverage (2026-10-27): V-JEPA 2-AC (#138, T6) is not run — typed reason `cap` (projection far above the remainder)

**Trigger.** Program decision #127 issuecomment-6436095504 funds T6 only if its projection fits the remainder (≈ 73 GPU-h after DIAMOND). #138 Phase 0 is complete on CPU:
- licence, route, adapter and fidelity design are posted: issuecomment-6436220188 / -6436321490 / -6436355731 / -6436495373;
- G-port passes (`gport/gport.json`: exact on the official training path, rollout ≤ 2.9e-6);
- the G-fid band is frozen and readiness is a GPU job (both specs prepared, not run).

**Projection** (`projection.json`, [INFERENCE]: CPU FlopCounterMode on the official code paths, mapped to an RTX 5090 at 100 / 40 sustained bf16 TFLOP/s):
- Native budget (the released checkpoint: 315 epochs × 300 iterations = 94 500 updates at global batch 256 = 8 per GPU × 32 GPUs). One update is 256 clips × 4.24 TFLOP = 1.09 PFLOP (ar4: 1.93 PFLOP).
- Native per seed: 285–713 GPU-h (ar4: 507–1 268). Selection (3 configurations) + 2 seeds + features (≈ 1–2.5) + readiness (≈ 0.5–1) + readouts / scoring (≈ 15–37, 0.74 PFLOP per decision) gives **2 109–5 271 GPU-h**.
- The remainder is ≈ 73 GPU-h. Even one native seed at the optimistic bound (≥ 285 GPU-h) is ≈ 3.9× the remainder and ≈ 12× the 24 GPU-h seed cap. The child cap is 90 GPU-h, the upper end of the scout estimate.

**Options.**
- (a) Fund: impossible within the remainder.
- (b) Cut under P-cap. The declared cut order drops `lr-half` and then `ar4`, leaving native × 3 seeds ≥ 855 GPU-h. A shorter schedule would weaken the baseline below its native budget, which #127 forbids for a fair comparison.
- (c) **Not run, typed `cap`**.
- (d) Spend the ≤ 1 GPU-h timing probe first (`draft-decision-timing.md`). The optimistic CPU bound already exceeds the remainder by ≈ 29×, so no plausible measurement changes the outcome. A sustained rate of ~2.9 PFLOP/s would be needed, > 10× the RTX 5090's dense bf16 peak.

**Choice.** (c). Write `not-run.json` `{"typed_reason": "cap", "decision": <this comment>}` and publish the not-run child summary (`python scripts/run_issue_138_vjepa2_ac.py --publish`, CPU). T6 is listed as not run in the program summary. No GPU time is spent on #138; the readiness, G-fid, timing and phase-A specs stay prepared and unqueued.

**Reason.** The projection depends only on the official native budget and the measured FLOPs of the official code paths, not on any result. The decision is the P-coverage rule applied as written.
