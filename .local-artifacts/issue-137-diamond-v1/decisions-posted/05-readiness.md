## Decision P-other (2026-10-27): readiness ceiling 0.6783 ≥ 0.65 — T5 is not stopped; Phase 0 continues

**Trigger.** The pre-registered readiness stop (Decision P-adapter, issuecomment-6426541110, item 7): the shared `FrameReadout`'s policy ceiling on true 64 × 64 frames, computed on CPU before any GPU work; < 0.65 stops T5 with `readiness_ceiling_below_threshold`.

**Result** (`readiness/readiness.json`, one run; fit / policy data only, evaluation not read; `deterministic_scoring()`):
- Readout: `FrameReadout` (652 559 parameters) fitted once on **220 887 true fit frames** (13 919 fit shots at positions 0, 15, …, 225 clamped; E99 targets), the #130 R1 recipe (25 epochs, 43 125 AdamW steps, batch 128, seed 20260908), CPU 4 threads, 75 min; final batch loss 0.051 (z-scored MSE); `readiness/readout.pt` `sha256:7c6d6bbb82071aa3e0d11f524fc643a12548795207aa4f3c2555c7d7fbeb0b70` — the frozen readout of every configuration and seed.
- **Ceiling = 0.6783** (pooled policy-grid AUC, 94 mixed cells, 1 504 candidates; tie-free `EndpointCosts` of readout(true endpoint frame at min(225, end)) against readout(anchor frame)); threshold 0.65 → **pass**. The E99 true-carrier ceiling on the same cells is 0.7223. Readout error against the E99 carriers of the same policy endpoints: carrier MAE 0.0157, pig-centre MAE 0.0211, pig-presence MAE 0.0012.

**Consequence.** No stop: the 64 × 64 frames carry the cost through the readout up to 0.678 (the small-object risk costs 0.044 AUC against the E99 carriers, disclosed). Phase 0 continues (G-port, G-fid, timing probe); the readiness record is also T5's G-readout value.
