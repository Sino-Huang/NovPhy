## Result (2026-10-28): T3 γ-model-car — disposition 4 (per-contrast tokens only); T3 fails G-action

The publication is validated: `134-publish-validate#1` exited 0 at 05:25 AEDT on 28 Oct. It hash-checked 11 records,
re-scored T3 seed 20260908 with exact row equality on every inventory, and re-derived the publication twice
byte-identically. Bindings: `validation.json` sha256 `41baa245…`, `records-manifest.json` `65970c95…`, `plan.json`
`4ef3ac5f…` (frozen 2026-10-27T13:32:33Z, commit 6be18dc4). Full write-up: `docs/issue-134-gamma-models.md`.

**Outcome.** Disposition 4: per-contrast tokens only. No scoped ranking claim either way. T3 is a coverage arm
(P-coverage), outside the program set: it neither supports nor vetoes the #127 program claim.

**Guards.**
- Every global guard passes (G-isolation, G-chronology including the novelty pass, G-scoring-policy, G-records). G-port
  (bit-identical) and G-fid (pendulum Fig. 4: ρ 0.6603 in [0.633, 1.0]) pass.
- **G-action fails on every seed**: action sensitivity of the expected occupancy carrier 0.0056 / 0.0265 / 0.0074
  against the frozen 0.05.
- G-rank, G-nonfinite (0 typed rows) and G-trained pass. G-collapse does not apply (no latent).

**Selection** (policy grid AUC, 94 cells): γ = 0.95 0.6375 (selected), γ = 1 − 1/45 0.6212, γ = 0.99 0.6117.

**Sealed grid (311 mixed cells × 3 seeds; DESCRIPTIVE).**

| ranker | pooled AUC [CI] | per seed (08 / 09 / 10) |
| --- | --- | --- |
| U | 0.7444 [0.7269, 0.7620] | 0.7141 / 0.8011 / 0.7181 |
| C\* | 0.7834 [0.7676, 0.7992] | 0.8262 / 0.7334 / 0.7905 |
| J_dec | 0.7460 [0.7277, 0.7637] | 0.7205 / 0.7933 / 0.7242 |
| P | 0.7203 [0.6965, 0.7442] | — |
| **T3** | 0.5102 [0.4919, 0.5281] | 0.6302 / 0.4652 / 0.4352 |

**Contrasts.**
- T3 − U −0.2342 [−0.2610, −0.2074], T3 − C\* −0.2732, T3 − P −0.2101, T3 − J_dec −0.2358. Every token is
  `readiness_or_precision_insufficient` (G-action).
- Wrong anchor (right − wrong): +0.0064 [+0.0004, +0.0125] (stop token).
- Descriptive inventories, where T3 falls below 0.5: offset 0.3129, angle 0.3712, power 0.3511.
- P-novelty: T3 was not scored (guard failure). The U / C\* / P references were computed.

**Compute.** 17.71 process GPU-h recorded (`compute.json`; jobs shared GPUs). About 4.2 GPU-h exclusive-equivalent for
the 250 000 training updates [estimate]. Cap: 12 GPU-h. Mean training compute is 12.26 × U's in process seconds; the
budget safeguard does not fire. The fidelity run used 1.30 CPU-h. Engine time 0. 10 230 696 parameters; 646.4 M MACs
per candidate (64 samples).

**Disclosures.**
- The smoke's cap check uses exclusive-equivalent device time (decision issuecomment-6438810533, posted before the
  smoke).
- The G-fid band is derived from a digitised figure.
- The selected seed-20260908 run (policy 0.6375) is reused as a main seed; the later seeds reach 0.4776 / 0.4307 on
  the policy split.
- The official bootstrapped loss fluctuates strongly; no run was stopped, repeated or tuned.
