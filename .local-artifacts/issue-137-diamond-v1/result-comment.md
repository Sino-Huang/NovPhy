## Result (2026-10-30): T5 DIAMOND-px — disposition 4 (per-contrast tokens only); T5 fails G-action

The publication is validated: `137-republish-validate#1` exited 0 at 05:40 AEDT on 30 Oct. It hash-checked 11 records, re-scored T5 seed 20260908 with exact row equality on every inventory, and re-derived the publication twice byte-identically. Bindings: `validation.json` sha256 `f7e82bfa…`, `records-manifest.json` `a79e8f18…`, `summary.json` `452d7e4d…`, `plan.json` `178b1d40…` (frozen 2026-10-28T10:36:53Z, commit 7677a2fd). Full write-up: `docs/issue-137-diamond.md`.

**Outcome.** Disposition 4: per-contrast tokens only. No scoped ranking claim either way. T5 is a coverage arm (P-coverage), outside the program set: it neither supports nor vetoes the #127 program claim.

**Guards.**
- Every global guard passes (G-isolation, G-chronology including the novelty pass, G-scoring-policy, G-records). G-port (bit-identical, including a 45-step rollout) and G-fid pass. G-fid is Tier E with the released Atari 100k agents: Pong 20.89, Freeway 32.22, Boxing 90.74, all inside the bands frozen before the run. The readiness ceiling passes (0.6783 ≥ 0.65 on true 64 × 64 frames; E99 0.7223).
- **G-action fails on every seed**: endpoint action sensitivity 0.0286 / 0.0287 / 0.0277 against the frozen 0.05.
- G-rank (tied-pair share ≤ 0.0022), G-readout, G-nonfinite (0 typed rows) and G-trained pass. G-collapse does not apply (pixel state through a readout).

**Selection** (policy grid AUC, 94 cells): native 0.6863 (selected, the Atari 100k recipe: 409 600 updates, batch 32), ar4 0.6802, native-2x 0.6523.

**Sealed grid (311 mixed cells × 3 seeds; DESCRIPTIVE).**

| ranker | pooled AUC [CI] | per seed (08 / 09 / 10) |
| --- | --- | --- |
| U | 0.7444 [0.7269, 0.7620] | 0.7141 / 0.8011 / 0.7181 |
| C\* | 0.7834 [0.7676, 0.7992] | 0.8262 / 0.7334 / 0.7905 |
| J_dec | 0.7460 [0.7277, 0.7637] | 0.7205 / 0.7933 / 0.7242 |
| P | 0.7203 [0.6965, 0.7442] | — |
| **T5** (4 406 019 params + 652 559 readout) | 0.7176 [0.6897, 0.7453] | 0.6928 / 0.7557 / 0.7044 |

**Contrasts.**
- T5 − U −0.0268 [−0.0589, +0.0046], T5 − C\* −0.0657 [−0.0973, −0.0346], T5 − P −0.0027 [−0.0409, +0.0347], T5 − J_dec −0.0283 [−0.0610, +0.0036]. Every token is `readiness_or_precision_insufficient` (G-action).
- Wrong anchor (right − wrong): +0.1601 [+0.1315, +0.1880] (stop token).
- Descriptive inventories: offset 0.6389, angle 0.6512, power 0.6604 (U 0.6670 / 0.6791 / 0.8058).
- Position error at frame 225 (descriptive): T5 0.0511 [0.0466, 0.0556], U 0.0474 [0.0440, 0.0508].
- G-action calibration (descriptive, fit split): the engine truth reaches 0.0395 in carrier space, below the frozen 0.05. T5 reaches 70–73 % of it (U 58 %, C\* 85 %). The frozen guard and the disposition stand.
- P-novelty: T5 was not scored (guard failure). The U / C\* / P references were computed.

**Compute.** 59.03 process GPU-h recorded (`compute.json`; most jobs shared GPUs). Another ≈ 6.9 GPU-h is unrecorded: a scoring launch ended by the window stop before its first record, both publish-validate runs, the smoke and the novelty pass. Total ≈ 65.9 GPU-h against the funded projection of 80.3 and the cap of 136. T5's mean training compute is 18.87 × U's (process seconds); the budget safeguard does not fire. 3.29 T MACs per candidate (8 paths × 45 steps × 3 denoising steps); 6.33 s per decision. CPU: readiness readout 75 min, G-fid 51 min. Engine time 0.

**Disclosures.**
- Post-freeze amendment #137 A1, reporting only, after scoring (issuecomment-6500578672 + addendum 6500794143). The first publication showed 0 trainable parameters (counted on the frozen model); A1 publishes the count from each training `report.json` (4 406 019) and fixes one test-setup line. No record, guard, token or disposition changed.
- Pigs are a few pixels wide at DIAMOND's native 64 × 64; the readiness ceiling is 0.044 below E99's.
- The launch enters through the official action table in the `csgo` branch's real-valued form.
- The sampler is stochastic: 8 paths per candidate with common random numbers.
- T5 chose among three pre-declared configurations on the policy split; U had no such tuning.
