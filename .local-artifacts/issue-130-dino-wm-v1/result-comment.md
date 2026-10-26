## Result (2026-10-27): C1 DINO-WM — C1-dino typed stop (readiness), C1-car disposition 4; per-contrast tokens only

Frozen plan `a683498d` (2026-10-25T11:38:19Z) → readouts frozen 2026-10-25T12:15:27Z → Tier-E G-fid decided
(2026-10-27 04:15 AEDT) → scored once (`130-score#1`, 04:15–04:25 AEDT) → published `b4b0df9a` / `f3be7485` →
`--validate` **exit 0** (10 records hash-checked, C1-car seed 20260908 re-scored with exact row equality, publication
re-derived byte-identically). Doc: `docs/issue-130-dino-wm.md`.

**Outcome.**
- **C1-dino** (primary, program set) stopped before any predictor training with the typed reason
  `readiness_ceiling_below_threshold`: the R1 readout of true-frame DINOv2 ViT-S/14 patch features reaches a policy
  grid ceiling of **0.5948** against the frozen 0.65 (the same ceiling on the true E99 carriers is 0.7223). It was not
  trained or scored. Every token involving it is `readiness_or_precision_insufficient`, and it **vetoes the #127
  program-level claim** (dropped).
- **C1-car** (secondary): every global guard passes, and G-port and G-fid pass. C1-car fails **G-action on every
  seed** (0.00044 / 0.00045 / 0.00163 against the frozen 0.05); G-rank, G-nonfinite and G-trained pass. By the frozen
  rule every C1-car token is `readiness_or_precision_insufficient`: **disposition 4**, no scoped ranking claim in
  either direction.
- **C1-dino − C1-car cannot be computed** (C1-dino not run), so Q3's input-feature question is not answered here.

**Gates.**
- **G-fid Tier E: PASS.** The released PushT checkpoint with the official `plan.py` (`plan_pusht.yaml`, 50 cases,
  MPC-CEM, `max_iter` 12 frozen in `band.json`) gives success **0.92 (46 / 50)**, inside the band [0.82, 0.98] frozen
  before the run (paper 0.90).
- **G-port: PASS.** The official `VWorldModel` + `Trainer.init_optimizers` (PushT configuration, DINOv2 in the loop)
  against the adapter: losses equal, gradient max relative L2 3.7e-7, parameters 99.996 % within 1e-7 after two
  steps.

**Grid (311 mixed cells × 3 seeds, tie-free cost).**

| ranker | pooled AUC [CI] |
| --- | --- |
| U | 0.7444 [0.7269, 0.7620] |
| C\* | 0.7834 [0.7676, 0.7992] |
| P | 0.7203 [0.6965, 0.7442] |
| J_dec | 0.7460 [0.7277, 0.7637] |
| **C1-car** (car-release, 1 243 648 params) | 0.5251 [0.5093, 0.5411] |
| C1-dino | not run (typed stop) |

**Contrasts.**

| contrast | estimate [CI] | token / reverse |
| --- | --- | --- |
| C1-car − U | −0.2193 [−0.2412, −0.1985] | stop / stop (guard) |
| C1-car − C\* | −0.2582 [−0.2790, −0.2385] | stop / stop |
| C1-car − P | −0.1952 [−0.2256, −0.1648] | stop / stop |
| C1-car − J_dec | −0.2208 [−0.2424, −0.1998] | stop / stop |
| C1-dino − C1-car (within family) | n/a (not computable) | stop / stop |
| wrong anchor C1-car (right − wrong) | −0.0101 [−0.0259, +0.0053] | stop |

"stop" = `readiness_or_precision_insufficient`. The estimates are reported as measured and license no statement in
either direction.

**Selection** (policy split, 94 mixed cells, seed 20260908): car-lr5e-4 (released lr, 100 epochs) 0.4811, car-lr5e-5
(paper lr, 100 epochs) 0.5189, **car-release** (the released checkpoint's sample budget, 123 858 steps) **0.5409**,
selected. With no LR scheduler, car-release is the first 6.7 % of car-lr5e-4's run.

**G-action calibration context** (#127 decision; descriptive; changes nothing). In carrier space the engine truth
reaches 0.0395, so the frozen 0.05 is not attainable by any model there. On that scale C1-car's endpoints move with the
launch about **2 %** as much as the true endpoints (1.1 % / 1.1 % / 4.1 % per seed; B1 33 %, U 58 %, C\* 85 %). The
frozen guard and the disposition stand.

**Also reported.**
- **Position error at frame 225** (descriptive): C1-car 0.0578 [0.0533, 0.0625] vs U 0.0474 [0.0440, 0.0508].
- **Descriptive inventories:** C1-car offset 0.4798, angle 0.4842, power 0.5104 (U 0.6670 / 0.6791 / 0.8058).
- **P-novelty:** not scored (C1-car fails G-action; C1-dino typed stop). The U / C\* / P references are computed and
  hashed.

**Tier-E infrastructure crashes (disclosed; no success value existed or was read before the deciding run).** One
scheduling stop before the first MPC iteration, then three CUDA OOMs:
1. an orphaned #128 process on GPU 2;
2. the plot-only VQ-VAE decode of the evaluator's rollout videos beside a co-scheduled job;
3. the same plot path on the exclusive card.

Each was relaunched under P-crash from the resumable MPC-iteration checkpoint. Post-freeze amendments of
`scripts/b127_dino_wm_fidelity.py`, infrastructure only and registered: **A1** (`009a87cc`, decode one case at a time)
and **A2** (`4048e33b`, the official evaluator's no-decoder branch, so no videos are drawn; planning, objective, success
metric and RNG unchanged). The deciding run resumed from iteration 2 and completed all 12 iterations.

**Compute.** About 38.1 process GPU-h (38.02 recorded + 0.08 unrecorded) against the C1 share of 88. About 10.85 GPU-h
of that is infrastructure loss from the stopped and crashed Tier-E launches.

| | C1-car | U |
| --- | --- | --- |
| trainable parameters | 1 243 648 | 1 837 690 |
| training GPU-s per seed (process) | 2 555 / 1 701 / 1 703 | mean 1 294 |
| MACs per candidate (to frame 225) | 4.48 G | 20.99 M |
| wall per decision | 0.074 s | — |

The budget safeguard does not fire: C1-car − U is not `supported`, and C1-car's training compute is 1.53 × U's. Engine
seconds: 0.

**Disclosures.**
- C1-car had the #127 tuning allowance (3 configurations on the policy split; U had none).
- The paper's predictor lr (5e-5) differs from the released config and checkpoint (5e-4); both were candidates.
- C1-dino used a full-frame, aspect-preserving resize at native scale and a slot-query attention R1; both were posted
  before the readiness smoke.
- Training windows start at frame-skip phase 0 only. There is no proprio channel and no decoder.
- The official stack ran in env `b127-dino_wm` (torch 2.7.1) with import-only shims (P-route).

**Links.**
- Decisions: [band](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382394238),
  [license](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382422982),
  [adapter](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382576605),
  [C1-dino stop](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6383216316),
  [resumable Tier-E](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6383358026),
  [selection](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6386706483),
  [novelty](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6386831073),
  [G-action](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6396584714),
  P-crash [1](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6386786676) /
  [2](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6400401642) /
  [3](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6402614864),
  [A1](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6400429696).
- Artefacts: `.local-artifacts/issue-130-dino-wm-v1/` (`summary.json`, `findings.md`, `compute.json`,
  `fidelity/gfid.json`, `readiness/readiness.json`).
