## Result (2026-10-25): B1 SlotFormer-E99 — disposition 4, per-contrast tokens only

Frozen plan `bd5c5577` (2026-10-24T20:06:47Z) → readouts frozen 2026-10-25T08:55:28Z → scored once
(`129-score#1`, 20:00–20:07 AEDT) → published `238ad763` → `--validate` **exit 0** (22 records hash-checked, B1 seed
20260908 re-scored with exact row equality, publication re-derived byte-identically). Doc:
`docs/issue-129-slotformer.md`.

**Outcome.** Every global guard passes, and G-port and G-fid pass. B1 (primary, program set) and B1-m (matched) fail
**G-action on every seed**: endpoint action sensitivity 0.0122–0.0138 and 0.00007–0.00026 against the frozen 0.05. By
the frozen rule every token involving them is `readiness_or_precision_insufficient`. **Disposition 4**: no scoped
ranking claim in either direction. B1 vetoes the #127 program-level claim (guard failed).

**Gates.**
- **G-fid Tier E: PASS.** CLEVRER mIoU 49.42 with the released weights and the official `test_vp.py`, inside the band
  [48.92, 49.92] frozen before the run.
- **G-port: PASS** under #123's exact rule. The official sources run on the adapter's torch 2.13, six arms, all deltas 0.

**Grid (311 mixed cells × 3 seeds, tie-free cost).**

| ranker | pooled AUC [CI] |
| --- | --- |
| U | 0.7444 [0.7269, 0.7620] |
| C\* | 0.7834 [0.7676, 0.7992] |
| P | 0.7203 [0.6965, 0.7442] |
| J_dec | 0.7460 [0.7277, 0.7637] |
| **B1** (C3-s15-long, 6 325 005 params) | 0.5476 [0.5264, 0.5692] |
| B1-m (1 793 445 params, 9 000 × 64) | 0.5348 [0.5176, 0.5523] |
| B1-head (exploratory) | 0.6069 [0.5813, 0.6328] |
| U+head (exploratory control) | 0.8123 [0.7948, 0.8303] |

**Contrasts.**

| contrast | estimate [CI] | token / reverse |
| --- | --- | --- |
| B1 − U | −0.1968 [−0.2228, −0.1705] | stop / stop (guard) |
| B1 − C\* | −0.2358 [−0.2614, −0.2098] | stop / stop |
| B1 − P | −0.1727 [−0.1973, −0.1479] | stop / stop |
| B1 − J_dec | −0.1984 [−0.2254, −0.1708] | stop / stop |
| B1-m − U | −0.2096 [−0.2303, −0.1899] | stop / stop |
| B1 − B1-m | +0.0128 [−0.0196, +0.0447] | stop / stop |
| B1-head − B1 | +0.0593 [+0.0227, +0.0965] | stop / stop |
| **U+head − U** (exploratory, global guards only) | **+0.0679 [+0.0583, +0.0773]** | **`supported`** / `not_supported_by_this_experiment` |
| wrong anchor B1 (right − wrong) | −0.0065 [−0.0239, +0.0113] | stop |

"stop" = `readiness_or_precision_insufficient`. U+head is SlotFormer's own PHYRE success head trained on U's predicted
fit endpoints; it is exploratory and enters no disposition.

**G-action calibration context** (descriptive, fit split only, measured after the score; changes nothing). With the
same `eb.action_sensitivity` call, the engine truth reaches **0.0395** in carrier space: the true carriers of the 16
executed grid shots of 479 fit levels. The frozen 0.05, set on #123's latent states, is therefore not attainable in
carrier space even by a perfect model. On the same scale: U 0.0229, C\* 0.0334, B1 0.0130, B1-m 0.00018. The frozen
guard and the disposition stand.

**Also reported.**
- **Position error at frame 225** (descriptive): B1 0.0508 [0.0462, 0.0555] vs U 0.0474 [0.0440, 0.0508]. As a point
  estimate this meets the #127 DDLP Tier-C trigger condition ("exceeds U's"); the intervals overlap. Funding is for
  P-budget.
- **CLEVRER / Physion-Dominoes** (exploratory, in-sample, no tokens): B1-bench position_mse is lower than #117 J's at
  every horizon on Physion-Dominoes and at t = 120 on CLEVRER, and higher at t ≤ 60 on CLEVRER.
- **P-novelty:** not scored, because both arms fail G-action. The references are computed and hashed.

**Compute.** About 14.95 process GPU-h (14.74 recorded + 0.21 unrecorded) against the child cap of 88. About 9.3 GPU-h
device-equivalent, because B1 seed 20260908 ran 4.4× slower on a shared GPU [estimate].

| | B1 | B1-m | U |
| --- | --- | --- | --- |
| training GPU-s per seed (process) | 26 538 / 6 490 / 6 050 | 1 190 / 1 492 / 1 184 | mean 1 294 |
| MACs per candidate | 5.69 M | 3.02 M | 20.99 M |
| wall per decision | 0.046 s | 0.038 s | — |

The budget safeguard does not fire: B1 − U is not `supported`, and B1's training compute is 10.1 × U's. Engine
seconds: 0.

**Disclosures.**
- Selection relaunched once after a co-scheduling OOM
  ([P-crash](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382755036)).
- G-port: the criterion was corrected
  ([correction](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6383196400)), then the reference was
  moved to the same torch before the freeze
  ([follow-up](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6383281200)).
- B1 had the #127 tuning allowance (3 configurations on the policy split; U had none).
- Training used fp16 / `GradScaler` as the official `--fp16`; scoring ran under `deterministic_scoring()`.

**Links.**
- Decisions: [license](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382274253),
  [adapter](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382359850),
  [route](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382402424),
  [band](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382402612),
  [selection](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382590719),
  [novelty](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6386830857).
- Artefacts: `.local-artifacts/issue-129-slotformer-v1/`
  (`summary.json`, `findings.md`, `compute.json`, `analysis/g-action-calibration.json`).
