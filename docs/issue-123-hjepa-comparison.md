# Issue #123: H-JEPA comparison on NovPhy (reproduction, guards, and the frozen disposition)

Parent: #108. Ticket: [#123](https://github.com/Sino-Huang/NovPhy/issues/123). The binding shared rules of #85 apply.
The Gate-B (#108), #101, #104, #109, #112, #113 and #117 records are read-only inputs. Nothing here edits the ICLR 2026
submission.

| | |
| --- | --- |
| Paper and code | Zhang, Terver, Rabbat, LeCun, Balestriero, *H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning*, arXiv 2610.06805; [kevinghst/H-JEPA](https://github.com/kevinghst/H-JEPA) @ `97054ec` (MIT) |
| Runner | `scripts/run_hjepa_comparison.py` (data / G-port / selection / smoke / freeze / train / readouts / score / publish / validate) |
| Modules | `world_model/hjepa/novphy.py` (NovPhy adapter); `world_model/hjepa/vendor/` (official code vendored at `97054ec`, per-file upstream and vendored sha256 in `VENDOR.json`; only the top-level imports are made package-relative) |
| Fidelity | `scripts/run_hjepa_fidelity.py` (Tier E / Tier T queues, `--table`), `scripts/hjepa_train_ckpt.py` (Tier-T launcher), `scripts/hjepa_port_reference.py` (G-port reference) |
| Tests | `tests/test_issue_123_hjepa_comparison.py` (18 tests), `tests/test_issue_123_training_resume.py` (3), `tests/test_gpu_window.py` (scheduler) |
| Artifacts | `.local-artifacts/issue-123-hjepa-comparison-v1/`: `plan.json`, `smoke.json`, `decisions.md`, `findings.md` (every table), `summary.json`, `compute.json`, `records-manifest.json`, `validation.json`, `readouts-manifest.json`, `fidelity/gfid.{json,md}`, `fidelity/ledger.jsonl`, `gport/gport.json`, `gport/a2/gport.json`, `ledger/`, `logs/` (including copies of the scheduler's #123 run logs, `logs/scheduler-123-{score,publish,validate}.log`, and events, `logs/scheduler-123-events.jsonl`). Checkpoints, pixel stores, carriers, the evaluation cache, the 51 records, `fidelity/logs/` and the fidelity failure archives (`fidelity/infrastructure-failures/`, `fidelity/setup-failures/`) stay local; records and readouts are hashed in `records-manifest.json` and `readouts-manifest.json` |
| Freeze | `plan.json` frozen 2026-10-18T12:24:06Z, commit `b06e7397`, after a passed smoke and before any main-arm training or evaluation-split read. Post-freeze amendments A1 (`73298595`) and A2 (`c9b8ad6c`), registered in `scripts/post_freeze_amendments.py` |
| Readout freeze | `readouts-manifest.json` frozen 2026-10-20T19:20:16Z (06:20 AEDT, 21 Oct; sha256 `853c114e…`), before any evaluation anchor was encoded |
| Scoring | once, in a fresh process: scheduler run `123-score#2`, 2026-10-21 18:00:08–18:10:04 AEDT, exit 0 (33 arm-seeds) |
| Validation | `python -u -m scripts.run_hjepa_comparison --validate`, scheduler run `123-validate#1`, 2026-10-21 18:15:06 AEDT, **exit 0**: 51 records hash-checked, every table re-derived and byte-compared, HJ-car-m seed 20260908 re-scored on 311 grid cells, max delta 0.0 (tie-free, pig presence, pig displacement). `validation.json` binds `records-manifest.json` sha256 `e67af31c…f2f988` |

Engine time is zero seconds: no conditional arm ran. The primary metric is the Gate-B pooled grid AUC over the 311 mixed
grid cells × 3 seeds of the #104 sealed #109 evaluation split, tie-free cost, under `deterministic_scoring()`. Intervals
are level-clustered percentile bootstraps (10 000 draws, PCG64 7201) and are DESCRIPTIVE. The margin is δ = 0.02.

## Outcome

**Disposition 3: per-contrast tokens only. Branch: no scoped ranking claim.** Neither conditional arm is triggered
(U-B3: disposition 5 is false; closed loop: no PC1 / PC2 `supported`).

H-JEPA, reproduced faithfully (G-fid pass), collapses on action-sparse NovPhy: its predicted endpoint ignores the launch
and ranks at chance; the frozen rule therefore makes no ranking claim.

- Every global guard passes: G-fid, G-port, G-isolation, G-chronology, G-scoring-policy and G-records.
- Every H-JEPA, LeWM and HWM arm fails G-collapse and/or G-action, and seven of the eleven arms also fail G-readout. By the
  frozen rule, every token involving these arms, primary or secondary, is `readiness_or_precision_insufficient`.
- Disposition 2 ("our Gate-B model ranks better than H-JEPA (scoped)") requires both primary H-JEPA arms to pass every
  guard. None does, so disposition 2 does not apply. This document makes no "better than H-JEPA" or state-of-the-art
  statement.

## Design summary

The design is frozen in the ticket and `plan.json`; the implementer decisions are in `decisions.md` (posted on #123 before
the freeze).

**Stage 0, fidelity.** The official code reproduces the paper's FourRoom Distractors numbers on our stack: Tier E evaluates
the released LeWM / H-JEPA-2 / H-JEPA-3 checkpoints (seeds 42/43/44, 50 tasks); Tier T retrains them with
`main_hjepa.py`. G-fid passes iff both tiers pass their frozen bands. G-port checks that the vendored modules plus our
loader, optimizer and scheduler reproduce the official forward pass and optimizer steps on fixed inputs.

**Arms.** Every arm trains on the #109 fit split only, seeds 20260908/09/10, with SIGReg λ = 0.18 for both input types
(selected on the policy split before the freeze). Layout H3-Δ: three levels at 1, 5 and 15 carrier frames, so the
225-frame endpoint is 15 top-level steps. The launch is the level-1 action at t = 0; the anchor-mixture sampler starts
50 % of clips at the launch frame.

| arm | input | model | budget | role | parameters |
| --- | --- | --- | --- | --- | --- |
| **HJ-px** | pixels 320 × 240, patch 16 | H-JEPA H3-Δ, paper sizes | B3 (128 045 × 128) | primary, faithful | 35 219 796 |
| **HJ-car-m** | E99 carriers | H-JEPA H3-Δ, MLP level-1 encoder, matched | B1 (9000 × 64) | primary, controlled | 1 855 172 |
| HJ-px-B1 | pixels | as HJ-px | B1 | budget-matched faithful | 35 219 796 |
| LeWM-px / HWM-px | pixels | flat LeWM / HWM | B3 | temporal / representational lever | 11 350 204 / 34 922 964 |
| LeWM-car-m / HWM-car-m | carriers | matched | B1 | levers on carriers | 1 834 220 / 1 840 100 |
| HJ-car-native | carriers | paper-sized predictors | B3 | native architecture on our input | 30 538 900 |
| HJ-car-m-uniform / -persist / -fs5 | carriers | uniform sampler / launch at every step / 2 levels, frameskip 5 | B1 | ablations | 1 855 172 / 1 855 172 / 1 835 440 |

Our arms need no training: **U** (Gate-B unit, LC : F-15-macro, hybrid, E99), **C\*** (LC : C-F-15), **J** (ICLR) and
**J_dec** (#101), and **P** (the no-model action prior). U, C\*, J and J_dec rows are their published records; P is the
Gate-B prior rule.

**Readouts.** Every readout is fitted on fit-split frames only, then frozen and hashed before any evaluation anchor is
encoded. R1 (primary): a 2-layer MLP probe from frozen top-level encodings of real fit frames to their E99 carriers;
cost = tie-free(R1(z₀), R1(ẑ at the endpoint)). R2 (secondary): the same probe fitted on predicted endpoint latents. R3
(exploratory): goal matching to a success prototype. R4 (exploratory): R2 plus a ranking loss on fit mixed cells.

**Contrasts.** PC1 = HJ-px − U and PC2 = HJ-car-m − U (primary); S1–S4 the temporal and representational levers; S5 / S6
against C\* and P; S7 HJ-px-B1 − U; S8 HJ-car-native − U; S9 the R2 versions of PC1 / PC2; S10 the wrong-anchor control;
S11 against J and J_dec. Each contrast is tokenised in both directions with the δ rule: `supported` iff mean ≥ δ and
lower bound > 0; `not_supported_by_this_experiment` iff upper bound < δ; `readiness_or_precision_insufficient` otherwise.

**Guards.** G-fid, G-port, G-isolation, G-chronology, G-scoring-policy and G-records are global. Per arm:
G-collapse (per seed: top-level effective rank ≥ 0.1·D and within-episode variance share χ ≥ 0.05), G-action (endpoint
action sensitivity ≥ 0.05 on 480 fit anchors × 16 grid launches), G-readout (R1 on the *true* frame-225 encodings reaches
policy-split grid AUC ≥ 0.65), G-nonfinite (≤ 10 % nonfinite R1 and R2 costs) and G-trained. A failed guard turns every
token involving the arm into `readiness_or_precision_insufficient`.

**Dispositions** (frozen before any evaluation score): 1, H-JEPA ranks better (scoped), needs a `supported` PC1 or PC2
in the H-JEPA − U direction, every guard of the arm passing and S10 `supported`; 2, our Gate-B model ranks better
(scoped), needs U − HJ-px and U − HJ-car-m both `supported` and both arms passing every guard; 3, otherwise per-contrast
tokens only; 4, S5 / S11 are reported beside every primary contrast; 5, budget asymmetry, which triggers U-B3.

## G-fid: the paper's FourRoom numbers are reproduced

Official code `97054ec`; success % on the 50 released FourRoom Distractors tasks; seeds 42/43/44. Record:
`fidelity/gfid.{json,md}` (gfid.json sha256 `4dfa74d3…`).

**Tier E (released checkpoints): PASS.** H-JEPA-3 − LeWM per seed +60 / +42 / +66 pp (mean +56.0); ordering holds.

| model | seed 42 | seed 43 | seed 44 | mean ± SE | paper | band | in band |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lewm | 40.0 | 50.0 | 30.0 | 40.0 ± 5.8 | 40.7 ± 6.8 | [27.1, 54.3] | True |
| hjepa_l2 | 86.0 | 72.0 | 92.0 | 83.3 ± 5.9 | 81.3 ± 6.7 | [67.9, 94.7] | True |
| hjepa_l3 | 100.0 | 92.0 | 96.0 | 96.0 ± 2.3 | 96.0 ± 1.1 | [92.8, 99.2] | True |

**Tier T (retrained with `main_hjepa.py`): PASS.** H-JEPA-3 − LeWM per seed +60 / +76 / +40 pp (mean +58.7), every seed
positive.

| model | seed 42 | seed 43 | seed 44 | mean ± SE | paper | band | in band |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lewm | 32.0 | 20.0 | 58.0 | 36.7 ± 11.2 | 40.7 ± 6.8 | [14.5, 66.9] | True |
| hjepa_l2 | 84.0 | 88.0 | 74.0 | 82.0 ± 4.2 | 81.3 ± 6.7 | [65.5, 97.1] | True |
| hjepa_l3 | 92.0 | 96.0 | 98.0 | 95.3 ± 1.8 | 96.0 ± 1.1 | [91.8, 100.2] | True |

**G-fid: PASS.** **G-port: PASS** on HJ-px, HWM-px, LeWM-px, HJ-car-m and HJ-car-m-fs5, both the frozen record
(`gport/gport.json`, `3be3bfea…`) and the A2 positive-LR record (`gport/a2/gport.json`, `c2b52c9d…`).

## Results

### Guards

| global guard | pass |
| --- | --- |
| G-fid | True |
| G-port | True |
| G-isolation | True |
| G-chronology | True |
| G-scoring-policy | True |
| G-records | True |

Arm guards (G-readout uses the arm's 3-seed mean policy ceiling):

| arm | role | params | G-collapse | G-action | G-readout (policy ceiling) | G-nonfinite (R1 / R2 share) | trained | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HJ-car-m | primary, controlled | 1,855,172 | False | False | False (0.5795) | True (0.000 / 0.000) | True | False |
| HJ-car-m-fs5 | ablation: 2 levels, level-1 frameskip 5 (no Delta-1 level) | 1,835,440 | False | False | False (0.5842) | True (0.000 / 0.000) | True | False |
| HJ-car-m-persist | ablation: launch at every step | 1,855,172 | False | False | False (0.5773) | True (0.000 / 0.000) | True | False |
| HJ-car-m-uniform | ablation: H-JEPA's uniform sampler | 1,855,172 | False | False | False (0.5795) | True (0.000 / 0.000) | True | False |
| HJ-car-native | native architecture on our input | 30,538,900 | False | True | True (0.7182) | True (0.000 / 0.000) | True | False |
| HJ-px | primary, faithful | 35,219,796 | False | False | True (0.8812) | True (0.000 / 0.000) | True | False |
| HJ-px-B1 | budget-matched faithful | 35,219,796 | False | False | True (0.8077) | True (0.000 / 0.000) | True | False |
| HWM-car-m | representational lever (carriers) | 1,840,100 | False | False | False (0.6092) | True (0.000 / 0.000) | True | False |
| HWM-px | representational lever (pixels) | 34,922,964 | False | False | True (0.8642) | True (0.000 / 0.000) | True | False |
| LeWM-car-m | temporal lever (carriers) | 1,834,220 | False | False | False (0.5738) | True (0.000 / 0.000) | True | False |
| LeWM-px | temporal lever (pixels) | 11,350,204 | True | False | False (0.5587) | True (0.000 / 0.000) | True | False |

Failing arm guards: HJ-px, HJ-px-B1, HWM-px: G-collapse, G-action. HJ-car-m, HJ-car-m-fs5, -persist, -uniform,
LeWM-car-m, HWM-car-m: G-collapse, G-action, G-readout. HJ-car-native: G-collapse. LeWM-px: G-action, G-readout.

Per-seed diagnostics (top level; G-collapse needs both the rank and the χ threshold; the effective rank passes on every
seed, so every G-collapse failure is a χ failure):

| arm | seed | erank / D (≥ 0.1·D) | χ within-episode share (≥ 0.05) | endpoint action sensitivity (≥ 0.05) | G-collapse | G-action | policy R1 AUC | policy ceiling AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HJ-px | 20260908 | 111.7 / 192 | 0.0014 | 1.14e-06 | fail | fail | 0.5483 | 0.8972 |
| HJ-px | 20260909 | 112.7 / 192 | 0.0015 | 1.14e-06 | fail | fail | 0.5674 | 0.8751 |
| HJ-px | 20260910 | 113.0 / 192 | 0.0015 | 1.14e-06 | fail | fail | 0.5145 | 0.8713 |
| HJ-car-m | 20260908 | 14.2 / 112 | 0.0167 | 1.71e-07 | fail | fail | 0.5053 | 0.5708 |
| HJ-car-m | 20260909 | 13.8 / 112 | 0.0167 | 1.66e-07 | fail | fail | 0.4731 | 0.6209 |
| HJ-car-m | 20260910 | 14.1 / 112 | 0.0161 | 1.67e-07 | fail | fail | 0.4881 | 0.5468 |
| HJ-px-B1 | 20260908 | 44.2 / 192 | 0.0021 | 4.90e-07 | fail | fail | 0.5179 | 0.7968 |
| HJ-px-B1 | 20260909 | 42.8 / 192 | 0.0022 | 4.81e-07 | fail | fail | 0.4960 | 0.8201 |
| HJ-px-B1 | 20260910 | 44.3 / 192 | 0.0019 | 4.77e-07 | fail | fail | 0.5229 | 0.8062 |
| HJ-car-native | 20260908 | 100.8 / 192 | 0.0113 | 1.61e-01 | fail | pass | 0.5517 | 0.7241 |
| HJ-car-native | 20260909 | 101.5 / 192 | 0.0114 | 1.57e-01 | fail | pass | 0.5134 | 0.6887 |
| HJ-car-native | 20260910 | 101.3 / 192 | 0.0112 | 1.45e-01 | fail | pass | 0.5501 | 0.7419 |
| HJ-car-m-fs5 | 20260908 | 18.8 / 112 | 0.0194 | 1.96e-07 | fail | fail | 0.4838 | 0.5779 |
| HJ-car-m-fs5 | 20260909 | 18.9 / 112 | 0.0202 | 1.99e-07 | fail | fail | 0.5047 | 0.5594 |
| HJ-car-m-fs5 | 20260910 | 18.3 / 112 | 0.0199 | 2.12e-07 | fail | fail | 0.4740 | 0.6152 |
| HJ-car-m-persist | 20260908 | 14.2 / 112 | 0.0167 | 2.77e-07 | fail | fail | 0.5520 | 0.5477 |
| HJ-car-m-persist | 20260909 | 13.8 / 112 | 0.0167 | 2.74e-07 | fail | fail | 0.5382 | 0.6347 |
| HJ-car-m-persist | 20260910 | 14.1 / 112 | 0.0161 | 2.91e-07 | fail | fail | 0.4653 | 0.5496 |
| HJ-car-m-uniform | 20260908 | 14.6 / 112 | 0.0225 | 1.56e-07 | fail | fail | 0.4978 | 0.6065 |
| HJ-car-m-uniform | 20260909 | 14.3 / 112 | 0.0291 | 1.55e-07 | fail | fail | 0.5136 | 0.6001 |
| HJ-car-m-uniform | 20260910 | 14.4 / 112 | 0.0221 | 1.57e-07 | fail | fail | 0.5123 | 0.5318 |
| LeWM-px | 20260908 | 170.5 / 192 | 0.0557 | 3.37e-02 | pass | fail | 0.4896 | 0.5878 |
| LeWM-px | 20260909 | 168.1 / 192 | 0.0515 | 5.19e-02 | pass | pass | 0.4727 | 0.5141 |
| LeWM-px | 20260910 | 169.1 / 192 | 0.0511 | 4.75e-02 | pass | fail | 0.5238 | 0.5743 |
| LeWM-car-m | 20260908 | 27.0 / 112 | 0.0468 | 4.95e-03 | fail | fail | 0.4890 | 0.5715 |
| LeWM-car-m | 20260909 | 27.0 / 112 | 0.0474 | 1.27e-02 | fail | fail | 0.4757 | 0.5824 |
| LeWM-car-m | 20260910 | 26.7 / 112 | 0.0499 | 1.38e-02 | fail | fail | 0.4478 | 0.5675 |
| HWM-px | 20260908 | 127.6 / 192 | 0.0006 | 1.38e-06 | fail | fail | 0.5521 | 0.8719 |
| HWM-px | 20260909 | 129.4 / 192 | 0.0007 | 4.22e-06 | fail | fail | 0.5157 | 0.8308 |
| HWM-px | 20260910 | 125.2 / 192 | 0.0005 | 1.77e-06 | fail | fail | 0.5209 | 0.8900 |
| HWM-car-m | 20260908 | 24.0 / 112 | 0.0253 | 1.77e-07 | fail | fail | 0.5174 | 0.5758 |
| HWM-car-m | 20260909 | 24.0 / 112 | 0.0261 | 1.66e-07 | fail | fail | 0.5087 | 0.6385 |
| HWM-car-m | 20260910 | 23.6 / 112 | 0.0255 | 1.77e-07 | fail | fail | 0.4963 | 0.6131 |

### Grid rankers (311 mixed cells × 3 seeds, R1)

| ranker | pooled AUC | seed 20260908 | seed 20260909 | seed 20260910 | top-1 | chance top-1 |
| --- | --- | --- | --- | --- | --- | --- |
| U | 0.7444 [0.7269, 0.7620] | 0.7141 [0.6917, 0.7362] | 0.8011 [0.7826, 0.8199] | 0.7181 [0.6961, 0.7404] | 0.195 | 0.089 |
| C* | 0.7834 [0.7676, 0.7992] | 0.8262 [0.8091, 0.8431] | 0.7334 [0.7127, 0.7536] | 0.7905 [0.7688, 0.8120] | 0.192 | 0.089 |
| P | 0.7203 [0.6965, 0.7442] | seed-free | |  | 0.180 | 0.089 |
| J | 0.7057 [0.6846, 0.7267] | 0.6690 [0.6449, 0.6931] | 0.7012 [0.6787, 0.7234] | 0.7470 [0.7261, 0.7678] | 0.160 | 0.089 |
| J_dec | 0.7460 [0.7277, 0.7637] | 0.7205 [0.6989, 0.7417] | 0.7933 [0.7739, 0.8124] | 0.7242 [0.7035, 0.7451] | 0.180 | 0.089 |
| HJ-px | 0.4997 [0.4827, 0.5165] | 0.4874 [0.4575, 0.5169] | 0.4995 [0.4681, 0.5308] | 0.5121 [0.4820, 0.5422] | 0.093 | 0.089 |
| HJ-car-m | 0.4896 [0.4694, 0.5099] | 0.4982 [0.4660, 0.5298] | 0.4856 [0.4552, 0.5157] | 0.4852 [0.4536, 0.5172] | 0.079 | 0.089 |
| HJ-px-B1 | 0.4753 [0.4575, 0.4935] | 0.4950 [0.4640, 0.5271] | 0.4750 [0.4446, 0.5056] | 0.4558 [0.4266, 0.4858] | 0.071 | 0.089 |
| HJ-car-native | 0.5407 [0.5185, 0.5631] | 0.5481 [0.5145, 0.5827] | 0.5283 [0.4908, 0.5651] | 0.5457 [0.5086, 0.5828] | 0.122 | 0.089 |
| HJ-car-m-fs5 | 0.4926 [0.4747, 0.5108] | 0.5077 [0.4777, 0.5374] | 0.4668 [0.4344, 0.4988] | 0.5032 [0.4730, 0.5332] | 0.090 | 0.089 |
| HJ-car-m-persist | 0.5387 [0.5208, 0.5566] | 0.5385 [0.5080, 0.5688] | 0.5644 [0.5319, 0.5973] | 0.5133 [0.4818, 0.5445] | 0.111 | 0.089 |
| HJ-car-m-uniform | 0.4934 [0.4764, 0.5111] | 0.4969 [0.4686, 0.5260] | 0.4805 [0.4494, 0.5121] | 0.5028 [0.4716, 0.5348] | 0.072 | 0.089 |
| LeWM-px | 0.5090 [0.4924, 0.5257] | 0.5219 [0.4930, 0.5497] | 0.5067 [0.4787, 0.5355] | 0.4984 [0.4692, 0.5276] | 0.020 | 0.089 |
| LeWM-car-m | 0.4945 [0.4781, 0.5112] | 0.4995 [0.4703, 0.5292] | 0.4898 [0.4636, 0.5165] | 0.4943 [0.4636, 0.5248] | 0.054 | 0.089 |
| HWM-px | 0.5001 [0.4835, 0.5165] | 0.4868 [0.4547, 0.5187] | 0.5097 [0.4801, 0.5384] | 0.5039 [0.4741, 0.5335] | 0.079 | 0.089 |
| HWM-car-m | 0.5033 [0.4852, 0.5217] | 0.5026 [0.4709, 0.5358] | 0.5032 [0.4727, 0.5339] | 0.5041 [0.4727, 0.5348] | 0.091 | 0.089 |

U 0.7444, C\* 0.7834, P 0.7203, J 0.7057, J_dec 0.7460. HJ-px 0.4997 and HJ-car-m 0.4896; every other H-JEPA / LeWM /
HWM arm lies in 0.4753–0.5407 under R1. The H-JEPA-family top-1 rates are within 0.004 of chance (0.089) or below it,
except HJ-car-native 0.122 and HJ-car-m-persist 0.111.

### Contrasts (grid, paired per (level, seed))

| contrast | first − second | estimate (DESCRIPTIVE) | token (first − second) | token (second − first) | failing arm guards |
| --- | --- | --- | --- | --- | --- |
| PC1 | HJ-px − U | -0.2447 [-0.2688, -0.2204] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` | HJ-px: G-collapse, G-action |
| PC2 | HJ-car-m − U | -0.2548 [-0.2829, -0.2273] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` | HJ-car-m: G-collapse, G-action, G-readout |
| S1_pixels | HJ-px − LeWM-px | -0.0094 [-0.0324, +0.0142] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` | HJ-px: G-collapse, G-action; LeWM-px: G-action, G-readout |
| S2_carriers | HJ-car-m − LeWM-car-m | -0.0049 [-0.0306, +0.0202] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` | HJ-car-m: G-collapse, G-action, G-readout; LeWM-car-m: G-collapse, G-action, G-readout |
| S3_pixels | HJ-px − HWM-px | -0.0004 [-0.0222, +0.0223] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` | HJ-px: G-collapse, G-action; HWM-px: G-collapse, G-action |
| S4_carriers | HJ-car-m − HWM-car-m | -0.0137 [-0.0418, +0.0142] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` | HJ-car-m: G-collapse, G-action, G-readout; HWM-car-m: G-collapse, G-action, G-readout |
| S5_pixels | HJ-px − C* | -0.2837 [-0.3068, -0.2605] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` | HJ-px: G-collapse, G-action |
| S5_carriers | HJ-car-m − C* | -0.2937 [-0.3206, -0.2675] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` | HJ-car-m: G-collapse, G-action, G-readout |
| S6_pixels | HJ-px − P | -0.2206 [-0.2502, -0.1912] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` | HJ-px: G-collapse, G-action |
| S6_carriers | HJ-car-m − P | -0.2307 [-0.2602, -0.2016] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` | HJ-car-m: G-collapse, G-action, G-readout |
| S7 | HJ-px-B1 − U | -0.2691 [-0.2944, -0.2441] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` | HJ-px-B1: G-collapse, G-action |
| S8 | HJ-car-native − U | -0.2037 [-0.2324, -0.1765] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` | HJ-car-native: G-collapse |
| S9_PC1_R2 | HJ-px:R2 − U | -0.2423 [-0.2656, -0.2186] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` | HJ-px: G-collapse, G-action |
| S9_PC2_R2 | HJ-car-m:R2 − U | -0.2379 [-0.2593, -0.2170] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` | HJ-car-m: G-collapse, G-action, G-readout |
| S11_pixels_J | HJ-px − J | -0.2061 [-0.2325, -0.1792] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` | HJ-px: G-collapse, G-action |
| S11_pixels_J_dec | HJ-px − J_dec | -0.2463 [-0.2707, -0.2214] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` | HJ-px: G-collapse, G-action |
| S11_carriers_J | HJ-car-m − J | -0.2161 [-0.2465, -0.1870] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` | HJ-car-m: G-collapse, G-action, G-readout |
| S11_carriers_J_dec | HJ-car-m − J_dec | -0.2563 [-0.2847, -0.2287] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` | HJ-car-m: G-collapse, G-action, G-readout |

S5 and S11 (the unit is not our best ranker; C\* 0.783 vs U 0.744 at Gate B) are reported beside the primary contrasts as
disposition 4 requires.

### S10: wrong-anchor control (right − wrong, 480 fit anchors)

| arm | right | wrong | right - wrong | token |
| --- | --- | --- | --- | --- |
| HJ-px | 0.4997 [0.4827, 0.5165] | 0.5015 [0.5002, 0.5028] | -0.0018 [-0.0190, +0.0151] | readiness_or_precision_insufficient |
| HJ-car-m | 0.4896 [0.4694, 0.5099] | 0.5011 [0.5004, 0.5019] | -0.0115 [-0.0317, +0.0088] | readiness_or_precision_insufficient |
| HJ-px-B1 | 0.4753 [0.4575, 0.4935] | 0.4999 [0.4989, 0.5008] | -0.0246 [-0.0423, -0.0062] | readiness_or_precision_insufficient |
| HJ-car-native | 0.5407 [0.5185, 0.5631] | 0.5384 [0.5297, 0.5470] | +0.0023 [-0.0201, +0.0243] | readiness_or_precision_insufficient |

For HJ-px, HJ-car-m and HJ-car-native, right − wrong is about zero with an interval that contains zero. For HJ-px-B1 it
is −0.0246 [−0.0423, −0.0062]: the right anchor ranks slightly *below* a wrong one. No HJ arm shows evidence that its cost
uses the anchor. Every S10 token is `readiness_or_precision_insufficient` (arm guards).

### Descriptive inventories (no tokens)

Offset, angle and power pooled AUC, R1 (the R2–R4 inventory rows are in `findings.md`):

| ranker | offset | angle | power |
| --- | --- | --- | --- |
| U | 0.6670 [0.6503, 0.6827] | 0.6791 [0.6663, 0.6912] | 0.8058 [0.7729, 0.8396] |
| C* | 0.7748 [0.7524, 0.7962] | 0.7590 [0.7422, 0.7752] | 0.8068 [0.7711, 0.8440] |
| P | 0.5000 [0.5000, 0.5000] | 0.8119 [0.7778, 0.8450] | 0.5000 [0.5000, 0.5000] |
| J | 0.7522 [0.7039, 0.7978] | 0.7765 [0.7402, 0.8115] | 0.8076 [0.7695, 0.8463] |
| J_dec | 0.6904 [0.6737, 0.7062] | 0.6988 [0.6827, 0.7147] | 0.7943 [0.7555, 0.8331] |
| HJ-car-m | 0.4792 [0.4338, 0.5235] | 0.5019 [0.4638, 0.5397] | 0.4982 [0.4547, 0.5428] |
| HJ-car-m-fs5 | 0.5186 [0.4811, 0.5550] | 0.4855 [0.4456, 0.5264] | 0.5355 [0.4877, 0.5832] |
| HJ-car-m-persist | 0.4954 [0.4555, 0.5366] | 0.4619 [0.4266, 0.4965] | 0.5071 [0.4558, 0.5571] |
| HJ-car-m-uniform | 0.4544 [0.4143, 0.4943] | 0.5027 [0.4665, 0.5391] | 0.4533 [0.4074, 0.4999] |
| HJ-car-native | 0.5609 [0.5122, 0.6092] | 0.5317 [0.4797, 0.5848] | 0.5231 [0.4639, 0.5808] |
| HJ-px | 0.5004 [0.4625, 0.5386] | 0.4367 [0.3995, 0.4741] | 0.4947 [0.4541, 0.5370] |
| HJ-px-B1 | 0.4983 [0.4584, 0.5389] | 0.4989 [0.4614, 0.5368] | 0.5059 [0.4559, 0.5555] |
| HWM-car-m | 0.4811 [0.4423, 0.5195] | 0.4898 [0.4533, 0.5269] | 0.4808 [0.4318, 0.5269] |
| HWM-px | 0.4804 [0.4403, 0.5201] | 0.4813 [0.4418, 0.5206] | 0.4987 [0.4564, 0.5424] |
| LeWM-car-m | 0.4467 [0.4138, 0.4794] | 0.4644 [0.4360, 0.4938] | 0.4497 [0.3976, 0.5003] |
| LeWM-px | 0.5177 [0.4864, 0.5499] | 0.5122 [0.4882, 0.5369] | 0.5128 [0.4716, 0.5547] |

P is seed-free and constant on offset and power (0.5000 by construction).

### Readout variants R2–R4 (grid pooled AUC)

| arm | R1 (primary) | R2 (secondary) | R3 (exploratory) | R4 (exploratory) |
| --- | --- | --- | --- | --- |
| HJ-px | 0.4997 [0.4827, 0.5165] | 0.5021 [0.4857, 0.5189] | 0.4916 [0.4753, 0.5081] | 0.5000 [0.4842, 0.5161] |
| HJ-car-m | 0.4896 [0.4694, 0.5099] | 0.5065 [0.4932, 0.5198] | 0.4932 [0.4773, 0.5087] | 0.5024 [0.4880, 0.5163] |
| HJ-px-B1 | 0.4753 [0.4575, 0.4935] | 0.5015 [0.4832, 0.5196] | 0.5188 [0.5019, 0.5357] | 0.5004 [0.4836, 0.5177] |
| HJ-car-native | 0.5407 [0.5185, 0.5631] | 0.5397 [0.5167, 0.5624] | 0.5453 [0.5201, 0.5706] | 0.6420 [0.6190, 0.6657] |
| HJ-car-m-fs5 | 0.4926 [0.4747, 0.5108] | 0.5233 [0.5090, 0.5374] | 0.5022 [0.4865, 0.5179] | 0.5054 [0.4898, 0.5207] |
| HJ-car-m-persist | 0.5387 [0.5208, 0.5566] | 0.5054 [0.4899, 0.5209] | 0.4851 [0.4685, 0.5017] | 0.5015 [0.4860, 0.5172] |
| HJ-car-m-uniform | 0.4934 [0.4764, 0.5111] | 0.4939 [0.4794, 0.5081] | 0.4960 [0.4794, 0.5122] | 0.5017 [0.4865, 0.5170] |
| LeWM-px | 0.5090 [0.4924, 0.5257] | 0.4991 [0.4820, 0.5159] | 0.4974 [0.4812, 0.5142] | 0.4955 [0.4785, 0.5122] |
| LeWM-car-m | 0.4945 [0.4781, 0.5112] | 0.5100 [0.4960, 0.5239] | 0.4884 [0.4737, 0.5028] | 0.5165 [0.5036, 0.5300] |
| HWM-px | 0.5001 [0.4835, 0.5165] | 0.5105 [0.4939, 0.5270] | 0.5582 [0.5406, 0.5762] | 0.5018 [0.4849, 0.5192] |
| HWM-car-m | 0.5033 [0.4852, 0.5217] | 0.4996 [0.4842, 0.5147] | 0.4992 [0.4823, 0.5164] | 0.4955 [0.4796, 0.5110] |

No readout variant lifts an H-JEPA-family arm off chance except HJ-car-native:R4 (0.6420), the exploratory readout that
adds a ranking loss on fit mixed cells; it is still below every one of our rankers, and HJ-car-native fails G-collapse.
S9 (R2 versions of PC1 / PC2) is in the contrast table above.

### Engine position error (#100 metric; DESCRIPTIVE)

H-JEPA arms: the slot-centre probe P_c on the predicted top-level state; floor = P_c on the true frame encoding. U / C\*:
their predicted carriers. The two read-out paths differ, so the H-JEPA rows are not like for like with U / C\*.

| model | t = 15 | t = 30 | t = 60 | t = 150 | t = 225 | floor t = 225 |
| --- | --- | --- | --- | --- | --- | --- |
| HJ-car-m | 0.0129 [0.0121, 0.0138] | 0.0138 [0.0130, 0.0146] | 0.0132 [0.0125, 0.0141] | 0.0135 [0.0127, 0.0143] | 0.0170 [0.0161, 0.0178] | 0.0112 [0.0108, 0.0117] |
| HJ-car-m-fs5 | 0.0126 [0.0119, 0.0133] | 0.0142 [0.0134, 0.0150] | 0.0138 [0.0130, 0.0146] | 0.0158 [0.0148, 0.0169] | 0.0201 [0.0191, 0.0212] | 0.0111 [0.0107, 0.0116] |
| HJ-car-m-persist | 0.0128 [0.0120, 0.0136] | 0.0136 [0.0129, 0.0144] | 0.0131 [0.0123, 0.0139] | 0.0133 [0.0125, 0.0140] | 0.0168 [0.0160, 0.0176] | 0.0112 [0.0108, 0.0117] |
| HJ-car-m-uniform | 0.0106 [0.0099, 0.0113] | 0.0132 [0.0122, 0.0143] | 0.0147 [0.0136, 0.0160] | 0.0163 [0.0155, 0.0172] | 0.0213 [0.0202, 0.0225] | 0.0112 [0.0108, 0.0117] |
| HJ-car-native | 0.0127 [0.0120, 0.0134] | 0.0127 [0.0120, 0.0134] | 0.0115 [0.0109, 0.0122] | 0.0121 [0.0115, 0.0128] | 0.0156 [0.0149, 0.0162] | 0.0150 [0.0143, 0.0157] |
| HJ-px | 0.0138 [0.0132, 0.0144] | 0.0138 [0.0132, 0.0145] | 0.0125 [0.0119, 0.0131] | 0.0121 [0.0115, 0.0127] | 0.0151 [0.0145, 0.0158] | 0.0147 [0.0141, 0.0153] |
| HJ-px-B1 | 0.0147 [0.0140, 0.0153] | 0.0146 [0.0140, 0.0153] | 0.0133 [0.0127, 0.0140] | 0.0130 [0.0123, 0.0136] | 0.0161 [0.0154, 0.0168] | 0.0164 [0.0156, 0.0171] |
| HWM-car-m | 0.0127 [0.0121, 0.0132] | 0.0136 [0.0130, 0.0142] | 0.0116 [0.0111, 0.0122] | 0.0113 [0.0107, 0.0119] | 0.0144 [0.0138, 0.0150] | 0.0121 [0.0116, 0.0127] |
| HWM-px | 0.0146 [0.0140, 0.0153] | 0.0147 [0.0140, 0.0153] | 0.0133 [0.0127, 0.0140] | 0.0128 [0.0121, 0.0134] | 0.0160 [0.0153, 0.0166] | 0.0152 [0.0146, 0.0159] |
| LeWM-car-m | 0.0142 [0.0136, 0.0148] | 0.0144 [0.0138, 0.0150] | 0.0140 [0.0134, 0.0146] | 0.0146 [0.0140, 0.0152] | 0.0186 [0.0178, 0.0194] | 0.0121 [0.0115, 0.0126] |
| LeWM-px | 0.0129 [0.0123, 0.0135] | 0.0144 [0.0137, 0.0150] | 0.0154 [0.0146, 0.0162] | 0.0214 [0.0204, 0.0225] | 0.0255 [0.0243, 0.0268] | 0.0113 [0.0108, 0.0119] |
| U | 0.0505 [0.0460, 0.0550] | 0.0501 [0.0457, 0.0545] | 0.0484 [0.0443, 0.0526] | 0.0459 [0.0423, 0.0496] | 0.0474 [0.0440, 0.0508] | n/a |
| C* | 0.0547 [0.0500, 0.0595] | 0.0545 [0.0499, 0.0592] | 0.0543 [0.0497, 0.0589] | 0.0594 [0.0544, 0.0645] | 0.0670 [0.0617, 0.0723] | n/a |

The H-JEPA predicted-state error sits at or near its true-encoding floor at t = 225 (HJ-px 0.0151 vs 0.0147). With a
launch-insensitive endpoint, this measures how well a launch-independent state predicts the engine positions, not
decision-relevant prediction.

### CLEVRER / Physion-Dominoes (EXPLORATORY; no ranking, no tokens)

Action-free HJ-car-m (HJ-bench) and LeWM-car-m (LeWM-bench) configurations on the #117 item-5 annotation carriers (12
scenes each, fold all, B1, uniform sampler, null action); R1 fitted in-sample on every clip frame; #79 `field_errors`
position_mse at t ∈ {15, 30, 60, 120} from contexts 0–3, mean over scenes × contexts × 3 seeds, beside #117 item-5 J (LC,
full, fold all). Records: `benchmarks/<bench>/<arm>/seed-*/record.json` (hashed in `records-manifest.json`).

**CLEVRER**

| model | t = 15 | t = 30 | t = 60 | t = 120 |
| --- | --- | --- | --- | --- |
| HJ-bench | 0.02704 | 0.03225 | 0.01674 | 0.01508 |
| LeWM-bench | 0.004557 | 0.00724 | 0.00905 | 0.01819 |
| J (#117 item 5, LC, full) F-15-continuous | 0.001795 | 0.002058 | 0.001214 | 0.006567 |
| J (#117 item 5, LC, full) F-5-continuous | 0.002574 | 0.004234 | 0.004886 | 0.01537 |

**Physion-Dominoes**

| model | t = 15 | t = 30 | t = 60 | t = 120 |
| --- | --- | --- | --- | --- |
| HJ-bench | 0.0009756 | 0.001497 | 0.00156 | 0.002182 |
| LeWM-bench | 0.0006563 | 0.002034 | 0.01986 | 0.04742 |
| J (#117 item 5, LC, full) F-15-continuous | 0.0001865 | 0.0002947 | 0.0005389 | 0.001101 |
| J (#117 item 5, LC, full) F-5-continuous | 0.0004014 | 0.0008574 | 0.001726 | 0.002862 |

Top-level effective rank / χ per seed: CLEVRER HJ-bench 16.2 / 0.084, 15.9 / 0.072, 16.3 / 0.059; LeWM-bench
34.1 / 0.483, 34.1 / 0.475, 33.7 / 0.475. Physion HJ-bench 11.0 / 0.059, 11.2 / 0.065, 10.7 / 0.069; LeWM-bench 21.4 / 0.377, 21.8 / 0.371, 21.5 / 0.375.
J has the lowest position error at every horizon on both benchmarks. There are no candidate actions, so these rows rank
nothing.

## Mechanism

- **The encoder represents the outcome.** HJ-px's R1 readout ceiling, on the *true* frame-225 encodings of the policy
  grid candidates, reaches policy AUC 0.8713–0.8972 per seed (arm mean 0.8812; HWM-px 0.8719 / 0.8308 / 0.8900, HJ-px-B1
  0.80–0.82). The information needed to rank is present in the top-level latent of the real endpoint frame.
- **The predicted endpoint ignores the launch.** HJ-px's endpoint action sensitivity is 1.14e-6 per seed (threshold
  0.05), and its within-episode share χ is 0.0014–0.0015 (threshold 0.05): the top-level latent barely moves within an
  episode, so it encodes the level, a slow feature, rather than the cascade. Ranking the predicted endpoints therefore
  ranks at chance (grid AUC 0.4997), and the wrong-anchor control is ≈ 0 (S10).
- This is the slow-feature collapse that the paper's App. L reports when a slow feature dominates (here the level
  identity), together with its App. K signatures. HJ-car-m and its three ablations show the same pattern at a lower
  ceiling (sensitivity 1.5–2.9e-7, χ 0.016–0.029, per-seed ceiling 0.53–0.63).
- **The temporal and representational levers do not change it.** S1–S4 are all within ±0.014 of zero (HJ − LeWM, HJ −
  HWM), and every lever arm fails its guards too. Flat LeWM-px keeps χ ≥ 0.05 and a sensitivity of 0.034–0.052 but loses
  the outcome (ceiling 0.5587).
- **HJ-car-native is the only H-JEPA arm with non-zero action sensitivity** (0.145–0.161 per seed; G-action passes and
  G-readout passes at 0.7182). Its χ is 0.011 on every seed, so it still fails G-collapse. Its grid AUC is 0.5407 under R1.

## Disposition (δ rule)

Frozen disposition 3 applies: **per-contrast tokens only; outcome branch: no scoped ranking claim.**

- PC1 (HJ-px − U) = −0.2447 [−0.2688, −0.2204] and PC2 (HJ-car-m − U) = −0.2548 [−0.2829, −0.2273]. Both tokens, in both
  directions, are `readiness_or_precision_insufficient`, because HJ-px and HJ-car-m fail G-collapse and G-action (and
  HJ-car-m G-readout).
- Disposition 1 does not apply: no PC1 / PC2 is `supported` in the H-JEPA − U direction, and S10 is not `supported`.
- Disposition 2 does not apply: it needs both primary arms to pass every guard. The manuscript may not state that our
  model beats or is better than H-JEPA, LeWM or HWM, and may not use "state of the art" against them, on this experiment.
- Disposition 5 (budget asymmetry) is false, so U-B3 is not run. The closed-loop secondary is not run.
- The permitted statement is: "H-JEPA, reproduced faithfully (G-fid pass), collapses on action-sparse NovPhy: its
  predicted endpoint ignores the launch and ranks at chance; the frozen rule therefore makes no ranking claim."

## Deviations and disclosures

1. **Owner GPU-window rule and the 10:38 stop (2026-10-19).** The owner ruled at 10:40 AEDT that GPU jobs run only from
   18:00 to 07:00 Australia/Melbourne. At 10:38 every #123 GPU process group was SIGSTOP-paused and then SIGKILLed.
   Training queues resumed from their atomic `resume.pt` (at most about 20 min per queue recomputed; `train-log.jsonl` is
   append-only and shows those steps twice). The Tier-T H-JEPA-3 segments started 09:14–10:03 were killed with no ledger
   line (about 2.9 GPU-h) and resumed from Lightning `last.ckpt` (global step 6566 / 2000 / 4000 of 45 962); the official
   mid-epoch resume does not restore the data order. Everything after that ran through the night-window scheduler
   (`scripts/gpu_window.py`, `docs/gpu-window-scheduler.md`): SIGTERM at 06:50 → `resume.pt` at the next optimizer step →
   exit 75 → relaunch the next night. Commit `73298595` (A1); scheduler review fixes `f7809a21`.
2. **Earlier infrastructure failures (pre-freeze and at the first training launch).** (a) Tier E first ran with
   transformers 4.57.6 and every load failed (`AttributeError: ViTMLP`; 9 runs × ~6 s;
   `fidelity/setup-failures/`); fixed by transformers 5.20.1. (b) Three Tier-T H-JEPA-3 launches (seeds 42/43/44) died
   of CUDA OOM before step 900: batch 128 does not fit a 32 GB RTX 5090 without activation checkpointing, and a per-step
   reference cycle kept each step's graph alive. The launcher enables non-reentrant activation checkpointing and calls
   `gc.collect()` every 10 batches (gradients unchanged); the runs were relaunched once from scratch
   (`fidelity/infrastructure-failures/`; commit `1d079096`, disclosed in the freeze decisions, `b06e7397`). (c) The first
   `--smoke` died after about 50 s with no output and was relaunched once, detached (`logs/smoke-attempt1-killed.log`).
   (d) The first main-training launch was stopped after about 2 min, before any checkpoint, because of allocator OOM
   retries beside a Tier-T run; it was relaunched with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   (`logs/train-attempt1-stopped/`). None of these is a typed failure, and none produced a model output.
3. **The fd-limit `--score` crash (2026-10-21, first attempt, before any score).** Run `123-score#1` (06:25 AEDT) died
   after 26 s in `build_eval_cache`: cron starts jobs with a soft open-file limit of 1024, PyTorch passes tensors returned
   by pool workers as file descriptors, and the pool broke (`RuntimeError: received 0 items of ancdata` →
   `BrokenProcessPool`). No model was rolled out, no score or record was written, and the cache file (written atomically)
   was never created; the workers had started reading evaluation-split branch files for the cache (a data-loading read
   only). Fix, infrastructure only: `gpu_window_cron.sh` raises the soft limit to the hard limit before every tick
   (commit `88413471`). Run `123-score#2` (18:00 AEDT) is the first and only scoring of #123. Log:
   `logs/scheduler-123-score.log`.
4. **Amendments A1 and A2** (registered in `scripts/post_freeze_amendments.py`; full text in `decisions.md`).
   A1 (`73298595`, scheduling only): SIGTERM/SIGUSR1 stop-and-resume of the training queue (RNG restored after the
   loader iterator; bitwise CPU resume test), Lightning graceful stop in `hjepa_train_ckpt.py`, SIGTERM forwarding and
   exit codes in `run_hjepa_fidelity.py`. A2 (`c9b8ad6c`, before any readout or evaluation-feature read): G-records
   gates every token until `--validate` passes; absent paired estimates abstain; G-nonfinite counts R1 and R2 separately;
   G-port adds a positive-LR optimizer step (re-run PASS on all five arms). No objective, threshold, contrast, rule or
   token definition changed; training, resume, RNG and data code are the A1 bytes.
5. **Evaluation-artefact access wording.** The pre-A2 wording "no evaluation file read" was literally inaccurate. Before
   scoring, `stream_rows` parsed the mixed-stream `branches.json` and kept only the fit / policy rows; the #109 cohort
   plan was parsed for split membership; and `--prepare` and every `load_plan` sha256-hashed pinned inputs that include
   evaluation artefacts (`branches.json`, the Gate-B evaluation anchors, the Gate-B U / C\* records, plan and summary,
   the #101 J / J_dec records). Metadata parsing and hashing only: no evaluation feature or label entered training,
   readout fitting, λ selection, the smoke or the scoring-path rehearsal. Corrected in A2 (`c9b8ad6c`).
6. **Hardware.** 4 × RTX 5090 (32 GB) instead of the paper's H200: hence the Tier-T activation checkpointing above, and
   the bf16 / cuDNN kernels are not bitwise reproducible run to run (the restored state on resume is exact).
7. **Chronology of #125.** The #125 trigger ran on fit data only and wrote `trigger.json` at 05:59 AEDT on 21 Oct, before
   #123's first evaluation read (06:25); `123-score` depended on it (`f7809a21`). #125's `plan.json` was frozen at
   2026-10-21T07:26:34Z, after #123's scoring but before the #125 implementer read any #123 result (disclosed in
   `c9d0b64c`).
8. **Compute ledger double count** (found at publication; see Compute). The published `compute.json` / `findings.md`
   total counts four training arm-seeds twice. The publication files are left as validated; the corrected figure is below.

## Compute

Caps count active compute time (GPU seconds = each process's wall time on its GPU; daytime pauses do not count). Cap:
260 GPU-h and 72 h wall on 4 × RTX 5090.

**As published** (`compute.json`): #123 788 488 GPU-s (219.0 GPU-h) + fidelity 328 671 GPU-s (91.3 GPU-h: Tier E 20.8,
Tier T 70.5) = **310.3 GPU-h**, which reads as over the cap.

**Correction.** The training queue's ledger (`mark()` in `run_hjepa_comparison.py::train`) adds the arm-seed's
`report.json` seconds every time the queue reaches that arm-seed. `train_run` returns the existing report when
`model.pt` exists, so an arm-seed that finished in one night and was passed over when its queue relaunched the next
night was counted twice. This hit exactly four arm-seeds, whose ledger is 2.000 × their `report.json`: HJ-px 20260908
(61 231 s), HJ-px 20260909 (78 610 s), HJ-px 20260910 (77 610 s) and HWM-px 20260908 (68 147 s). The other 29 training
ledgers equal their reports. The overcount is 285 598 GPU-s (79.3 GPU-h).

| item | GPU-h |
| --- | --- |
| main training, 33 arm-seeds (from `report.json`) | 133.5 |
| λ selection (train + readouts) | 3.5 |
| readouts (33 arm-seeds) | 1.2 |
| CLEVRER / Physion benchmarks | 1.4 |
| smoke, scoring (incl. S10), position cache | 0.1 |
| **#123 recorded, corrected** | **139.7** |
| fidelity Tier E / Tier T (ledger) | 20.8 / 70.5 |
| **recorded total, corrected** | **231.0** |
| unrecorded: H-JEPA-3 segments killed at 10:38 (≈ 2.9), training steps recomputed after 10:38 (≈ 0.5), first training launch, first smoke, first `--score` (≈ 0.2) | ≈ 3.6 |
| **active compute total** | **≈ 234.6 (cap 260)** |

Active wall time with any #123 GPU process running was about 42 h over four windows (10-18 18:37 → 10-19 10:38; 10-19
18:00 → 10-20 06:50; 10-20 18:00 → 10-21 06:26; 10-21 18:00 → 18:23), against the 72 h cap. HJ-px trained at 0.48–0.61
s/step (17.0–21.8 GPU-h per seed) against the smoke's 0.442 s/step (15.7 GPU-h projected), because the GPUs were shared
with Tier T. Engine seconds: 0. No scope was cut.

## Follow-up: #125

[#125](https://github.com/Sino-Huang/NovPhy/issues/125) (`docs/issue-125-hjepa-idm.md`) was pre-registered on
2026-10-19, before any #123 evaluation read, as the follow-up if the primary H-JEPA arms collapse. It adds the paper's
own anti-collapse remedy, an inverse-dynamics term with action-SIGReg (§4.3 Eq. 8, App. K), and changes nothing else in
#123's frozen plan. Its trigger fired on fit data: all six #123 primary arm-seeds (HJ-px, HJ-car-m × 3) fail G-collapse
and G-action. It is frozen (`c9d0b64c`) and training in the night window.

## Reproduction

```
conda activate novphy && source env.sh
M="python -u -m scripts.run_hjepa_comparison"
python scripts/run_hjepa_fidelity.py --table                  # fidelity/gfid.json
CUDA_VISIBLE_DEVICES=g $M --train --gpu g                     # g = 0..3, night window only (scheduler)
CUDA_VISIBLE_DEVICES=g $M --readouts --gpu g && $M --freeze-readouts
CUDA_VISIBLE_DEVICES=0 $M --score                             # once
$M --publish
CUDA_VISIBLE_DEVICES=0 $M --validate                          # exit 0
CUDA_VISIBLE_DEVICES="" pytest tests/test_issue_123_hjepa_comparison.py    # 18 passed
```
