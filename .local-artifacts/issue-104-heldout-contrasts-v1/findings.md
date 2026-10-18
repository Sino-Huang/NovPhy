# Issue-104 work item 3: the #96 contrast set on the held-out #109 evaluation split — findings

- identity `issue-104-heldout-contrasts-v1`, plan v1 frozen 2026-10-18T07:28:19Z before any evaluation score
- validation command: `python -u -m scripts.run_heldout_96_contrasts --validate`; zero engine seconds; endpoint 225; every interval DESCRIPTIVE (level-clustered, 10000 draws, PCG64 7201)
- stack: #77 N1 hybrid / continuous checkpoints bound by the #87 plan (the #96 stack); #77 N1 MatchedController per seed and family (the #96 controllers); anchors: #105 F77 store: the #71 repaired-adapter batch-1 parse of each evaluation level's decision frame (shot frame 0 of its first complete candidate, the Gate-B rule), parsed under deterministic_scoring()
- cells (Gate-B, mixed only): grid 311, offset 76, angle 78, power 47 levels x 3 seeds; 54 records, 419364 candidate rollouts
- primary cost: tie_free (pre-declared); secondary: count (#96's cost)
- #96 reference (development, count cost): theta_grid 0.0700 [-0.0057, 0.1437] over 42 cells, half-width 0.0747
- post-freeze amendment A1 (reporting / guard propagation, after scoring; registered in `scripts/post_freeze_amendments.py`): the frozen plan's guard_failure rule ('any failed guard turns every token of that cost into the stop token') is applied to the grid secondaries J - M and J - OL too; the frozen #96 secondary rule (own half-width in place of G6) and the #96 literal G6 bound are kept as DESCRIPTIVE readings beside the tokens; no record, estimate or interval changed

## tie_free cost (primary)

- primary theta_grid = AUC_J - AUC_F* (F* = F-5-macro): **0.0134 [-0.0163, 0.0431]**, half-width 0.0297 -> **readiness_or_precision_insufficient** (delta 0.02; G6 half-width <= 0.02; DESCRIPTIVE reading with the #96 plan's literal G6 bound 0.05: readiness_or_precision_insufficient)
- secondary J-M (grid): 0.0080 [-0.0044, 0.0204], own half-width 0.0124 -> **readiness_or_precision_insufficient** (DESCRIPTIVE readings, not tokens: by the #96 secondary rule, which skips the primary's G6, readiness_or_precision_insufficient; at the literal bound readiness_or_precision_insufficient)
- secondary J-OL (grid): 0.0401 [0.0247, 0.0556], own half-width 0.0155 -> **readiness_or_precision_insufficient** (DESCRIPTIVE readings, not tokens: by the #96 secondary rule, which skips the primary's G6, supported; at the literal bound supported)
- replication of J - F* on offset/angle/power (mean >= delta AND lower > 0, DESCRIPTIVE): 0/3 (not interpretable: a guard failed)
- failed guards: G6; every token of this cost is readiness_or_precision_insufficient

### Guards

| guard | pass | observed |
|---|---|---|
| G1 | True | `{"exact": true, "exact_record": "decision--angle--issue-77-n1-001--seed20260910--D", "g1_96_criterion": [{"chosen_equal": true, "ok": true, "record": "decision--angle--issue-77-n1-001--seed20260908--J", "selected_cost_delta": 0.000301361083984375, "tolerance": 0.001}, {"chosen_equal": true, "ok": true, "record": "decision--angle--issue-77-n1-001--seed20260908--F-15-continuous", "selected_cost_delta": 2.8848648071289062e-05, "tolerance": 0.001}]}` |
| G2 | True | `{"levels": 337, "reparse_bit_equal": true, "store_policy": {"cudnn_allow_tf32": false, "cudnn_benchmark": false, "cudnn_deterministic": true, "matmul_allow_tf32": false}, "store_sha256": "sha256:75931f58a24bf1e583b1b7fd66fc67bfa9950f230877978994bc2233af753c0a"}` |
| G3 | True | `{"nonfinite_share": 0.0, "verdict_resolution": 1.0}` |
| G4 | True | `{"cells": 933, "tied_cells": 0, "tied_share": 0.0}` |
| G5 | True | `{"J_modal_pair": "1-macro", "J_nonmodal_step_share": 0.2567397294919628, "rank_divergent_share": 0.969989281886388}` |
| G6 | False | `{"half_width": 0.029724644103663385, "threshold": 0.02}` |
| G7 | True | `{"records": 54}` |
| caps | True | `{"elapsed_wall_seconds": 1344.0, "gpu_seconds": 3801.7045359690674, "workers_terminal": true}` |

### G6 per inventory (J - F*)

| inventory | mixed levels / cells | half-width | <= 0.02 | <= 0.05 (#96 literal) | <= design delta | levels for 0.02 by 1/sqrt(m) |
|---|---|---|---|---|---|---|
| grid | 311 / 933 | 0.0297 | False | True | False (0.02) | 687 |
| offset | 76 / 228 | 0.0632 | False | False | False (0.05) | 758 |
| angle | 78 / 234 | 0.0466 | False | True | True (0.05) | 424 |
| power | 47 / 141 | 0.0720 | False | False | False (0.05) | 609 |

### Contrasts per inventory (AUC difference, paired by level x seed, level-clustered; never pooled)

| inventory | contrast | estimate [95% DESCRIPTIVE] | half-width | <= 0.02 | <= design delta | #104 design members | levels for 0.02 |
|---|---|---|---|---|---|---|---|
| grid | J - F* [F-5-macro] | 0.0134 [-0.0163, 0.0431] | 0.0297 | False | False | 208 | 687 |
| grid | J - F_hindsight [F-15-continuous] | -0.0841 [-0.1147, -0.0537] | 0.0305 | False | False | 293 | 724 |
| grid | J - F(1,macro) | 0.0338 [0.0117, 0.0555] | 0.0219 | False | False | 293 | 374 |
| grid | J - F(1,continuous) | 0.0271 [0.0080, 0.0456] | 0.0188 | True | True | 159 | 274 |
| grid | J - M | 0.0080 [-0.0044, 0.0204] | 0.0124 | True | True | 84 | 119 |
| grid | J - OL | 0.0401 [0.0247, 0.0556] | 0.0155 | True | True | 176 | 187 |
| grid | J - T | -0.0011 [-0.0145, 0.0128] | 0.0136 | True | True | 57 | 145 |
| grid | J - D | 0.0582 [0.0368, 0.0794] | 0.0213 | False | False | 253 | 352 |
| grid | C-J - C-F* [C-F-5] | -0.0493 [-0.0602, -0.0386] | 0.0108 | True | True | 235 | 91 |
| offset | J - F* [F-5-continuous] | 0.0111 [-0.0520, 0.0743] | 0.0632 | False | False | 32 | 758 |
| offset | J - F_hindsight [F-5-macro] | -0.1606 [-0.2071, -0.1128] | 0.0472 | False | True | 17 | 423 |
| offset | J - F(1,macro) | 0.1061 [0.0636, 0.1504] | 0.0434 | False | True | 32 | 359 |
| offset | J - F(1,continuous) | -0.0404 [-0.0776, -0.0009] | 0.0384 | False | True | 22 | 280 |
| offset | J - M | -0.0611 [-0.0927, -0.0300] | 0.0314 | False | True | 20 | 187 |
| offset | J - OL | 0.0044 [-0.0333, 0.0439] | 0.0386 | False | True | 10 | 283 |
| offset | J - T | -0.0610 [-0.1022, -0.0181] | 0.0421 | False | True | 12 | 337 |
| offset | J - D | 0.0102 [-0.0346, 0.0561] | 0.0453 | False | True | 31 | 391 |
| offset | C-J - C-F* [C-F-5] | -0.0646 [-0.0795, -0.0480] | 0.0158 | True | True | 28 | 48 |
| angle | J - F* [F-5-macro] | -0.1596 [-0.2061, -0.1130] | 0.0466 | False | True | 10 | 424 |
| angle | J - F_hindsight [F-5-macro] | -0.1596 [-0.2061, -0.1130] | 0.0466 | False | True | 10 | 424 |
| angle | J - F(1,macro) | 0.1053 [0.0706, 0.1419] | 0.0356 | False | True | 10 | 248 |
| angle | J - F(1,continuous) | -0.0390 [-0.0729, -0.0034] | 0.0347 | False | True | 6 | 236 |
| angle | J - M | -0.0490 [-0.0789, -0.0189] | 0.0300 | False | True | 2 | 176 |
| angle | J - OL | -0.0086 [-0.0462, 0.0308] | 0.0385 | False | True | 8 | 289 |
| angle | J - T | -0.0603 [-0.0943, -0.0242] | 0.0350 | False | True | 4 | 240 |
| angle | J - D | 0.0131 [-0.0281, 0.0558] | 0.0420 | False | True | 55 | 344 |
| angle | C-J - C-F* [C-F-5] | -0.0604 [-0.0752, -0.0437] | 0.0157 | True | True | 44 | 49 |
| power | J - F* [F-5-macro] | -0.2617 [-0.3360, -0.1921] | 0.0720 | False | False | 36 | 609 |
| power | J - F_hindsight [F-15-micro] | -0.3095 [-0.3684, -0.2550] | 0.0567 | False | False | 15 | 378 |
| power | J - F(1,macro) | 0.1299 [0.0669, 0.1944] | 0.0637 | False | False | 15 | 478 |
| power | J - F(1,continuous) | -0.1030 [-0.1739, -0.0331] | 0.0704 | False | False | 17 | 582 |
| power | J - M | -0.0250 [-0.0695, 0.0194] | 0.0444 | False | True | 12 | 233 |
| power | J - OL | 0.0098 [-0.0398, 0.0588] | 0.0493 | False | True | 15 | 287 |
| power | J - T | -0.0703 [-0.1217, -0.0227] | 0.0495 | False | True | 19 | 289 |
| power | J - D | 0.0629 [0.0123, 0.1146] | 0.0511 | False | False | 13 | 308 |
| power | C-J - C-F* [C-F-5] | -0.1715 [-0.1949, -0.1496] | 0.0226 | False | True | 53 | 61 |

### Per inventory

| inventory | F* (cross-fit) | F_hindsight | C-F* | AUC J | AUC F* | top-1 J / F* / chance | fixed-pair spread | J modal pair, non-modal share | tied share |
|---|---|---|---|---|---|---|---|---|---|
| grid | F-5-macro | F-15-continuous | C-F-5 | 0.5550 [0.5393, 0.5707] | 0.5416 [0.5198, 0.5637] | 55/933 / 73/933 / 0.0890 | 0.4969 [0.4765, 0.5177] | 1-macro, 0.2567 | 0.0000 |
| offset | F-5-continuous | F-5-macro | C-F-5 | 0.4098 [0.3767, 0.4441] | 0.3987 [0.3483, 0.4491] | 10/228 / 8/228 / 0.0921 | 0.5045 [0.4763, 0.5326] | 1-macro, 0.2584 | 0.0000 |
| angle | F-5-macro | F-5-macro | C-F-5 | 0.4390 [0.4097, 0.4704] | 0.5987 [0.5667, 0.6315] | 10/234 / 28/234 / 0.0789 | 0.4754 [0.4469, 0.5036] | 1-macro, 0.2657 | 0.0000 |
| power | F-5-macro | F-15-micro | C-F-5 | 0.4591 [0.4131, 0.5032] | 0.7207 [0.6797, 0.7630] | 9/141 / 17/141 / 0.0543 | 0.5592 [0.4968, 0.6256] | 1-macro, 0.2694 | 0.0000 |

### Per-arm member-mean AUC (mixed cells)

| arm | grid | offset | angle | power |
|---|---|---|---|---|
| F-1-continuous | 0.5280 | 0.4502 | 0.4781 | 0.5621 |
| F-1-micro | 0.5130 | 0.4423 | 0.4676 | 0.5309 |
| F-1-macro | 0.5213 | 0.3037 | 0.3337 | 0.3292 |
| F-5-continuous | 0.6293 | 0.3987 | 0.4858 | 0.7653 |
| F-5-micro | 0.6310 | 0.3943 | 0.4770 | 0.7632 |
| F-5-macro | 0.5416 | 0.5704 | 0.5987 | 0.7207 |
| F-15-continuous | 0.6391 | 0.3460 | 0.4277 | 0.7665 |
| F-15-micro | 0.6301 | 0.3797 | 0.4600 | 0.7686 |
| F-15-macro | 0.6060 | 0.4194 | 0.4595 | 0.7377 |
| J | 0.5550 | 0.4098 | 0.4390 | 0.4591 |
| M | 0.5470 | 0.4710 | 0.4880 | 0.4841 |
| T | 0.5562 | 0.4709 | 0.4993 | 0.5294 |
| D | 0.4968 | 0.3997 | 0.4259 | 0.3961 |
| OL | 0.5149 | 0.4054 | 0.4476 | 0.4492 |
| C-F-1 | 0.4842 | 0.5441 | 0.5332 | 0.5476 |
| C-F-5 | 0.6103 | 0.5539 | 0.6216 | 0.7765 |
| C-F-15 | 0.6229 | 0.4447 | 0.5202 | 0.7671 |
| C-J | 0.5610 | 0.4893 | 0.5612 | 0.6050 |

## count cost (secondary)

- primary theta_grid = AUC_J - AUC_F* (F* = F-5-macro): **0.0893 [0.0685, 0.1107]**, half-width 0.0211 -> **readiness_or_precision_insufficient** (delta 0.02; G6 half-width <= 0.02; DESCRIPTIVE reading with the #96 plan's literal G6 bound 0.05: readiness_or_precision_insufficient)
- secondary J-M (grid): 0.0126 [0.0027, 0.0226], own half-width 0.0100 -> **readiness_or_precision_insufficient** (DESCRIPTIVE readings, not tokens: by the #96 secondary rule, which skips the primary's G6, readiness_or_precision_insufficient; at the literal bound readiness_or_precision_insufficient)
- secondary J-OL (grid): 0.0284 [0.0141, 0.0429], own half-width 0.0144 -> **readiness_or_precision_insufficient** (DESCRIPTIVE readings, not tokens: by the #96 secondary rule, which skips the primary's G6, readiness_or_precision_insufficient; at the literal bound readiness_or_precision_insufficient)
- replication of J - F* on offset/angle/power (mean >= delta AND lower > 0, DESCRIPTIVE): 0/3 (not interpretable: a guard failed)
- failed guards: G4, G6; every token of this cost is readiness_or_precision_insufficient

### Guards

| guard | pass | observed |
|---|---|---|
| G1 | True | `{"exact": true, "exact_record": "decision--angle--issue-77-n1-001--seed20260910--D", "g1_96_criterion": [{"chosen_equal": true, "ok": true, "record": "decision--angle--issue-77-n1-001--seed20260908--J", "selected_cost_delta": 0.000301361083984375, "tolerance": 0.001}, {"chosen_equal": true, "ok": true, "record": "decision--angle--issue-77-n1-001--seed20260908--F-15-continuous", "selected_cost_delta": 2.8848648071289062e-05, "tolerance": 0.001}]}` |
| G2 | True | `{"levels": 337, "reparse_bit_equal": true, "store_policy": {"cudnn_allow_tf32": false, "cudnn_benchmark": false, "cudnn_deterministic": true, "matmul_allow_tf32": false}, "store_sha256": "sha256:75931f58a24bf1e583b1b7fd66fc67bfa9950f230877978994bc2233af753c0a"}` |
| G3 | True | `{"nonfinite_share": 0.0, "verdict_resolution": 1.0}` |
| G4 | False | `{"cells": 933, "tied_cells": 395, "tied_share": 0.42336548767416937}` |
| G5 | True | `{"J_modal_pair": "1-macro", "J_nonmodal_step_share": 0.2567397294919628, "rank_divergent_share": 0.3333333333333333}` |
| G6 | False | `{"half_width": 0.021133949673941624, "threshold": 0.02}` |
| G7 | True | `{"records": 54}` |
| caps | True | `{"elapsed_wall_seconds": 1344.0, "gpu_seconds": 3801.7045359690674, "workers_terminal": true}` |

### G6 per inventory (J - F*)

| inventory | mixed levels / cells | half-width | <= 0.02 | <= 0.05 (#96 literal) | <= design delta | levels for 0.02 by 1/sqrt(m) |
|---|---|---|---|---|---|---|
| grid | 311 / 933 | 0.0211 | False | True | False (0.02) | 348 |
| offset | 76 / 228 | 0.0398 | False | True | True (0.05) | 302 |
| angle | 78 / 234 | 0.0362 | False | True | True (0.05) | 257 |
| power | 47 / 141 | 0.0503 | False | False | False (0.05) | 298 |

### Contrasts per inventory (AUC difference, paired by level x seed, level-clustered; never pooled)

| inventory | contrast | estimate [95% DESCRIPTIVE] | half-width | <= 0.02 | <= design delta | #104 design members | levels for 0.02 |
|---|---|---|---|---|---|---|---|
| grid | J - F* [F-5-macro] | 0.0893 [0.0685, 0.1107] | 0.0211 | False | False | 208 | 348 |
| grid | J - F_hindsight [F-1-continuous] | 0.0148 [0.0010, 0.0288] | 0.0139 | True | True | 293 | 150 |
| grid | J - F(1,macro) | 0.0282 [0.0123, 0.0437] | 0.0157 | True | True | 293 | 193 |
| grid | J - F(1,continuous) | 0.0148 [0.0010, 0.0288] | 0.0139 | True | True | 159 | 150 |
| grid | J - M | 0.0126 [0.0027, 0.0226] | 0.0100 | True | True | 84 | 78 |
| grid | J - OL | 0.0284 [0.0141, 0.0429] | 0.0144 | True | True | 176 | 162 |
| grid | J - T | 0.0316 [0.0179, 0.0452] | 0.0136 | True | True | 57 | 145 |
| grid | J - D | -0.0008 [-0.0159, 0.0139] | 0.0149 | True | True | 253 | 172 |
| grid | C-J - C-F* [C-F-5] | 0.0264 [0.0082, 0.0440] | 0.0179 | True | True | 235 | 249 |
| offset | J - F* [F-5-macro] | -0.1152 [-0.1559, -0.0762] | 0.0398 | False | True | 32 | 302 |
| offset | J - F_hindsight [F-5-micro] | -0.1200 [-0.1533, -0.0867] | 0.0333 | False | True | 17 | 211 |
| offset | J - F(1,macro) | 0.0035 [-0.0250, 0.0318] | 0.0284 | False | True | 32 | 154 |
| offset | J - F(1,continuous) | 0.0112 [-0.0162, 0.0377] | 0.0269 | False | True | 22 | 138 |
| offset | J - M | -0.0153 [-0.0430, 0.0119] | 0.0274 | False | True | 20 | 144 |
| offset | J - OL | -0.0176 [-0.0458, 0.0102] | 0.0280 | False | True | 10 | 150 |
| offset | J - T | -0.0334 [-0.0566, -0.0102] | 0.0232 | False | True | 12 | 103 |
| offset | J - D | -0.0023 [-0.0346, 0.0305] | 0.0325 | False | True | 31 | 202 |
| offset | C-J - C-F* [C-F-5] | -0.0636 [-0.0857, -0.0401] | 0.0228 | False | True | 28 | 99 |
| angle | J - F* [F-5-macro] | -0.0874 [-0.1243, -0.0518] | 0.0362 | False | True | 10 | 257 |
| angle | J - F_hindsight [F-5-macro] | -0.0874 [-0.1243, -0.0518] | 0.0362 | False | True | 10 | 257 |
| angle | J - F(1,macro) | -0.0047 [-0.0288, 0.0192] | 0.0240 | False | True | 10 | 113 |
| angle | J - F(1,continuous) | 0.0034 [-0.0184, 0.0251] | 0.0217 | False | True | 6 | 93 |
| angle | J - M | -0.0149 [-0.0393, 0.0089] | 0.0241 | False | True | 2 | 114 |
| angle | J - OL | -0.0156 [-0.0410, 0.0087] | 0.0248 | False | True | 8 | 121 |
| angle | J - T | -0.0204 [-0.0422, 0.0012] | 0.0217 | False | True | 4 | 92 |
| angle | J - D | -0.0166 [-0.0466, 0.0128] | 0.0297 | False | True | 55 | 173 |
| angle | C-J - C-F* [C-F-5] | -0.0497 [-0.0759, -0.0227] | 0.0266 | False | True | 44 | 138 |
| power | J - F* [F-5-macro] | -0.0323 [-0.0830, 0.0177] | 0.0503 | False | False | 36 | 298 |
| power | J - F_hindsight [F-15-macro] | -0.0391 [-0.0875, 0.0098] | 0.0487 | False | True | 15 | 279 |
| power | J - F(1,macro) | -0.0238 [-0.0649, 0.0160] | 0.0404 | False | True | 15 | 193 |
| power | J - F(1,continuous) | 0.0001 [-0.0340, 0.0333] | 0.0336 | False | True | 17 | 133 |
| power | J - M | -0.0202 [-0.0561, 0.0159] | 0.0360 | False | True | 12 | 153 |
| power | J - OL | -0.0434 [-0.0810, -0.0051] | 0.0380 | False | True | 15 | 170 |
| power | J - T | -0.0166 [-0.0565, 0.0224] | 0.0394 | False | True | 19 | 183 |
| power | J - D | -0.0498 [-0.0947, -0.0050] | 0.0448 | False | True | 13 | 237 |
| power | C-J - C-F* [C-F-5] | -0.0042 [-0.0556, 0.0464] | 0.0510 | False | False | 53 | 306 |

### Per inventory

| inventory | F* (cross-fit) | F_hindsight | C-F* | AUC J | AUC F* | top-1 J / F* / chance | fixed-pair spread | J modal pair, non-modal share | tied share |
|---|---|---|---|---|---|---|---|---|---|
| grid | F-5-macro | F-1-continuous | C-F-5 | 0.5226 [0.5098, 0.5355] | 0.4333 [0.4176, 0.4490] | 34/933 / 17/933 / 0.0890 | 0.5128 [0.4935, 0.5325] | 1-macro, 0.2567 | 0.4234 |
| offset | F-5-macro | F-5-micro | C-F-5 | 0.5105 [0.4844, 0.5357] | 0.6257 [0.5906, 0.6604] | 10/228 / 15/228 / 0.0921 | 0.5232 [0.4859, 0.5613] | 1-macro, 0.2584 | 0.5307 |
| angle | F-5-macro | F-5-macro | C-F-5 | 0.5048 [0.4834, 0.5258] | 0.5922 [0.5609, 0.6237] | 5/234 / 10/234 / 0.0789 | 0.4492 [0.4197, 0.4795] | 1-macro, 0.2657 | 0.5085 |
| power | F-5-macro | F-15-macro | C-F-5 | 0.4928 [0.4615, 0.5228] | 0.5251 [0.4889, 0.5612] | 2/141 / 6/141 / 0.0543 | 0.5391 [0.4841, 0.5968] | 1-macro, 0.2694 | 0.5248 |

### Per-arm member-mean AUC (mixed cells)

| arm | grid | offset | angle | power |
|---|---|---|---|---|
| F-1-continuous | 0.5078 | 0.4993 | 0.5014 | 0.4927 |
| F-1-micro | 0.5036 | 0.4832 | 0.4886 | 0.4951 |
| F-1-macro | 0.4944 | 0.5070 | 0.5095 | 0.5166 |
| F-5-continuous | 0.4290 | 0.6284 | 0.5822 | 0.3822 |
| F-5-micro | 0.4148 | 0.6305 | 0.5689 | 0.4034 |
| F-5-macro | 0.4333 | 0.6257 | 0.5922 | 0.5251 |
| F-15-continuous | 0.4569 | 0.4368 | 0.4454 | 0.3997 |
| F-15-micro | 0.4524 | 0.4707 | 0.4724 | 0.3772 |
| F-15-macro | 0.4908 | 0.5322 | 0.5192 | 0.5319 |
| J | 0.5226 | 0.5105 | 0.5048 | 0.4928 |
| M | 0.5100 | 0.5258 | 0.5197 | 0.5130 |
| T | 0.4910 | 0.5439 | 0.5252 | 0.5094 |
| D | 0.5234 | 0.5128 | 0.5214 | 0.5426 |
| OL | 0.4942 | 0.5281 | 0.5204 | 0.5362 |
| C-F-1 | 0.4786 | 0.4976 | 0.5012 | 0.4662 |
| C-F-5 | 0.4893 | 0.6074 | 0.6099 | 0.5858 |
| C-F-15 | 0.4461 | 0.5225 | 0.5482 | 0.5535 |
| C-J | 0.5157 | 0.5438 | 0.5601 | 0.5817 |

## Secondary reading: the #112 Gate-B predictor's #96-style contrasts (#101, re-tabulated, not re-scored)

- source `.local-artifacts/issue-101-decision-aware-policy-v1/compute.json`; #112 v3 LC hybrid Gate-B handoff checkpoints (fold 'all', seeds 20260908/09/10), E99 Gate-B anchors, tie-free cost; J = J-pe-1e-04 (ICLR prediction-error DP), J_dec = J-dec-1e-01; #101 guards {'G1_isolation': True, 'G2_scoring_policy': True, 'G3_chronology': True, 'G4_checkpoint_binding': True, 'G5_gate_b_replication': True}
- #96 contrasts not in the #101 publication: J-F(1,macro), J-F(1,continuous), J-T, J-D, C-J-C-F*

| inventory | contrast | estimate [95% DESCRIPTIVE] | half-width | <= 0.02 | <= design delta | #101 token |
|---|---|---|---|---|---|---|
| grid | J - F* (cross-inventory r-dagger) [F-5-micro] | -0.0388 [-0.0476, -0.0306] | 0.0085 | True | True | not_supported_by_this_experiment |
| grid | J - F_hindsight [F-15-continuous] | -0.0565 [-0.0677, -0.0458] | 0.0110 | True | True | not_supported_by_this_experiment |
| grid | J - M | -0.0006 [-0.0032, 0.0019] | 0.0025 | True | True | not_supported_by_this_experiment |
| grid | J - OL | 0.0127 [0.0098, 0.0156] | 0.0029 | True | True | not_supported_by_this_experiment |
| grid | J_dec - F* [F-5-micro] | 0.0014 [-0.0050, 0.0079] | 0.0065 | True | True | not_supported_by_this_experiment |
| grid | J_dec - F_hindsight [F-15-continuous] | -0.0163 [-0.0203, -0.0122] | 0.0041 | True | True | not_supported_by_this_experiment |
| grid | J_dec - M_dec | 0.0005 [-0.0005, 0.0018] | 0.0012 | True | True | not_supported_by_this_experiment |
| grid | J_dec - OL_dec | -0.0031 [-0.0064, 0.0003] | 0.0033 | True | True | not_supported_by_this_experiment |
| offset | J - F* (cross-inventory r-dagger) [F-5-micro] | -0.0899 [-0.1259, -0.0553] | 0.0353 | False | True | not_supported_by_this_experiment |
| offset | J - F_hindsight [F-5-micro] | -0.0899 [-0.1259, -0.0553] | 0.0353 | False | True | not_supported_by_this_experiment |
| offset | J - M | 0.0385 [0.0145, 0.0625] | 0.0240 | False | True | supported |
| offset | J - OL | 0.0798 [0.0610, 0.0982] | 0.0186 | True | True | supported |
| offset | J_dec - F* [F-5-micro] | -0.1517 [-0.1762, -0.1275] | 0.0243 | False | True | not_supported_by_this_experiment |
| offset | J_dec - F_hindsight [F-5-micro] | -0.1517 [-0.1762, -0.1275] | 0.0243 | False | True | not_supported_by_this_experiment |
| offset | J_dec - M_dec | 0.0011 [-0.0039, 0.0055] | 0.0047 | True | True | not_supported_by_this_experiment |
| offset | J_dec - OL_dec | -0.0231 [-0.0324, -0.0144] | 0.0090 | True | True | not_supported_by_this_experiment |
| angle | J - F* (cross-inventory r-dagger) [F-5-micro] | -0.0793 [-0.1055, -0.0540] | 0.0258 | False | True | not_supported_by_this_experiment |
| angle | J - F_hindsight [F-5-micro] | -0.0793 [-0.1055, -0.0540] | 0.0258 | False | True | not_supported_by_this_experiment |
| angle | J - M | -0.0022 [-0.0182, 0.0145] | 0.0164 | True | True | not_supported_by_this_experiment |
| angle | J - OL | 0.0495 [0.0336, 0.0652] | 0.0158 | True | True | supported |
| angle | J_dec - F* [F-5-micro] | -0.1569 [-0.1802, -0.1348] | 0.0227 | False | True | not_supported_by_this_experiment |
| angle | J_dec - F_hindsight [F-5-micro] | -0.1569 [-0.1802, -0.1348] | 0.0227 | False | True | not_supported_by_this_experiment |
| angle | J_dec - M_dec | 0.0033 [-0.0019, 0.0091] | 0.0055 | True | True | not_supported_by_this_experiment |
| angle | J_dec - OL_dec | -0.0225 [-0.0317, -0.0134] | 0.0091 | True | True | not_supported_by_this_experiment |
| power | J - F* (cross-inventory r-dagger) [F-5-micro] | -0.0061 [-0.0135, 0.0007] | 0.0071 | True | True | not_supported_by_this_experiment |
| power | J - F_hindsight [F-5-micro] | -0.0061 [-0.0135, 0.0007] | 0.0071 | True | True | not_supported_by_this_experiment |
| power | J - M | 0.0315 [0.0145, 0.0502] | 0.0178 | True | True | supported |
| power | J - OL | 0.0113 [0.0028, 0.0192] | 0.0082 | True | True | not_supported_by_this_experiment |
| power | J_dec - F* [F-5-micro] | -0.0194 [-0.0353, -0.0064] | 0.0145 | True | True | not_supported_by_this_experiment |
| power | J_dec - F_hindsight [F-5-micro] | -0.0194 [-0.0353, -0.0064] | 0.0145 | True | True | not_supported_by_this_experiment |
| power | J_dec - M_dec | 0.0004 [-0.0007, 0.0015] | 0.0011 | True | True | not_supported_by_this_experiment |
| power | J_dec - OL_dec | -0.0120 [-0.0248, -0.0028] | 0.0110 | True | True | not_supported_by_this_experiment |

## Compute

| arm | records | candidate rollouts | transition calls | controller calls | linear MACs | GPU s |
|---|---|---|---|---|---|---|
| F-1-continuous | 3 | 23298 | 5242050 | 0 | 7508964038400 | 256.3 |
| F-1-micro | 3 | 23298 | 5242050 | 0 | 9411870124800 | 326.3 |
| F-1-macro | 3 | 23298 | 5242050 | 0 | 7678051603200 | 288.2 |
| F-5-continuous | 3 | 23298 | 1048410 | 0 | 1501792807680 | 54.1 |
| F-5-micro | 3 | 23298 | 1048410 | 0 | 1882374024960 | 65.8 |
| F-5-macro | 3 | 23298 | 1048410 | 0 | 1535610320640 | 58.7 |
| F-15-continuous | 3 | 23298 | 349470 | 0 | 500597602560 | 18.0 |
| F-15-micro | 3 | 23298 | 349470 | 0 | 627458008320 | 22.1 |
| F-15-macro | 3 | 23298 | 349470 | 0 | 511870106880 | 19.2 |
| J | 3 | 23298 | 3401472 | 3401472 | 5182643732480 | 239.5 |
| M | 3 | 23298 | 3401200 | 3401200 | 5181668746752 | 306.2 |
| T | 3 | 23298 | 3500138 | 3500138 | 5221784640000 | 624.5 |
| D | 3 | 23298 | 3844170 | 3844170 | 5825546983936 | 650.4 |
| OL | 3 | 23298 | 3425090 | 0 | 5108393660672 | 424.5 |
| C-F-1 | 3 | 23298 | 5242050 | 0 | 9709325010000 | 195.5 |
| C-F-5 | 3 | 23298 | 1048410 | 0 | 1941865002000 | 56.9 |
| C-F-15 | 3 | 23298 | 349470 | 0 | 647288334000 | 19.1 |
| C-J | 3 | 23298 | 3101838 | 3101838 | 5844777834210 | 176.4 |

- scoring: 3802 GPU s summed over workers, 3970 s process wall, 1344 s elapsed; smoke 308 s (census of the 3186 #96 records 302 s); engine seconds 0

## Claim boundary

decision-only rollouts of the frozen ICLR #77 N1 checkpoints and controllers from the #105 F77 anchors of the sealed #109 evaluation levels to endpoint 225 on the Gate-B mixed cells; inventories never pooled; every interval DESCRIPTIVE; zero engine seconds; no training
