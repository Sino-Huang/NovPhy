# Issue #105: independent axes vs hierarchy on the #104 sealed #109 evaluation split

Plan frozen 2026-10-14T11:43:40Z after a passed smoke, before training and scoring; scored once in processes that did not train, under `deterministic_scoring()`. Validation: `python -u -m scripts.run_hierarchy_baselines --validate`. Intervals: level-clustered percentile bootstrap (10 000 draws), DESCRIPTIVE; margin δ = 0.02; zero engine seconds.

**Disposition on "independent axes add value over tied axes" (grid): not_supported_by_this_experiment.** J-J_diag not_supported_by_this_experiment; J-J_monotone not_supported_by_this_experiment; J-H* not_supported_by_this_experiment. H* = H-5-micro (cross-fit).

## Acceptance contrasts (AUC difference, paired by level and seed)

| contrast | grid | offset | angle | power |
| --- | --- | --- | --- | --- |
| J-J_diag | -0.0068 [-0.0164, 0.0026] not_supported_by_this_experiment | -0.0218 [-0.0616, 0.0141] not_supported_by_this_experiment | -0.0246 [-0.0580, 0.0074] not_supported_by_this_experiment | 0.0203 [-0.0010, 0.0462] readiness_or_precision_insufficient |
| J-J_monotone | 0.0028 [0.0014, 0.0043] not_supported_by_this_experiment | -0.0022 [-0.0057, 0.0013] not_supported_by_this_experiment | -0.0039 [-0.0085, 0.0004] not_supported_by_this_experiment | -0.0017 [-0.0040, 0.0008] not_supported_by_this_experiment |
| J-H* | -0.0006 [-0.0188, 0.0170] not_supported_by_this_experiment | 0.0572 [-0.0202, 0.1316] readiness_or_precision_insufficient | 0.1041 [0.0474, 0.1577] supported | 0.0033 [-0.0119, 0.0175] not_supported_by_this_experiment |
| J-J_anti | 0.0056 [0.0015, 0.0098] not_supported_by_this_experiment | 0.1371 [0.1111, 0.1630] supported | 0.1019 [0.0798, 0.1244] supported | 0.0180 [0.0071, 0.0289] readiness_or_precision_insufficient |
| J_diag-J_anti | 0.0124 [0.0025, 0.0223] readiness_or_precision_insufficient | 0.1590 [0.1276, 0.1921] supported | 0.1265 [0.0993, 0.1538] supported | -0.0023 [-0.0271, 0.0184] not_supported_by_this_experiment |
| H*-J_diag | -0.0062 [-0.0189, 0.0070] not_supported_by_this_experiment | -0.0790 [-0.1365, -0.0185] not_supported_by_this_experiment | -0.1288 [-0.1734, -0.0822] not_supported_by_this_experiment | 0.0169 [-0.0057, 0.0441] readiness_or_precision_insufficient |
| J-F_hindsight | -0.0565 [-0.0677, -0.0458] not_supported_by_this_experiment | -0.0899 [-0.1259, -0.0553] not_supported_by_this_experiment | -0.0793 [-0.1055, -0.0540] not_supported_by_this_experiment | -0.0061 [-0.0135, 0.0007] not_supported_by_this_experiment |
| J-F* | -0.0388 [-0.0476, -0.0306] not_supported_by_this_experiment | -0.0899 [-0.1259, -0.0553] not_supported_by_this_experiment | -0.0793 [-0.1055, -0.0540] not_supported_by_this_experiment | -0.0061 [-0.0135, 0.0007] not_supported_by_this_experiment |
| J_dec-J_dec_diag | 0.0514 [0.0358, 0.0665] supported | 0.0712 [0.0500, 0.0930] supported | 0.0625 [0.0417, 0.0837] supported | 0.0451 [0.0047, 0.0897] supported |
| J_dec-J_dec_monotone | 0.0001 [0.0000, 0.0002] not_supported_by_this_experiment | 0.0000 [0.0000, 0.0000] not_supported_by_this_experiment | 0.0000 [0.0000, 0.0000] not_supported_by_this_experiment | 0.0000 [0.0000, 0.0000] not_supported_by_this_experiment |
| J_dec-H* | 0.0397 [0.0303, 0.0486] supported | -0.0045 [-0.0368, 0.0272] readiness_or_precision_insufficient | 0.0265 [-0.0006, 0.0533] readiness_or_precision_insufficient | -0.0099 [-0.0224, 0.0009] not_supported_by_this_experiment |
| J_dec-J_dec_anti | -0.0163 [-0.0203, -0.0122] not_supported_by_this_experiment | -0.1274 [-0.1402, -0.1147] not_supported_by_this_experiment | -0.0991 [-0.1122, -0.0861] not_supported_by_this_experiment | -0.0100 [-0.0239, 0.0012] not_supported_by_this_experiment |

H* and F* are cross-fit per inventory (best mean member-mean AUC over the other three inventories); F_hindsight is the in-inventory best fixed request. Per inventory: grid: H* H-5-micro, F* F-5-micro, F_hindsight F-15-continuous; offset: H* H-5-micro, F* F-5-micro, F_hindsight F-5-micro; angle: H* H-5-micro, F* F-5-micro, F_hindsight F-5-micro; power: H* H-5-micro, F* F-5-micro, F_hindsight F-5-micro.

## Ranking AUC per arm (tie-free cost)

| arm | carrier | grid | offset | angle | power | MACs / candidate (grid) | tied pair share (grid) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F-1-continuous | E99 | 0.6772 [0.6548, 0.6994] | 0.6833 [0.6491, 0.7158] | 0.7260 [0.6959, 0.7548] | 0.7930 [0.7552, 0.8319] | 322.3M | 0.0000 |
| F-1-micro | E99 | 0.6500 [0.6283, 0.6716] | 0.7083 [0.6833, 0.7325] | 0.7243 [0.7046, 0.7426] | 0.7912 [0.7524, 0.8302] | 404.0M | 0.0000 |
| F-1-macro | E99 | 0.6661 [0.6434, 0.6893] | 0.5058 [0.4583, 0.5527] | 0.5823 [0.5399, 0.6247] | 0.7818 [0.7485, 0.8166] | 329.6M | 0.0000 |
| F-5-continuous | E99 | 0.7486 [0.7286, 0.7683] | 0.8186 [0.7952, 0.8406] | 0.8347 [0.8137, 0.8553] | 0.8103 [0.7704, 0.8509] | 64.5M | 0.0000 |
| F-5-micro | E99 | 0.7446 [0.7250, 0.7642] | 0.8421 [0.8193, 0.8636] | 0.8557 [0.8373, 0.8735] | 0.8137 [0.7761, 0.8525] | 80.8M | 0.0000 |
| F-5-macro | E99 | 0.7426 [0.7238, 0.7620] | 0.6095 [0.5838, 0.6329] | 0.6593 [0.6398, 0.6776] | 0.8131 [0.7829, 0.8441] | 65.9M | 0.0000 |
| F-15-continuous | E99 | 0.7622 [0.7429, 0.7811] | 0.8179 [0.7974, 0.8373] | 0.7979 [0.7804, 0.8154] | 0.8043 [0.7650, 0.8437] | 21.5M | 0.0000 |
| F-15-micro | E99 | 0.7601 [0.7413, 0.7787] | 0.7922 [0.7720, 0.8115] | 0.7765 [0.7588, 0.7934] | 0.8078 [0.7713, 0.8457] | 26.9M | 0.0000 |
| F-15-macro | E99 | 0.7444 [0.7269, 0.7620] | 0.6670 [0.6503, 0.6827] | 0.6791 [0.6663, 0.6912] | 0.8058 [0.7729, 0.8396] | 22.0M | 0.0000 |
| C* (LC continuous C-F-15) | E99 | 0.7834 [0.7676, 0.7992] | 0.7748 [0.7524, 0.7962] | 0.7590 [0.7422, 0.7752] | 0.8068 [0.7711, 0.8440] | 27.8M | 0.0000 |
| J (ICLR, #101 pe 1e-4) | E99 | 0.7057 [0.6846, 0.7267] | 0.7522 [0.7039, 0.7978] | 0.7765 [0.7402, 0.8115] | 0.8076 [0.7695, 0.8463] | 184.0M | 0.0000 |
| J_diag | E99 | 0.7125 [0.6922, 0.7329] | 0.7740 [0.7363, 0.8086] | 0.8011 [0.7671, 0.8321] | 0.7873 [0.7415, 0.8324] | 110.5M | 0.0000 |
| J_monotone | E99 | 0.7029 [0.6818, 0.7241] | 0.7544 [0.7061, 0.8004] | 0.7804 [0.7440, 0.8157] | 0.8093 [0.7713, 0.8481] | 186.7M | 0.0000 |
| J_anti (control) | E99 | 0.7001 [0.6790, 0.7212] | 0.6151 [0.5833, 0.6455] | 0.6746 [0.6451, 0.7039] | 0.7896 [0.7547, 0.8262] | 221.3M | 0.0000 |
| J_dec (#101 dec 1e-1) | E99 | 0.7460 [0.7277, 0.7637] | 0.6904 [0.6737, 0.7062] | 0.6988 [0.6827, 0.7147] | 0.7943 [0.7555, 0.8331] | 22.7M | 0.0000 |
| J_dec_diag | E99 | 0.6946 [0.6786, 0.7111] | 0.6192 [0.5934, 0.6437] | 0.6364 [0.6147, 0.6573] | 0.7492 [0.7220, 0.7776] | 26.6M | 0.0000 |
| J_dec_monotone | E99 | 0.7459 [0.7277, 0.7637] | 0.6904 [0.6737, 0.7062] | 0.6988 [0.6827, 0.7147] | 0.7943 [0.7555, 0.8331] | 22.7M | 0.0000 |
| J_dec_anti (control) | E99 | 0.7622 [0.7429, 0.7811] | 0.8179 [0.7974, 0.8373] | 0.7979 [0.7804, 0.8154] | 0.8043 [0.7650, 0.8437] | 22.1M | 0.0000 |
| H-1-continuous | E99 | 0.5080 [0.4953, 0.5205] | 0.5351 [0.5140, 0.5557] | 0.5078 [0.4830, 0.5320] | 0.5267 [0.4886, 0.5651] | 410.8M | 0.0000 |
| H-5-micro | E99 | 0.7063 [0.6875, 0.7251] | 0.6950 [0.6606, 0.7288] | 0.6723 [0.6472, 0.6978] | 0.8042 [0.7707, 0.8387] | 58.5M | 0.0000 |
| H-15-macro | E99 | 0.4159 [0.4021, 0.4299] | 0.5577 [0.5456, 0.5698] | 0.5328 [0.5224, 0.5431] | 0.2829 [0.2428, 0.3217] | 7.9M | 0.0000 |
| TAWM-1 | F77 (#71) | 0.4587 [0.4414, 0.4763] | 0.6314 [0.5967, 0.6684] | 0.5829 [0.5564, 0.6105] | 0.4636 [0.4174, 0.5123] | 416.7M | 0.0000 |
| TAWM-5 | F77 (#71) | 0.6007 [0.5757, 0.6260] | 0.5937 [0.5496, 0.6364] | 0.6377 [0.6001, 0.6736] | 0.7885 [0.7541, 0.8238] | 83.3M | 0.0000 |
| TAWM-15 | F77 (#71) | 0.6103 [0.5846, 0.6361] | 0.4968 [0.4412, 0.5526] | 0.5773 [0.5278, 0.6278] | 0.7894 [0.7559, 0.8232] | 27.8M | 0.0000 |
| VLWM-1 | F77 (#71) | 0.4480 [0.4257, 0.4695] | 0.4318 [0.4171, 0.4461] | 0.4519 [0.4386, 0.4647] | 0.3155 [0.2715, 0.3582] | 416.7M | 0.0000 |
| VLWM-5 | F77 (#71) | 0.5520 [0.5423, 0.5618] | 0.5456 [0.5281, 0.5632] | 0.5327 [0.5179, 0.5475] | 0.5617 [0.5320, 0.5917] | 83.3M | 0.0000 |
| VLWM-15 | F77 (#71) | 0.6422 [0.6187, 0.6658] | 0.4481 [0.4053, 0.4907] | 0.5209 [0.4834, 0.5589] | 0.7781 [0.7460, 0.8109] | 27.8M | 0.0000 |
| TAWM-adaptive | F77 (#71) | 0.4497 [0.4304, 0.4691] | 0.6033 [0.5714, 0.6350] | 0.5510 [0.5233, 0.5793] | 0.5238 [0.4821, 0.5643] | 236.0M | 0.0000 |
| no-model action prior | — | 0.7203 [0.6965, 0.7442] | 0.5000 [0.5000, 0.5000] | 0.8119 [0.7778, 0.8450] | 0.5000 [0.5000, 0.5000] | n/a | 0.0022 |

| arm (#78) | reading |
| --- | --- |
| VLWM-adaptive | unsupported: #78 trained a horizon controller for TAWM only (its adaptive evaluation arm); no VLWM controller exists |
| TAWM/VLWM diagonal, monotone and anti-diagonal arms | unsupported: the #78 ports are continuous-only (ContinuousDynamics, horizon conditioning only): no description axis alpha |
| TAWM/VLWM wrong-anchor control | unsupported: their carrier is the #71 parser's; the fit-split wrong-anchor pool exists only as E99 carriers |

## Request schedules (grid, pooled over seeds)

| arm | modal request | diagonal step share | non-modal step share | cells with differing schedules |
| --- | --- | --- | --- | --- |
| J | 1-macro | 0.135 | 0.210 | 760 / 933 |
| J_diag | 1-continuous | 1.000 | 0.358 | 673 / 933 |
| J_monotone | 1-macro | 0.134 | 0.206 | 719 / 933 |
| J_anti | 1-macro | 0.065 | 0.085 | 604 / 933 |
| J_dec | 15-continuous | 0.382 | 0.480 | 732 / 933 |
| J_dec_diag | 15-macro | 1.000 | 0.169 | 311 / 933 |
| J_dec_monotone | 15-continuous | 0.382 | 0.480 | 732 / 933 |
| J_dec_anti | 15-continuous | 0.000 | 0.006 | 18 / 933 |
| TAWM-adaptive | C-F-1 | n/a | 0.171 | 713 / 933 |

## Rollout error at t = 225 (grid candidates; engine_position_mse (r100) at t = 225, observed frames; lower is better)

| arm | error | nonfinite endpoints |
| --- | --- | --- |
| F-1-continuous | 0.0309 [0.0298, 0.0320] | 0 |
| F-1-micro | 0.0303 [0.0291, 0.0314] | 0 |
| F-1-macro | 0.0281 [0.0271, 0.0291] | 0 |
| F-5-continuous | 0.0383 [0.0361, 0.0405] | 0 |
| F-5-micro | 0.0375 [0.0353, 0.0397] | 0 |
| F-5-macro | 0.0446 [0.0420, 0.0474] | 0 |
| F-15-continuous | 0.0415 [0.0386, 0.0445] | 0 |
| F-15-micro | 0.0419 [0.0389, 0.0450] | 0 |
| F-15-macro | 0.0474 [0.0440, 0.0508] | 0 |
| C* (LC continuous C-F-15) | 0.0670 [0.0617, 0.0723] | 0 |
| J (ICLR, #101 pe 1e-4) | 0.0317 [0.0303, 0.0331] | 0 |
| J_diag | 0.0361 [0.0343, 0.0381] | 0 |
| J_monotone | 0.0314 [0.0300, 0.0328] | 0 |
| J_anti (control) | 0.0342 [0.0327, 0.0358] | 0 |
| J_dec (#101 dec 1e-1) | 0.0436 [0.0406, 0.0466] | 0 |
| J_dec_diag | 0.0466 [0.0435, 0.0498] | 0 |
| J_dec_monotone | 0.0436 [0.0406, 0.0466] | 0 |
| J_dec_anti (control) | 0.0415 [0.0386, 0.0445] | 0 |
| H-1-continuous | 4353.6829 [4280.4567, 4431.9443] | 0 |
| H-5-micro | 0.1096 [0.1054, 0.1139] | 0 |
| H-15-macro | 186.1743 [185.2112, 187.1723] | 0 |
| TAWM-1 | 5312.4079 [5213.8670, 5413.8780] | 0 |
| TAWM-5 | 0.4533 [0.4430, 0.4640] | 0 |
| TAWM-15 | 0.1632 [0.1604, 0.1660] | 0 |
| VLWM-1 | 12240.2657 [12103.5814, 12382.9482] | 0 |
| VLWM-5 | 107.9323 [106.4310, 109.4456] | 0 |
| VLWM-15 | 0.2801 [0.2736, 0.2866] | 0 |
| TAWM-adaptive | 763.3045 [697.4312, 829.4319] | 0 |

| contrast (paired by level) | estimate |
| --- | --- |
| J-J_diag (J − J_diag) | -0.0044 [-0.0053, -0.0036] |
| J-J_monotone (J − J_monotone) | 0.0003 [0.0002, 0.0004] |
| J-H* (J − H-5-micro) | -0.0779 [-0.0812, -0.0747] |
| J_diag-H* (J_diag − H-5-micro) | -0.0735 [-0.0761, -0.0708] |
| H-1-continuous-F-1-continuous (H-1-continuous − F-1-continuous) | 4353.6519 [4280.4267, 4431.9131] |
| H-5-micro-F-5-micro (H-5-micro − F-5-micro) | 0.0721 [0.0696, 0.0747] |
| H-15-macro-F-15-macro (H-15-macro − F-15-macro) | 186.1269 [185.1619, 187.1255] |
| TAWM-1-F-15-continuous (TAWM-1 − F-15-continuous) | 5312.3663 [5213.8253, 5413.8347] |
| TAWM-5-F-15-continuous (TAWM-5 − F-15-continuous) | 0.4118 [0.4003, 0.4237] |
| TAWM-15-F-15-continuous (TAWM-15 − F-15-continuous) | 0.1217 [0.1178, 0.1255] |
| VLWM-1-F-15-continuous (VLWM-1 − F-15-continuous) | 12240.2241 [12103.5373, 12382.9077] |
| VLWM-5-F-15-continuous (VLWM-5 − F-15-continuous) | 107.8908 [106.3893, 109.4064] |
| VLWM-15-F-15-continuous (VLWM-15 − F-15-continuous) | 0.2386 [0.2320, 0.2452] |
| TAWM-adaptive-F-15-continuous (TAWM-adaptive − F-15-continuous) | 763.2630 [697.3882, 829.3912] |

## Controls

- No-model action prior (grid): 0.7203 [0.6965, 0.7442]; J − prior -0.0146 [-0.0388, 0.0088]; H* − prior -0.0140 [-0.0440, 0.0167].
- Wrong-anchor control of J_diag (480 fit anchors): right 0.7125 [0.6922, 0.7329], wrong 0.6466 [0.6265, 0.6665], right − wrong 0.0659 [0.0568, 0.0751] (supported); stability 0.7206 [0.7108, 0.7300].
- Wrong-anchor control of H-1-continuous (480 fit anchors): right 0.5080 [0.4953, 0.5205], wrong 0.4864 [0.4786, 0.4941], right − wrong 0.0216 [0.0120, 0.0313] (supported); stability 0.6121 [0.6011, 0.6227].
- Wrong-anchor control of H-5-micro (480 fit anchors): right 0.7063 [0.6875, 0.7251], wrong 0.5700 [0.5544, 0.5857], right − wrong 0.1363 [0.1234, 0.1499] (supported); stability 0.4612 [0.4480, 0.4743].
- Wrong-anchor control of H-15-macro (480 fit anchors): right 0.4159 [0.4021, 0.4299], wrong 0.4159 [0.4021, 0.4299], right − wrong 0.0000 [0.0000, 0.0000] (not_supported_by_this_experiment); stability 1.0000 [1.0000, 1.0000].

## Hierarchical predictor

- Three levels (1, continuous) / (5, micro) / (15, macro), width 202, 1,840,322 parameters vs 1,837,690 for the LC hybrid; #112 v3 LC recipe, 9000 updates (3000 per level), #109 fit-split payload of the v3 retrain.
- seed 20260908: usable True (complete True, skipped 0, excursion False), 1937 GPU s.
- seed 20260909: usable True (complete True, skipped 0, excursion False), 1949 GPU s.
- seed 20260910: usable True (complete True, skipped 0, excursion False), 1838 GPU s.

## Guards

- G1_isolation: True
- G2_scoring_policy: True
- G3_chronology: True
- G4_checkpoint_binding: True
- G5_replication_101: True
- G6_restriction_integrity: True
- G7_nonfinite: True

Compute: 10386 GPU s, 11982 process wall s (2 x RTX 5090 (GPUs 2, 3; one process per GPU)); 0 engine s.
