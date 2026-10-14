# Issue #117 ranking (confirmatory): joint vs specialist on the #104 sealed split

Frozen 2026-10-14T16:12:54Z; validation `python -u -m scripts.run_request_interference_ranking --validate`. Recipe LC3 (#112 v3 LC, the Gate-B recipe); joint = the Gate-B handoff LC hybrid (3 seeds); specialists trained here on the same #109 fit payload. Grid primary (311 mixed levels × 3 seeds); offset, angle and power DESCRIPTIVE. Intervals: level-clustered percentile bootstraps (DESCRIPTIVE). δ = 0.02. Engine seconds 0.

## Disposition

Claim: sharing all weights does not cost per-request ranking on the sealed split: the request-mean grid AUC joint - specialist, and that of the Gate-B unit F-15-macro, under LC3 at the per-request budget (the #117 LOLO T2 analogue, now confirmatory on the #104 sealed #109 evaluation split)

Rule: supported iff both the request-mean and the F-15-macro grid AUC J - S have lower bound > -0.02 (non-inferiority at delta); not_supported_by_this_experiment iff either has upper bound < -0.02; otherwise readiness_or_precision_insufficient; any failed guard turns every token into readiness_or_precision_insufficient. The #117 T2 condition (upper bound < 0) is reported next to the token

- request_mean: J − S 0.0848 [0.0742, 0.0952] → `supported`
- F-15-macro: J − S 0.0561 [0.0435, 0.0682] → `supported`
- #117 T2 reading (upper bound < 0): none
- Guards pass: True

**Disposition: `supported`**

## Per-request grid AUC, per-request budget (primary; token = δ non-inferiority of J)

| request | J | S (request) | J − S | token | D (request) | J − D | token |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1-continuous | 0.6772 [0.6548, 0.6994] | 0.5918 [0.5702, 0.6131] | 0.0854 [0.0692, 0.1016] | supported | 0.5030 [0.4806, 0.5253] | 0.1647 [0.1355, 0.1950] | supported |
| 1-micro | 0.6500 [0.6283, 0.6716] | 0.5247 [0.5058, 0.5441] | 0.1254 [0.0989, 0.1515] | supported | 0.5771 [0.5539, 0.6003] | 0.0637 [0.0469, 0.0804] | supported |
| 1-macro | 0.6661 [0.6434, 0.6893] | 0.5654 [0.5449, 0.5870] | 0.1007 [0.0759, 0.1251] | supported | 0.5830 [0.5586, 0.6075] | 0.0899 [0.0650, 0.1146] | supported |
| 5-continuous | 0.7486 [0.7286, 0.7683] | 0.6813 [0.6591, 0.7029] | 0.0673 [0.0558, 0.0784] | supported | 0.7382 [0.7194, 0.7566] | 0.0103 [-0.0038, 0.0243] | supported |
| 5-micro | 0.7446 [0.7250, 0.7642] | 0.6475 [0.6258, 0.6689] | 0.0970 [0.0851, 0.1088] | supported | 0.7377 [0.7186, 0.7564] | 0.0069 [-0.0072, 0.0211] | supported |
| 5-macro | 0.7426 [0.7238, 0.7620] | 0.6757 [0.6541, 0.6969] | 0.0668 [0.0499, 0.0832] | supported | 0.7518 [0.7333, 0.7700] | -0.0093 [-0.0235, 0.0044] | readiness_or_precision_insufficient |
| 15-continuous | 0.7622 [0.7429, 0.7811] | 0.6802 [0.6587, 0.7013] | 0.0820 [0.0734, 0.0903] | supported | 0.7545 [0.7360, 0.7728] | 0.0078 [-0.0023, 0.0179] | supported |
| 15-micro | 0.7601 [0.7413, 0.7787] | 0.6778 [0.6565, 0.6985] | 0.0823 [0.0739, 0.0905] | supported | 0.7511 [0.7327, 0.7695] | 0.0090 [-0.0011, 0.0192] | supported |
| 15-macro | 0.7444 [0.7269, 0.7620] | 0.6883 [0.6662, 0.7102] | 0.0561 [0.0435, 0.0682] | supported | 0.7578 [0.7399, 0.7755] | -0.0134 [-0.0226, -0.0043] | readiness_or_precision_insufficient |
| **request mean** | 0.7218 [0.7029, 0.7406] | 0.6370 [0.6184, 0.6553] | **0.0848 [0.0742, 0.0952]** | **supported** | 0.6844 [0.6683, 0.7001] | **0.0324 [0.0219, 0.0431]** | **supported** |

## Asymmetry (grid, per-request budget)

| group | J − S | J − D | D − S |
| --- | --- | --- | --- |
| delta_1 | 0.1038 [0.0888, 0.1185] | 0.1061 [0.0906, 0.1216] | 0.0049 [-0.0106, 0.0204] |
| delta_5 | 0.0770 [0.0639, 0.0897] | 0.0027 [-0.0110, 0.0162] | 0.0744 [0.0624, 0.0862] |
| delta_15 | 0.0735 [0.0639, 0.0826] | 0.0011 [-0.0083, 0.0107] | 0.0724 [0.0612, 0.0836] |
| alpha_continuous | 0.0782 [0.0682, 0.0880] | 0.0493 [0.0343, 0.0645] | 0.0402 [0.0254, 0.0547] |
| alpha_micro | 0.1016 [0.0907, 0.1122] | 0.0167 [0.0053, 0.0281] | 0.0933 [0.0804, 0.1058] |
| alpha_macro | 0.0746 [0.0583, 0.0903] | 0.0313 [0.0202, 0.0422] | 0.0680 [0.0569, 0.0791] |
| all | 0.0848 [0.0742, 0.0952] | 0.0324 [0.0219, 0.0431] | 0.0672 [0.0567, 0.0775] |
| Δ=1 − Δ=15 | 0.0303 [0.0160, 0.0446] | 0.1083 [0.0917, 0.1253] | -0.0853 [-0.1064, -0.0649] |
| continuous − macro | 0.0037 [-0.0066, 0.0141] | 0.0179 [0.0046, 0.0318] | -0.0278 [-0.0405, -0.0149] |

## Full budget (DESCRIPTIVE; 9000 updates per specialist / per-Δ model)

| request | J | S (full) | J − S | token | D (full) | J − D | token |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1-continuous | 0.6772 [0.6548, 0.6994] | 0.5685 [0.5503, 0.5866] | 0.1087 [0.0915, 0.1265] | supported | 0.6060 [0.5813, 0.6308] | 0.0712 [0.0571, 0.0847] | supported |
| 1-micro | 0.6500 [0.6283, 0.6716] | 0.5077 [0.4898, 0.5252] | 0.1424 [0.1173, 0.1679] | supported | 0.5607 [0.5402, 0.5817] | 0.0893 [0.0760, 0.1025] | supported |
| 1-macro | 0.6661 [0.6434, 0.6893] | 0.5014 [0.4783, 0.5243] | 0.1711 [0.1403, 0.2028] | supported | 0.6136 [0.5877, 0.6396] | 0.0525 [0.0291, 0.0756] | supported |
| 5-continuous | 0.7486 [0.7286, 0.7683] | 0.8028 [0.7926, 0.8130] | -0.0542 [-0.0715, -0.0369] | not_supported_by_this_experiment | 0.8737 [0.8605, 0.8868] | -0.1251 [-0.1405, -0.1099] | not_supported_by_this_experiment |
| 5-micro | 0.7446 [0.7250, 0.7642] | 0.6566 [0.6403, 0.6725] | 0.0880 [0.0699, 0.1065] | supported | 0.8590 [0.8460, 0.8718] | -0.1144 [-0.1304, -0.0986] | not_supported_by_this_experiment |
| 5-macro | 0.7426 [0.7238, 0.7620] | 0.8638 [0.8534, 0.8739] | -0.1213 [-0.1368, -0.1052] | not_supported_by_this_experiment | 0.8695 [0.8555, 0.8836] | -0.1269 [-0.1375, -0.1160] | not_supported_by_this_experiment |
| 15-continuous | 0.7622 [0.7429, 0.7811] | 0.8678 [0.8566, 0.8790] | -0.1056 [-0.1260, -0.0852] | not_supported_by_this_experiment | 0.8613 [0.8456, 0.8768] | -0.0991 [-0.1177, -0.0806] | not_supported_by_this_experiment |
| 15-micro | 0.7601 [0.7413, 0.7787] | 0.8292 [0.8198, 0.8386] | -0.0690 [-0.0866, -0.0512] | not_supported_by_this_experiment | 0.8725 [0.8566, 0.8882] | -0.1124 [-0.1296, -0.0947] | not_supported_by_this_experiment |
| 15-macro | 0.7444 [0.7269, 0.7620] | 0.8850 [0.8717, 0.8976] | -0.1406 [-0.1560, -0.1252] | not_supported_by_this_experiment | 0.8427 [0.8256, 0.8593] | -0.0983 [-0.1135, -0.0826] | not_supported_by_this_experiment |
| **request mean** | 0.7218 [0.7029, 0.7406] | 0.7215 [0.7125, 0.7302] | **0.0148 [-0.0007, 0.0303]** | **supported** | 0.7732 [0.7601, 0.7866] | **-0.0515 [-0.0603, -0.0428]** | **not_supported_by_this_experiment** |

| group | J − S | J − D | D − S |
| --- | --- | --- | --- |
| delta_1 | 0.1402 [0.1201, 0.1611] | 0.0710 [0.0568, 0.0848] | 0.0776 [0.0557, 0.0999] |
| delta_5 | -0.0292 [-0.0441, -0.0139] | -0.1222 [-0.1357, -0.1086] | 0.0930 [0.0834, 0.1026] |
| delta_15 | -0.1051 [-0.1224, -0.0879] | -0.1032 [-0.1201, -0.0862] | -0.0018 [-0.0114, 0.0080] |
| alpha_continuous | -0.0170 [-0.0326, -0.0014] | -0.0510 [-0.0606, -0.0414] | 0.0340 [0.0252, 0.0427] |
| alpha_micro | 0.0538 [0.0371, 0.0704] | -0.0458 [-0.0574, -0.0343] | 0.0996 [0.0896, 0.1099] |
| alpha_macro | -0.0204 [-0.0378, -0.0026] | -0.0576 [-0.0661, -0.0488] | 0.0288 [0.0160, 0.0421] |
| all | 0.0148 [-0.0007, 0.0303] | -0.0515 [-0.0603, -0.0428] | 0.0536 [0.0446, 0.0630] |
| Δ=1 − Δ=15 | 0.2250 [0.2063, 0.2440] | 0.1743 [0.1467, 0.2012] | 0.0893 [0.0665, 0.1124] |
| continuous − macro | 0.0364 [0.0233, 0.0490] | 0.0066 [-0.0011, 0.0142] | 0.0262 [0.0139, 0.0381] |

## Controls (grid)

No-model action prior (Gate-B rule, fit-verdicts.json): AUC 0.7203 [0.6965, 0.7442].

| ranker (request mean) | AUC | − prior | token (δ superiority) | wrong-anchor AUC | right − wrong | stability |
| --- | --- | --- | --- | --- | --- | --- |
| J | 0.7218 [0.7029, 0.7406] | 0.0015 [-0.0246, 0.0265] | readiness_or_precision_insufficient | 0.6400 [0.6215, 0.6584] | 0.0817 [0.0753, 0.0883] | 0.7542 [0.7479, 0.7606] |
| S/request | 0.6370 [0.6184, 0.6553] | -0.0833 [-0.1134, -0.0532] | not_supported_by_this_experiment | 0.5699 [0.5526, 0.5873] | 0.0671 [0.0583, 0.0762] | 0.5679 [0.5607, 0.5750] |
| D/request | 0.6844 [0.6683, 0.7001] | -0.0359 [-0.0627, -0.0093] | not_supported_by_this_experiment | 0.5663 [0.5541, 0.5788] | 0.1181 [0.1081, 0.1280] | 0.2976 [0.2900, 0.3053] |
| S/full | 0.7215 [0.7125, 0.7302] | 0.0012 [-0.0213, 0.0234] | readiness_or_precision_insufficient | 0.5682 [0.5595, 0.5769] | 0.1533 [0.1435, 0.1629] | 0.2952 [0.2847, 0.3051] |
| D/full | 0.7732 [0.7601, 0.7866] | 0.0529 [0.0288, 0.0770] | supported | 0.5918 [0.5802, 0.6035] | 0.1814 [0.1718, 0.1910] | 0.5349 [0.5253, 0.5441] |

Wrong anchors: the 480 fit-split anchors over the 16 shared grid candidates. Per request (per-request budget):

| request | J right − wrong | S right − wrong | D right − wrong | J − prior | S − prior |
| --- | --- | --- | --- | --- | --- |
| 1-continuous | 0.0016 [0.0007, 0.0026] | 0.0267 [0.0130, 0.0400] | -0.0220 [-0.0446, 0.0003] | -0.0431 [-0.0719, -0.0151] | -0.1285 [-0.1601, -0.0975] |
| 1-micro | 0.0015 [-0.0002, 0.0036] | 0.0126 [-0.0059, 0.0319] | 0.0032 [-0.0116, 0.0178] | -0.0703 [-0.1034, -0.0372] | -0.1956 [-0.2250, -0.1656] |
| 1-macro | 0.0008 [0.0005, 0.0012] | -0.0038 [-0.0146, 0.0065] | 0.0150 [-0.0011, 0.0311] | -0.0542 [-0.0766, -0.0322] | -0.1549 [-0.1891, -0.1204] |
| 5-continuous | 0.0988 [0.0901, 0.1076] | 0.0935 [0.0820, 0.1053] | 0.1695 [0.1548, 0.1843] | 0.0283 [0.0019, 0.0538] | -0.0390 [-0.0708, -0.0068] |
| 5-micro | 0.1016 [0.0931, 0.1104] | 0.0718 [0.0619, 0.0818] | 0.1711 [0.1563, 0.1859] | 0.0243 [-0.0033, 0.0508] | -0.0728 [-0.1062, -0.0391] |
| 5-macro | 0.1050 [0.0967, 0.1132] | 0.0923 [0.0803, 0.1046] | 0.1791 [0.1658, 0.1927] | 0.0223 [-0.0007, 0.0448] | -0.0446 [-0.0761, -0.0129] |
| 15-continuous | 0.1443 [0.1321, 0.1565] | 0.0998 [0.0869, 0.1130] | 0.1805 [0.1673, 0.1939] | 0.0419 [0.0142, 0.0690] | -0.0401 [-0.0715, -0.0085] |
| 15-micro | 0.1498 [0.1372, 0.1627] | 0.1011 [0.0885, 0.1140] | 0.1800 [0.1663, 0.1939] | 0.0398 [0.0125, 0.0669] | -0.0425 [-0.0743, -0.0108] |
| 15-macro | 0.1322 [0.1216, 0.1430] | 0.1097 [0.0958, 0.1241] | 0.1829 [0.1704, 0.1954] | 0.0241 [-0.0008, 0.0487] | -0.0320 [-0.0635, 0.0001] |

## Descriptive inventories (request mean)

| inventory | cells | prior | J | J − S (request) | J − D (request) | J − S (full) | J − D (full) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grid | 933 | 0.7203 [0.6965, 0.7442] | 0.7218 [0.7029, 0.7406] | 0.0848 [0.0742, 0.0952] | 0.0324 [0.0219, 0.0431] | 0.0148 [-0.0007, 0.0303] | -0.0515 [-0.0603, -0.0428] |
| offset | 228 | 0.5000 [0.5000, 0.5000] | 0.7161 [0.6996, 0.7316] | 0.0933 [0.0704, 0.1153] | 0.1547 [0.1181, 0.1902] | 0.0046 [-0.0168, 0.0251] | -0.0575 [-0.0723, -0.0430] |
| angle | 234 | 0.8119 [0.7778, 0.8450] | 0.7373 [0.7229, 0.7516] | 0.1068 [0.0889, 0.1241] | 0.1583 [0.1302, 0.1859] | 0.0071 [-0.0077, 0.0213] | -0.0519 [-0.0617, -0.0425] |
| power | 141 | 0.5000 [0.5000, 0.5000] | 0.8023 [0.7671, 0.8390] | 0.0678 [0.0542, 0.0813] | 0.0530 [0.0342, 0.0714] | 0.1329 [0.1055, 0.1634] | 0.0130 [-0.0018, 0.0296] |

## Training (LC3, #109 fit payload, fold 'all')

| unit | updates | usable seeds | typed failures | GPU s |
| --- | --- | --- | --- | --- |
| D1/request | 3000 | 2/3 | 20260910: final_checkpoint_excursion | 2366 |
| D1/full | 9000 | 3/3 | — | 7162 |
| D5/request | 3000 | 3/3 | — | 610 |
| D5/full | 9000 | 3/3 | — | 1859 |
| D15/request | 3000 | 3/3 | — | 316 |
| D15/full | 9000 | 3/3 | — | 998 |
| S-1-continuous/request | 1000 | 3/3 | — | 700 |
| S-1-continuous/full | 9000 | 3/3 | — | 6269 |
| S-1-micro/request | 1000 | 3/3 | — | 835 |
| S-1-micro/full | 9000 | 3/3 | — | 7478 |
| S-1-macro/request | 1000 | 3/3 | — | 815 |
| S-1-macro/full | 9000 | 2/3 | 20260908: final_checkpoint_excursion | 7295 |
| S-5-continuous/request | 1000 | 3/3 | — | 166 |
| S-5-continuous/full | 9000 | 3/3 | — | 1534 |
| S-5-micro/request | 1000 | 3/3 | — | 216 |
| S-5-micro/full | 9000 | 3/3 | — | 1973 |
| S-5-macro/request | 1000 | 3/3 | — | 212 |
| S-5-macro/full | 9000 | 3/3 | — | 1921 |
| S-15-continuous/request | 1000 | 3/3 | — | 80 |
| S-15-continuous/full | 9000 | 3/3 | — | 741 |
| S-15-micro/request | 1000 | 3/3 | — | 110 |
| S-15-micro/full | 9000 | 3/3 | — | 1052 |
| S-15-macro/request | 1000 | 3/3 | — | 104 |
| S-15-macro/full | 9000 | 3/3 | — | 1034 |

Typed failures are terminal (never retried or replaced); a cell enters a paired contrast only when every piece of it is defined, so a failed seed drops that seed's cells from the contrasts that read it.

## Guards

- G1_isolation: True
- G2_scoring_policy: True
- G3_chronology: True
- G4_checkpoint_binding: True
- G5_replication: True
- G6_training_integrity: True
- G7_nonfinite: True

## Compute

GPU seconds 54899 (training 45845, scoring 8869); engine seconds 0; 4 x RTX 5090 (one process per GPU).
