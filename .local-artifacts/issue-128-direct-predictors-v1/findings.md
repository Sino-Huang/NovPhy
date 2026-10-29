# Issue #128 findings: direct success predictors (A1, A2, A2-native, A4; A3 when scored) vs U

Plan frozen 2026-10-25T09:26:24Z. Score = success logit (Q for A3); cost = -score. Intervals: member-clustered percentile bootstraps (10 000 draws, PCG64 7201), DESCRIPTIVE. delta = 0.02.

## Global guards

| guard | pass |
| --- | --- |
| G_isolation | True |
| G_chronology | True |
| G_scoring_policy | True |
| G_records | True |

## Arms (grid, 311 mixed cells x 3 seeds)

| arm | role | params | guards | grid AUC [CI] | X − U [CI] | token X−U | token U−X | disposition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | primary (program set) | 11,309,889 | pass | 0.9946 [0.9917, 0.9971] | +0.2502 [+0.2327, +0.2675] | supported | not_supported_by_this_experiment | 1 |
| A2 | primary (program set) | 1,838,781 | pass | 0.9757 [0.9692, 0.9815] | +0.2313 [+0.2131, +0.2493] | supported | not_supported_by_this_experiment | 1 |
| A2-native | secondary | 11,333,771 | pass | 0.9740 [0.9666, 0.9805] | +0.2296 [+0.2114, +0.2474] | supported | not_supported_by_this_experiment | 1 |
| A3-img | secondary | 11,633,920 | fail (G_fid) | 0.9870 [0.9814, 0.9918] | +0.2426 [+0.2244, +0.2610] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | 4 |
| A3-sym | secondary | 11,662,208 | fail (G_fid) | 0.9700 [0.9597, 0.9791] | +0.2256 [+0.2057, +0.2455] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | 4 |
| A4 | primary (program set) | 12,751,957 | pass | 0.9945 [0.9922, 0.9965] | +0.2501 [+0.2327, +0.2674] | supported | not_supported_by_this_experiment | 1 |

## Within-family contrasts

| contrast | estimate | token | reverse token | guards |
| --- | --- | --- | --- | --- |
| A2 - A1 | -0.0190 [-0.0251, -0.0134] | not_supported_by_this_experiment | readiness_or_precision_insufficient | True |
| A4 - A1 | -0.0002 [-0.0021, +0.0021] | not_supported_by_this_experiment | not_supported_by_this_experiment | True |

## Secondary contrasts per arm

| arm | contrast | estimate | token | reverse token |
| --- | --- | --- | --- | --- |
| A1 | X-U | +0.2502 [+0.2327, +0.2675] | supported | not_supported_by_this_experiment |
| A1 | X-C* | +0.2113 [+0.1954, +0.2268] | supported | not_supported_by_this_experiment |
| A1 | X-P | +0.2743 [+0.2505, +0.2980] | supported | not_supported_by_this_experiment |
| A1 | X-J_dec | +0.2487 [+0.2311, +0.2666] | supported | not_supported_by_this_experiment |
| A2 | X-U | +0.2313 [+0.2131, +0.2493] | supported | not_supported_by_this_experiment |
| A2 | X-C* | +0.1923 [+0.1760, +0.2084] | supported | not_supported_by_this_experiment |
| A2 | X-P | +0.2554 [+0.2318, +0.2787] | supported | not_supported_by_this_experiment |
| A2 | X-J_dec | +0.2297 [+0.2113, +0.2484] | supported | not_supported_by_this_experiment |
| A2-native | X-U | +0.2296 [+0.2114, +0.2474] | supported | not_supported_by_this_experiment |
| A2-native | X-C* | +0.1906 [+0.1745, +0.2066] | supported | not_supported_by_this_experiment |
| A2-native | X-P | +0.2537 [+0.2297, +0.2772] | supported | not_supported_by_this_experiment |
| A2-native | X-J_dec | +0.2280 [+0.2096, +0.2463] | supported | not_supported_by_this_experiment |
| A3-img | X-U | +0.2426 [+0.2244, +0.2610] | readiness_or_precision_insufficient | readiness_or_precision_insufficient |
| A3-img | X-C* | +0.2036 [+0.1870, +0.2204] | readiness_or_precision_insufficient | readiness_or_precision_insufficient |
| A3-img | X-P | +0.2667 [+0.2428, +0.2905] | readiness_or_precision_insufficient | readiness_or_precision_insufficient |
| A3-img | X-J_dec | +0.2410 [+0.2227, +0.2600] | readiness_or_precision_insufficient | readiness_or_precision_insufficient |
| A3-sym | X-U | +0.2256 [+0.2057, +0.2455] | readiness_or_precision_insufficient | readiness_or_precision_insufficient |
| A3-sym | X-C* | +0.1866 [+0.1679, +0.2052] | readiness_or_precision_insufficient | readiness_or_precision_insufficient |
| A3-sym | X-P | +0.2497 [+0.2250, +0.2739] | readiness_or_precision_insufficient | readiness_or_precision_insufficient |
| A3-sym | X-J_dec | +0.2240 [+0.2036, +0.2445] | readiness_or_precision_insufficient | readiness_or_precision_insufficient |
| A4 | X-U | +0.2501 [+0.2327, +0.2674] | supported | not_supported_by_this_experiment |
| A4 | X-C* | +0.2111 [+0.1954, +0.2267] | supported | not_supported_by_this_experiment |
| A4 | X-P | +0.2742 [+0.2504, +0.2977] | supported | not_supported_by_this_experiment |
| A4 | X-J_dec | +0.2485 [+0.2309, +0.2665] | supported | not_supported_by_this_experiment |

## Compute

| arm | params | training GPU-s per seed | MACs / candidate | wall / decision (s) |
| --- | --- | --- | --- | --- |
| A1 | 11,309,889 | 20260908: 269, 20260909: 487, 20260910: 486 | 805390848 | 0.0043 |
| A2 | 1,838,781 | 20260908: 48, 20260909: 58, 20260910: 63 | 1426102 | 0.0035 |
| A2-native | 11,333,771 | 20260908: 54, 20260909: 68, 20260910: 75 | 8742400 | 0.0070 |
| A3-img | 11,633,920 | 20260908: 6452, 20260909: 4482, 20260910: 4962 | 45219360 | 0.0037 |
| A3-sym | 11,662,208 | 20260908: 742, 20260909: 792, 20260910: 737 | 53686560 | 0.0017 |
| A4 | 12,751,957 | 20260908: 2425, 20260909: 2429, 20260910: 2450 | 805390848 | 0.0043 |

Engine time (CPU game-server worker-hours; A3 native-task fidelity): 6.30 h; A3-img 3.07 h; A3-sym 3.23 h.

## No-model nearest-fit references (DESCRIPTIVE, never in a disposition, no tokens)

Rule: score of a candidate = the engine verdict (1 / 0) of the identical launch executed in the nearest fit level; cost = -score (eb.score_rows); distance ties -> lowest fit member id; if that fit level did not execute the launch, the next fit level in (distance, member) order that did. NN-px / NN-car are defined on grid and angle only; the fit split never executed offset or power launches, so those inventory columns are empty by construction.

| reference | pooled grid AUC [CI] |
| --- | --- |
| NN-px | 0.9039 [0.8838, 0.9231] |
| NN-car | 0.7773 [0.7502, 0.8038] |

| contrast | estimate |
| --- | --- |
| A1 - NN-px | +0.0907 [+0.0719, +0.1105] |
| A2 - NN-px | +0.0717 [+0.0530, +0.0916] |
| A2-native - NN-px | +0.0701 [+0.0512, +0.0898] |
| A3-img - NN-px | +0.0830 [+0.0644, +0.1028] |
| A3-sym - NN-px | +0.0660 [+0.0466, +0.0864] |
| A4 - NN-px | +0.0905 [+0.0718, +0.1103] |
| U - NN-px | -0.1595 [-0.1852, -0.1332] |
| A1 - NN-car | +0.2174 [+0.1910, +0.2442] |
| A2 - NN-car | +0.1984 [+0.1735, +0.2237] |
| A2-native - NN-car | +0.1967 [+0.1719, +0.2219] |
| A3-img - NN-car | +0.2097 [+0.1826, +0.2372] |
| A3-sym - NN-car | +0.1927 [+0.1661, +0.2199] |
| A4 - NN-car | +0.2172 [+0.1908, +0.2441] |
| U - NN-car | -0.0329 [-0.0638, -0.0012] |

## Novelty (P-novelty, DESCRIPTIVE: no tokens, never part of a disposition)

#110 novelty levels of #107's in-scope cells, mixed levels only; chronology pass True.

| ranker | novelty level | novel AUC [CI] | paired normal AUC [CI] | novel − normal [CI] | paired level pairs |
| --- | --- | --- | --- | --- | --- |
| A1 | 1 | 0.8721 [0.7740, 0.9470] | 0.8849 [0.7641, 0.9698] | -0.0001 [-0.0068, +0.0078] | 17 |
| A1 | 2 | 0.7231 [0.6105, 0.8213] | 0.7623 [0.6679, 0.8445] | -0.1347 [-0.2890, -0.0270] | 6 |
| A1 | 3 | 0.6095 [0.4095, 0.8243] | 0.7883 [0.7417, 0.8371] | -0.0175 [-0.0175, -0.0175] | 1 |
| A1 | 4 | 0.4791 [0.4145, 0.5403] | 0.4500 [0.2333, 0.7500] | +0.0889 [-0.2680, +0.3564] | 4 |
| A1 | 5 | 0.5902 [0.5101, 0.6690] | 0.8872 [0.8263, 0.9381] | -0.3298 [-0.4428, -0.2049] | 15 |
| A1 | 6 | 0.3128 [0.2480, 0.3859] | 0.6811 [0.5827, 0.7766] | -0.3317 [-0.4469, -0.2189] | 15 |
| A1 | 7 | 0.5244 [0.3899, 0.6604] | 0.5852 [0.4682, 0.6989] | +0.0268 [-0.2179, +0.2544] | 14 |
| A1 | 1–7 pooled | 0.5776 [0.5269, 0.6272] | 0.7495 [0.7006, 0.7949] | -0.1392 [-0.2104, -0.0651] | 72 |
| A1 | 0 (all normal sides) | n/a | 0.7739 [0.7347, 0.8113] | n/a | n/a |
| A2 | 1 | 0.8503 [0.7549, 0.9314] | 0.9283 [0.8711, 0.9760] | -0.0064 [-0.0178, +0.0018] | 17 |
| A2 | 2 | 0.6507 [0.5425, 0.7521] | 0.5531 [0.4494, 0.6765] | +0.0317 [-0.2361, +0.2327] | 6 |
| A2 | 3 | 0.6238 [0.5159, 0.7222] | 0.5709 [0.4861, 0.6543] | +0.0992 [+0.0992, +0.0992] | 1 |
| A2 | 4 | 0.4610 [0.3736, 0.5494] | 0.5111 [0.1778, 0.8444] | +0.0821 [-0.3802, +0.5444] | 4 |
| A2 | 5 | 0.4523 [0.3512, 0.5524] | 0.8414 [0.7526, 0.9178] | -0.3724 [-0.5333, -0.2014] | 15 |
| A2 | 6 | 0.2515 [0.1855, 0.3204] | 0.7382 [0.6513, 0.8154] | -0.4359 [-0.5415, -0.3319] | 15 |
| A2 | 7 | 0.4440 [0.3323, 0.5620] | 0.6641 [0.5635, 0.7637] | -0.1306 [-0.2840, +0.0204] | 14 |
| A2 | 1–7 pooled | 0.5172 [0.4661, 0.5681] | 0.7400 [0.6941, 0.7842] | -0.1867 [-0.2629, -0.1122] | 72 |
| A2 | 0 (all normal sides) | n/a | 0.7420 [0.7019, 0.7803] | n/a | n/a |
| A2-native | 1 | 0.8682 [0.7775, 0.9430] | 0.9373 [0.8888, 0.9765] | -0.0022 [-0.0117, +0.0063] | 17 |
| A2-native | 2 | 0.6795 [0.5801, 0.7653] | 0.6285 [0.5407, 0.7297] | -0.0251 [-0.2729, +0.1380] | 6 |
| A2-native | 3 | 0.6271 [0.5567, 0.7271] | 0.5287 [0.4314, 0.6342] | -0.0738 [-0.0738, -0.0738] | 1 |
| A2-native | 4 | 0.4696 [0.3877, 0.5526] | 0.6111 [0.2278, 0.9222] | -0.0703 [-0.5115, +0.4089] | 4 |
| A2-native | 5 | 0.4406 [0.3425, 0.5375] | 0.8713 [0.7902, 0.9387] | -0.4258 [-0.5786, -0.2611] | 15 |
| A2-native | 6 | 0.2020 [0.1343, 0.2740] | 0.7462 [0.6512, 0.8317] | -0.5002 [-0.6177, -0.3835] | 15 |
| A2-native | 7 | 0.4571 [0.3208, 0.5933] | 0.6943 [0.5791, 0.8003] | -0.1160 [-0.2800, +0.0556] | 14 |
| A2-native | 1–7 pooled | 0.5174 [0.4627, 0.5706] | 0.7629 [0.7165, 0.8070] | -0.2230 [-0.3015, -0.1448] | 72 |
| A2-native | 0 (all normal sides) | n/a | 0.7707 [0.7314, 0.8076] | n/a | n/a |
| A4 | 1 | 0.8121 [0.6737, 0.9277] | 0.8184 [0.6578, 0.9501] | -0.0068 [-0.0218, +0.0042] | 17 |
| A4 | 2 | 0.6628 [0.5538, 0.7646] | 0.6716 [0.4345, 0.8716] | -0.1406 [-0.2456, -0.0500] | 6 |
| A4 | 3 | 0.6274 [0.5237, 0.7422] | 0.8357 [0.6190, 0.9917] | -0.1079 [-0.1079, -0.1079] | 1 |
| A4 | 4 | 0.4779 [0.3640, 0.5888] | 0.5278 [0.1722, 0.8667] | -0.1049 [-0.4848, +0.4255] | 4 |
| A4 | 5 | 0.5974 [0.5197, 0.6724] | 0.8500 [0.7531, 0.9311] | -0.3185 [-0.4299, -0.1874] | 15 |
| A4 | 6 | 0.3012 [0.2312, 0.3803] | 0.7048 [0.5926, 0.8075] | -0.3558 [-0.4639, -0.2318] | 15 |
| A4 | 7 | 0.5797 [0.4056, 0.7500] | 0.5119 [0.4258, 0.6008] | +0.2322 [-0.0666, +0.4876] | 14 |
| A4 | 1–7 pooled | 0.5690 [0.5127, 0.6237] | 0.7198 [0.6616, 0.7740] | -0.1160 [-0.2000, -0.0278] | 72 |
| A4 | 0 (all normal sides) | n/a | 0.7512 [0.7042, 0.7967] | n/a | n/a |
| U | 1 | 0.6505 [0.5581, 0.7371] | 0.6662 [0.5801, 0.7459] | +0.0050 [-0.0066, +0.0221] | 17 |
| U | 2 | 0.4677 [0.3971, 0.5452] | 0.7064 [0.6296, 0.7909] | -0.2992 [-0.4595, -0.1770] | 6 |
| U | 3 | 0.5706 [0.3706, 0.7675] | 0.7892 [0.7611, 0.8200] | -0.6246 [-0.6246, -0.6246] | 1 |
| U | 4 | 0.5142 [0.4309, 0.5915] | 0.5833 [0.3333, 0.8333] | -0.0762 [-0.3735, +0.1978] | 4 |
| U | 5 | 0.4542 [0.3594, 0.5474] | 0.7131 [0.6314, 0.7882] | -0.2551 [-0.4096, -0.0963] | 15 |
| U | 6 | 0.2747 [0.2114, 0.3450] | 0.5969 [0.5099, 0.6853] | -0.2809 [-0.3514, -0.2090] | 15 |
| U | 7 | 0.4755 [0.4110, 0.5403] | 0.5825 [0.4881, 0.6775] | -0.0274 [-0.1761, +0.1069] | 14 |
| U | 1–7 pooled | 0.4743 [0.4354, 0.5119] | 0.6562 [0.6170, 0.6935] | -0.1537 [-0.2127, -0.0960] | 72 |
| U | 0 (all normal sides) | n/a | 0.6506 [0.6168, 0.6839] | n/a | n/a |
| C* | 1 | 0.6860 [0.5940, 0.7696] | 0.7176 [0.6403, 0.7893] | +0.0020 [-0.0140, +0.0213] | 17 |
| C* | 2 | 0.4810 [0.4078, 0.5575] | 0.6832 [0.5672, 0.8035] | -0.2569 [-0.4270, -0.1308] | 6 |
| C* | 3 | 0.4844 [0.3066, 0.6622] | 0.7490 [0.6889, 0.8026] | -0.6810 [-0.6810, -0.6810] | 1 |
| C* | 4 | 0.4634 [0.3637, 0.5514] | 0.5056 [0.1222, 0.8889] | -0.0636 [-0.4067, +0.2818] | 4 |
| C* | 5 | 0.4394 [0.3448, 0.5325] | 0.7222 [0.6395, 0.7979] | -0.2726 [-0.4261, -0.1105] | 15 |
| C* | 6 | 0.3073 [0.2480, 0.3679] | 0.6303 [0.5426, 0.7165] | -0.2824 [-0.3696, -0.1977] | 15 |
| C* | 7 | 0.3701 [0.2774, 0.4664] | 0.5829 [0.4905, 0.6761] | -0.1016 [-0.2430, +0.0318] | 14 |
| C* | 1–7 pooled | 0.4551 [0.4148, 0.4950] | 0.6656 [0.6244, 0.7048] | -0.1693 [-0.2293, -0.1092] | 72 |
| C* | 0 (all normal sides) | n/a | 0.6654 [0.6308, 0.6988] | n/a | n/a |
| P | 1 | 0.6746 [0.5765, 0.7710] | 0.6583 [0.5669, 0.7535] | +0.0021 [-0.0067, +0.0130] | 17 |
| P | 2 | 0.7796 [0.6465, 0.8905] | 0.8730 [0.7222, 0.9815] | -0.2032 [-0.4399, -0.0804] | 6 |
| P | 3 | 0.7345 [0.6258, 0.8444] | 0.9100 [0.7917, 0.9933] | -0.0131 [-0.0131, -0.0131] | 1 |
| P | 4 | 0.5366 [0.4696, 0.5975] | 0.3667 [0.0667, 0.7667] | +0.2560 [-0.1103, +0.5359] | 4 |
| P | 5 | 0.8122 [0.7226, 0.8920] | 0.8353 [0.7809, 0.8858] | -0.0299 [-0.0620, -0.0066] | 15 |
| P | 6 | 0.2516 [0.1723, 0.3361] | 0.7425 [0.6226, 0.8478] | -0.4402 [-0.5673, -0.3124] | 15 |
| P | 7 | 0.6052 [0.4324, 0.7748] | 0.6971 [0.6082, 0.7852] | -0.0044 [-0.2681, +0.2255] | 14 |
| P | 1–7 pooled | 0.6022 [0.5459, 0.6559] | 0.7501 [0.7005, 0.7948] | -0.1012 [-0.1781, -0.0274] | 72 |
| P | 0 (all normal sides) | n/a | 0.7645 [0.7269, 0.8001] | n/a | n/a |

Not novelty-scored (failed guards): A3-img (G_fid), A3-sym (G_fid)
