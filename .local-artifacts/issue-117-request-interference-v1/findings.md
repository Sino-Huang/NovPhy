# Issue #117 — interference between requests in the shared predictor (items 1–3, NovPhy)

Frozen plan 2026-10-11T03:48:41Z (before training and scoring). Validation: `python -u -m scripts.run_request_interference --validate`.
Guards pass: True (G1 True, G2 True, G3 True, G4 True, chronology True, caps True). Engine seconds 0. LOLO: EXPLORATORY; held-out: DESCRIPTIVE.

**Decision (pre-declared v4 trigger, LC3 at the per-request budget): keep_v3.** T1 False ([]), T2 False ([]), T3 False ([]). v3 stays the Gate-B recipe: no pre-declared interference criterion fired.

## 1.5× disposition (joint / specialist rollout error at t = 225, LOLO)

| recipe / budget | token | 1-continuous | 1-micro | 1-macro | 5-continuous | 5-micro | 5-macro | 15-continuous | 15-micro | 15-macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B/request | supported | supported | supported | supported | supported | supported | supported | supported | supported | supported |
| B/full | readiness_or_precision_insufficient | supported | supported | supported | supported | supported | supported | insufficient | supported | supported |
| LC1/request | supported | supported | supported | supported | supported | supported | supported | supported | supported | supported |
| LC1/full | readiness_or_precision_insufficient | insufficient | insufficient | insufficient | insufficient | supported | insufficient | supported | supported | supported |
| LC3/request | supported | supported | supported | supported | supported | supported | supported | supported | supported | supported |
| LC3/full | readiness_or_precision_insufficient | supported | supported | supported | insufficient | supported | insufficient | supported | supported | supported |

## Item 1. Per-request ratios at t = 225 (geometric mean over lineages [95% interval])

### B/request

| request | joint / specialist | joint / per-Δ | per-Δ / specialist |
| --- | --- | --- | --- |
| F-1-continuous | 0.382 [0.317, 0.458] | 0.476 [0.390, 0.585] | 0.804 [0.679, 0.971] |
| F-1-micro | 0.350 [0.298, 0.413] | 0.502 [0.414, 0.603] | 0.697 [0.582, 0.832] |
| F-1-macro | 0.375 [0.318, 0.447] | 0.514 [0.435, 0.620] | 0.730 [0.629, 0.847] |
| F-5-continuous | 0.005 [0.004, 0.007] | 0.015 [0.010, 0.023] | 0.350 [0.263, 0.453] |
| F-5-micro | 0.005 [0.004, 0.007] | 0.013 [0.009, 0.020] | 0.387 [0.292, 0.510] |
| F-5-macro | 0.004 [0.003, 0.005] | 0.042 [0.028, 0.063] | 0.090 [0.064, 0.128] |
| F-15-continuous | 0.481 [0.395, 0.587] | 0.773 [0.651, 0.918] | 0.622 [0.516, 0.745] |
| F-15-micro | 0.429 [0.367, 0.502] | 0.738 [0.632, 0.858] | 0.581 [0.496, 0.674] |
| F-15-macro | 0.213 [0.186, 0.243] | 0.684 [0.603, 0.770] | 0.311 [0.268, 0.356] |

### B/full

| request | joint / specialist | joint / per-Δ | per-Δ / specialist |
| --- | --- | --- | --- |
| F-1-continuous | 0.625 [0.520, 0.746] | 0.616 [0.534, 0.694] | 1.015 [0.897, 1.137] |
| F-1-micro | 0.609 [0.532, 0.708] | 0.581 [0.501, 0.661] | 1.049 [0.888, 1.260] |
| F-1-macro | 0.678 [0.585, 0.786] | 0.675 [0.596, 0.750] | 1.005 [0.891, 1.126] |
| F-5-continuous | 0.173 [0.091, 0.329] | 0.554 [0.388, 0.800] | 0.313 [0.178, 0.528] |
| F-5-micro | 0.239 [0.128, 0.432] | 0.513 [0.346, 0.760] | 0.466 [0.249, 0.803] |
| F-5-macro | 1.106 [0.863, 1.418] | 1.093 [0.875, 1.409] | 1.011 [0.814, 1.273] |
| F-15-continuous | 1.220 [0.956, 1.552] | 1.307 [1.146, 1.479] | 0.933 [0.741, 1.148] |
| F-15-micro | 1.158 [0.926, 1.450] | 1.266 [1.122, 1.416] | 0.915 [0.728, 1.147] |
| F-15-macro | 0.800 [0.737, 0.866] | 1.010 [0.918, 1.103] | 0.792 [0.697, 0.902] |

### LC1/request

| request | joint / specialist | joint / per-Δ | per-Δ / specialist |
| --- | --- | --- | --- |
| F-1-continuous | 0.002 [0.001, 0.003] | 2.37e-04 [2.37e-04, 2.37e-04] | 5.874 [5.874, 5.874] |
| F-1-micro | 0.002 [0.001, 0.003] | 1.65e-04 [1.65e-04, 1.65e-04] | 10.081 [10.081, 10.081] |
| F-1-macro | 9.43e-04 [7.43e-04, 0.001] | 1.77e-04 [1.77e-04, 1.77e-04] | 7.075 [7.075, 7.075] |
| F-5-continuous | 0.439 [0.306, 0.628] | 0.624 [0.416, 0.935] | 0.703 [0.601, 0.815] |
| F-5-micro | 0.365 [0.263, 0.505] | 0.621 [0.429, 0.888] | 0.588 [0.539, 0.636] |
| F-5-macro | 0.536 [0.358, 0.821] | 0.699 [0.477, 1.025] | 0.766 [0.662, 0.881] |
| F-15-continuous | 0.600 [0.456, 0.775] | 0.731 [0.530, 1.002] | 0.821 [0.697, 0.997] |
| F-15-micro | 0.513 [0.406, 0.641] | 0.742 [0.533, 1.024] | 0.692 [0.560, 0.872] |
| F-15-macro | 0.673 [0.507, 0.892] | 0.859 [0.596, 1.244] | 0.784 [0.616, 0.977] |

### LC1/full

| request | joint / specialist | joint / per-Δ | per-Δ / specialist |
| --- | --- | --- | --- |
| F-1-continuous | n/a | n/a | n/a |
| F-1-micro | n/a | n/a | n/a |
| F-1-macro | 1.682 [1.682, 1.682] | n/a | n/a |
| F-5-continuous | 1.223 [0.876, 1.680] | 1.094 [0.725, 1.608] | 1.119 [0.884, 1.445] |
| F-5-micro | 0.102 [0.060, 0.177] | 1.131 [0.782, 1.587] | 0.090 [0.046, 0.188] |
| F-5-macro | 1.234 [0.850, 1.779] | 1.189 [0.856, 1.625] | 1.037 [0.803, 1.345] |
| F-15-continuous | 0.788 [0.543, 1.131] | 1.043 [0.739, 1.437] | 0.755 [0.619, 0.895] |
| F-15-micro | 0.864 [0.601, 1.245] | 1.057 [0.752, 1.456] | 0.817 [0.687, 0.959] |
| F-15-macro | 0.941 [0.630, 1.467] | 1.198 [0.817, 1.772] | 0.785 [0.655, 0.934] |

### LC3/request

| request | joint / specialist | joint / per-Δ | per-Δ / specialist |
| --- | --- | --- | --- |
| F-1-continuous | 0.280 [0.140, 0.529] | 0.309 [0.150, 0.600] | 0.908 [0.333, 2.360] |
| F-1-micro | 0.108 [0.052, 0.225] | 0.300 [0.151, 0.577] | 0.361 [0.112, 1.166] |
| F-1-macro | 0.308 [0.130, 0.664] | 0.321 [0.143, 0.695] | 0.959 [0.269, 3.376] |
| F-5-continuous | 0.501 [0.398, 0.612] | 0.713 [0.550, 0.886] | 0.703 [0.601, 0.815] |
| F-5-micro | 0.367 [0.298, 0.448] | 0.697 [0.543, 0.868] | 0.527 [0.461, 0.596] |
| F-5-macro | 0.605 [0.448, 0.796] | 0.790 [0.596, 0.998] | 0.766 [0.662, 0.881] |
| F-15-continuous | 0.655 [0.512, 0.816] | 0.836 [0.649, 1.072] | 0.783 [0.652, 0.948] |
| F-15-micro | 0.563 [0.457, 0.701] | 0.846 [0.658, 1.083] | 0.666 [0.542, 0.816] |
| F-15-macro | 0.656 [0.519, 0.825] | 0.875 [0.679, 1.150] | 0.750 [0.583, 0.925] |

### LC3/full

| request | joint / specialist | joint / per-Δ | per-Δ / specialist |
| --- | --- | --- | --- |
| F-1-continuous | 0.182 [0.109, 0.332] | 0.214 [0.087, 0.503] | 0.852 [0.335, 2.364] |
| F-1-micro | 0.068 [0.029, 0.142] | 0.227 [0.095, 0.522] | 0.299 [0.087, 1.134] |
| F-1-macro | 0.167 [0.058, 0.413] | 0.292 [0.115, 0.728] | 0.571 [0.111, 2.556] |
| F-5-continuous | 1.396 [1.119, 1.684] | 1.248 [0.936, 1.626] | 1.119 [0.884, 1.445] |
| F-5-micro | 0.114 [0.067, 0.197] | 1.269 [0.975, 1.621] | 0.090 [0.046, 0.188] |
| F-5-macro | 1.284 [1.052, 1.553] | 1.344 [1.055, 1.703] | 0.955 [0.753, 1.200] |
| F-15-continuous | 0.860 [0.649, 1.112] | 1.000 [0.741, 1.346] | 0.859 [0.678, 1.096] |
| F-15-micro | 0.939 [0.689, 1.265] | 1.007 [0.746, 1.362] | 0.933 [0.732, 1.177] |
| F-15-macro | 0.917 [0.723, 1.159] | 1.028 [0.781, 1.359] | 0.892 [0.682, 1.143] |

## Item 1. Rollout error at t = 225 (LOLO mean over lineages)

### B

| request | J | D req | D full | S req | S full |
| --- | --- | --- | --- | --- | --- |
| F-1-continuous | 3841 | 8165 | 6147 | 10059 | 6150 |
| F-1-micro | 3793 | 7811 | 6451 | 10784 | 6334 |
| F-1-macro | 4039 | 7961 | 5959 | 11016 | 5979 |
| F-5-continuous | 0.2587 | 21.4427 | 0.5912 | 56.5923 | 2.9226 |
| F-5-micro | 0.2842 | 25.3764 | 0.7175 | 60.8559 | 2.3456 |
| F-5-macro | 0.2858 | 8.8542 | 0.2789 | 77.6361 | 0.2682 |
| F-15-continuous | 0.0347 | 0.0482 | 0.0281 | 0.0797 | 0.0319 |
| F-15-micro | 0.0345 | 0.0498 | 0.0284 | 0.0823 | 0.0316 |
| F-15-macro | 0.0271 | 0.0400 | 0.0275 | 0.1199 | 0.0332 |

### LC1

| request | J | D req | D full | S req | S full |
| --- | --- | --- | --- | --- | --- |
| F-1-continuous | 0.0710 | 265.9702 | n/a | 35.5092 | n/a |
| F-1-micro | 0.0698 | 423.8580 | n/a | 34.2356 | n/a |
| F-1-macro | 0.0754 | 343.3448 | n/a | 73.6424 | 0.0261 |
| F-5-continuous | 0.0591 | 0.0853 | 0.0504 | 0.1208 | 0.0425 |
| F-5-micro | 0.0574 | 0.0866 | 0.0494 | 0.1450 | 0.7928 |
| F-5-macro | 0.0641 | 0.0843 | 0.0506 | 0.1094 | 0.0494 |
| F-15-continuous | 0.0354 | 0.0480 | 0.0340 | 0.0582 | 0.0470 |
| F-15-micro | 0.0362 | 0.0483 | 0.0343 | 0.0688 | 0.0419 |
| F-15-macro | 0.0438 | 0.0466 | 0.0339 | 0.0621 | 0.0428 |

### LC3

| request | J | D req | D full | S req | S full |
| --- | --- | --- | --- | --- | --- |
| F-1-continuous | 0.0607 | 0.5066 | 0.8895 | 0.4227 | 0.4861 |
| F-1-micro | 0.0607 | 0.4145 | 0.8086 | 0.9282 | 2.0341 |
| F-1-macro | 0.0744 | 0.4612 | 0.8016 | 0.7211 | 2.0097 |
| F-5-continuous | 0.0619 | 0.0853 | 0.0504 | 0.1208 | 0.0425 |
| F-5-micro | 0.0612 | 0.0866 | 0.0494 | 0.1614 | 0.7928 |
| F-5-macro | 0.0679 | 0.0843 | 0.0506 | 0.1094 | 0.0519 |
| F-15-continuous | 0.0382 | 0.0462 | 0.0419 | 0.0582 | 0.0470 |
| F-15-micro | 0.0389 | 0.0462 | 0.0422 | 0.0683 | 0.0419 |
| F-15-macro | 0.0398 | 0.0450 | 0.0408 | 0.0621 | 0.0428 |

## Item 1. LOLO grid AUC (tie-free) and joint − specialist

### B (11 lineages, 33 cells)

| request | J | S req | S full | J − S req | J − S full |
| --- | --- | --- | --- | --- | --- |
| F-1-continuous | 0.5598 [0.5013, 0.6204] | 0.4968 [0.4196, 0.5759] | 0.4569 [0.3601, 0.5419] | 0.0630 [-0.0306, 0.1574] | 0.1029 [-0.0196, 0.2466] |
| F-1-micro | 0.5505 [0.4877, 0.6109] | 0.5468 [0.4950, 0.6001] | 0.4931 [0.3918, 0.5919] | 0.0037 [-0.0890, 0.0965] | 0.0574 [-0.0481, 0.1680] |
| F-1-macro | 0.5304 [0.4916, 0.5695] | 0.5046 [0.4651, 0.5486] | 0.5654 [0.4722, 0.6649] | 0.0258 [-0.0217, 0.0682] | -0.0351 [-0.1431, 0.0711] |
| F-5-continuous | 0.4763 [0.4385, 0.5134] | 0.4721 [0.4057, 0.5382] | 0.4213 [0.3422, 0.4892] | 0.0043 [-0.0733, 0.0903] | 0.0551 [-0.0015, 0.1113] |
| F-5-micro | 0.4944 [0.4455, 0.5424] | 0.4870 [0.4489, 0.5253] | 0.5072 [0.4033, 0.6045] | 0.0074 [-0.0525, 0.0665] | -0.0128 [-0.1091, 0.0825] |
| F-5-macro | 0.4803 [0.4316, 0.5290] | 0.5007 [0.4675, 0.5336] | 0.5952 [0.5230, 0.6620] | -0.0204 [-0.0706, 0.0377] | -0.1149 [-0.1969, -0.0133] |
| F-15-continuous | 0.5294 [0.4876, 0.5746] | 0.5483 [0.4384, 0.6582] | 0.4992 [0.3985, 0.5842] | -0.0189 [-0.0957, 0.0571] | 0.0302 [-0.0507, 0.1350] |
| F-15-micro | 0.5211 [0.4828, 0.5622] | 0.5084 [0.4775, 0.5424] | 0.5360 [0.4313, 0.6345] | 0.0127 [-0.0123, 0.0393] | -0.0149 [-0.1323, 0.1020] |
| F-15-macro | 0.4861 [0.4172, 0.5529] | 0.5625 [0.5020, 0.6427] | 0.4499 [0.3657, 0.5232] | -0.0764 [-0.1411, -0.0135] | 0.0361 [-0.0937, 0.1756] |
| request-mean | | | | 0.0001 [-0.0335, 0.0378] | 0.0116 [-0.0212, 0.0478] |

### LC1 (11 lineages, 33 cells)

| request | J | S req | S full | J − S req | J − S full |
| --- | --- | --- | --- | --- | --- |
| F-1-continuous | 0.5459 [0.4131, 0.6623] | 0.5511 [0.4504, 0.6530] | n/a | 0.0283 [-0.1045, 0.1780] | n/a |
| F-1-micro | 0.5700 [0.4594, 0.6647] | 0.5351 [0.4332, 0.6187] | n/a | 0.0007 [-0.1232, 0.1211] | n/a |
| F-1-macro | 0.5490 [0.4290, 0.6577] | 0.5035 [0.4237, 0.5879] | 0.7333 [0.7333, 0.7333] | 0.0165 [-0.1038, 0.1324] | 0.1333 [0.1333, 0.1333] |
| F-5-continuous | 0.6230 [0.5109, 0.7326] | 0.6314 [0.5391, 0.7160] | 0.6494 [0.5595, 0.7382] | -0.0008 [-0.0621, 0.0738] | -0.0327 [-0.1617, 0.0887] |
| F-5-micro | 0.6101 [0.5074, 0.7066] | 0.6097 [0.5147, 0.6984] | 0.6227 [0.5076, 0.7170] | -0.0009 [-0.0340, 0.0367] | 0.0119 [-0.0861, 0.1065] |
| F-5-macro | 0.6516 [0.5441, 0.7545] | 0.6152 [0.5190, 0.7055] | 0.6884 [0.6295, 0.7448] | 0.0538 [-0.0143, 0.1435] | -0.0647 [-0.1757, 0.0372] |
| F-15-continuous | 0.6815 [0.6012, 0.7598] | 0.6444 [0.5372, 0.7448] | 0.6978 [0.6444, 0.7555] | 0.0320 [-0.0190, 0.0789] | 0.0151 [-0.0862, 0.1083] |
| F-15-micro | 0.6725 [0.5923, 0.7475] | 0.6553 [0.5636, 0.7382] | 0.6659 [0.5839, 0.7419] | 0.0202 [-0.0635, 0.0834] | -0.0113 [-0.0913, 0.0754] |
| F-15-macro | 0.6831 [0.5847, 0.7734] | 0.6483 [0.5428, 0.7468] | 0.7454 [0.6765, 0.8133] | 0.0302 [-0.0250, 0.0813] | -0.0800 [-0.2126, 0.0425] |
| request-mean | | | | 0.0200 [-0.0166, 0.0569] | n/a |

### LC3 (11 lineages, 33 cells)

| request | J | S req | S full | J − S req | J − S full |
| --- | --- | --- | --- | --- | --- |
| F-1-continuous | 0.5047 [0.3823, 0.6183] | 0.4975 [0.3685, 0.6320] | 0.5236 [0.4549, 0.5861] | 0.0072 [-0.1144, 0.1206] | -0.0189 [-0.1284, 0.0896] |
| F-1-micro | 0.5113 [0.4119, 0.6028] | 0.5232 [0.4323, 0.6139] | 0.4873 [0.3898, 0.5776] | -0.0119 [-0.1091, 0.0765] | 0.0240 [-0.1043, 0.1560] |
| F-1-macro | 0.5248 [0.4105, 0.6348] | 0.5620 [0.4477, 0.6824] | 0.4330 [0.3240, 0.5358] | -0.0372 [-0.1128, 0.0325] | 0.0918 [0.0217, 0.1623] |
| F-5-continuous | 0.5921 [0.5090, 0.6513] | 0.6314 [0.5391, 0.7160] | 0.6494 [0.5595, 0.7382] | -0.0392 [-0.1136, 0.0385] | -0.0573 [-0.1714, 0.0463] |
| F-5-micro | 0.5859 [0.5143, 0.6470] | 0.6325 [0.5319, 0.7185] | 0.6227 [0.5076, 0.7170] | -0.0465 [-0.1141, 0.0219] | -0.0368 [-0.1231, 0.0594] |
| F-5-macro | 0.6310 [0.5587, 0.7105] | 0.6152 [0.5190, 0.7055] | 0.7003 [0.6200, 0.7851] | 0.0158 [-0.0595, 0.0860] | -0.0693 [-0.1908, 0.0409] |
| F-15-continuous | 0.6861 [0.6261, 0.7475] | 0.6444 [0.5372, 0.7448] | 0.6978 [0.6444, 0.7555] | 0.0417 [-0.0390, 0.1263] | -0.0118 [-0.0739, 0.0480] |
| F-15-micro | 0.6918 [0.6331, 0.7524] | 0.6452 [0.5587, 0.7279] | 0.6659 [0.5839, 0.7419] | 0.0467 [-0.0338, 0.1377] | 0.0259 [-0.0538, 0.1086] |
| F-15-macro | 0.7245 [0.6452, 0.8072] | 0.6483 [0.5428, 0.7468] | 0.7454 [0.6765, 0.8133] | 0.0763 [0.0277, 0.1237] | -0.0209 [-0.1361, 0.1023] |
| request-mean | | | | 0.0059 [-0.0504, 0.0621] | -0.0081 [-0.0776, 0.0583] |

Every inventory (request-mean paired AUC, J − S / J − D / D − S):

| inventory | recipe / budget | J − S | J − D | D − S |
| --- | --- | --- | --- | --- |
| grid | B/request | 0.0001 [-0.0335, 0.0378] | 0.0152 [-0.0141, 0.0430] | -0.0151 [-0.0371, 0.0057] |
| grid | B/full | 0.0116 [-0.0212, 0.0478] | 0.0327 [0.0025, 0.0623] | -0.0212 [-0.0474, 0.0065] |
| grid | LC1/request | 0.0200 [-0.0166, 0.0569] | n/a | 0.0317 [0.0317, 0.0317] |
| grid | LC1/full | n/a | n/a | n/a |
| grid | LC3/request | 0.0059 [-0.0504, 0.0621] | -0.0023 [-0.0628, 0.0620] | 0.0082 [-0.0159, 0.0337] |
| grid | LC3/full | -0.0081 [-0.0776, 0.0583] | -0.0333 [-0.0899, 0.0169] | 0.0252 [-0.0055, 0.0518] |
| offset | B/request | 0.0117 [-0.0142, 0.0364] | -0.0049 [-0.0370, 0.0235] | 0.0167 [-0.0290, 0.0617] |
| offset | B/full | -0.0312 [-0.0815, 0.0151] | 0.0157 [-0.0167, 0.0451] | -0.0469 [-0.0975, 0.0160] |
| offset | LC1/request | 0.0775 [0.0429, 0.1012] | n/a | 0.0000 [0.0000, 0.0000] |
| offset | LC1/full | n/a | n/a | n/a |
| offset | LC3/request | 0.0994 [0.0327, 0.1914] | 0.0809 [-0.0130, 0.1809] | 0.0185 [-0.0272, 0.0827] |
| offset | LC3/full | -0.1019 [-0.1704, -0.0302] | -0.1262 [-0.1846, -0.0509] | 0.0244 [-0.0272, 0.0735] |
| angle | B/request | 0.0459 [-0.0189, 0.1276] | 0.0230 [-0.0414, 0.1020] | 0.0229 [-0.0264, 0.0722] |
| angle | B/full | 0.0341 [-0.0281, 0.1099] | 0.0050 [-0.0519, 0.0618] | 0.0291 [-0.0200, 0.0667] |
| angle | LC1/request | 0.0337 [-0.0061, 0.0785] | n/a | n/a |
| angle | LC1/full | n/a | n/a | n/a |
| angle | LC3/request | -0.0109 [-0.0449, 0.0296] | -0.0329 [-0.0840, 0.0181] | 0.0220 [-0.0247, 0.0772] |
| angle | LC3/full | -0.0630 [-0.1123, 0.0160] | -0.0804 [-0.1322, -0.0285] | 0.0173 [-0.0340, 0.0692] |
| power | B/request | 0.0057 [-0.0351, 0.0591] | 0.0239 [0.0077, 0.0402] | -0.0182 [-0.0596, 0.0232] |
| power | B/full | -0.0539 [-0.0986, -0.0162] | 0.0777 [0.0066, 0.1319] | -0.1317 [-0.1669, -0.0997] |
| power | LC1/request | 0.1064 [0.0715, 0.1470] | n/a | 0.0635 [0.0635, 0.0635] |
| power | LC1/full | n/a | n/a | n/a |
| power | LC3/request | 0.0382 [-0.0648, 0.1458] | 0.0616 [-0.0456, 0.1965] | -0.0235 [-0.0784, 0.0351] |
| power | LC3/full | 0.0508 [-0.0564, 0.1579] | 0.0315 [-0.0703, 0.1267] | 0.0193 [-0.0339, 0.0725] |

## Item 2. Asymmetry by Δ and by description (joint / specialist at t = 225; grid AUC J − S)

| recipe / budget | group | rollout ratio | grid AUC J − S |
| --- | --- | --- | --- |
| B/request | delta_1 | 0.369 [0.320, 0.422] | 0.0308 [-0.0331, 0.0945] |
| B/request | delta_5 | 0.005 [0.004, 0.006] | -0.0029 [-0.0583, 0.0563] |
| B/request | delta_15 | 0.353 [0.309, 0.403] | -0.0275 [-0.0591, -0.0035] |
| B/request | alpha_continuous | 0.099 [0.086, 0.115] | 0.0161 [-0.0520, 0.0918] |
| B/request | alpha_micro | 0.092 [0.081, 0.107] | 0.0079 [-0.0198, 0.0385] |
| B/request | alpha_macro | 0.067 [0.061, 0.074] | -0.0237 [-0.0419, -0.0075] |
| B/request | all | 0.085 [0.076, 0.096] | 0.0001 [-0.0335, 0.0378] |
| B/request | Δ1 vs Δ15 (ratio of ratios; AUC difference) | 1.045 [0.828, 1.327] | 0.0583 [-0.0011, 0.1178] |
| B/full | delta_1 | 0.636 [0.569, 0.723] | 0.0418 [-0.0175, 0.1038] |
| B/full | delta_5 | 0.358 [0.233, 0.556] | -0.0242 [-0.0832, 0.0432] |
| B/full | delta_15 | 1.042 [0.889, 1.219] | 0.0171 [-0.0741, 0.1192] |
| B/full | alpha_continuous | 0.509 [0.388, 0.676] | 0.0627 [0.0044, 0.1221] |
| B/full | alpha_micro | 0.553 [0.423, 0.722] | 0.0099 [-0.0335, 0.0689] |
| B/full | alpha_macro | 0.843 [0.772, 0.913] | -0.0380 [-0.0898, 0.0158] |
| B/full | all | 0.619 [0.518, 0.746] | 0.0116 [-0.0212, 0.0478] |
| B/full | Δ1 vs Δ15 (ratio of ratios; AUC difference) | 0.611 [0.503, 0.736] | 0.0246 [-0.1195, 0.1596] |
| LC1/request | delta_1 | 0.001 [0.001, 0.002] | 0.0152 [-0.0757, 0.1016] |
| LC1/request | delta_5 | 0.441 [0.311, 0.622] | 0.0174 [-0.0271, 0.0799] |
| LC1/request | delta_15 | 0.592 [0.460, 0.753] | 0.0275 [-0.0193, 0.0701] |
| LC1/request | alpha_continuous | 0.078 [0.057, 0.106] | 0.0198 [-0.0362, 0.0826] |
| LC1/request | alpha_micro | 0.069 [0.051, 0.093] | 0.0067 [-0.0256, 0.0400] |
| LC1/request | alpha_macro | 0.070 [0.053, 0.092] | 0.0335 [-0.0118, 0.0743] |
| LC1/request | all | 0.072 [0.054, 0.096] | 0.0200 [-0.0166, 0.0569] |
| LC1/request | Δ1 vs Δ15 (ratio of ratios; AUC difference) | 0.002 [0.002, 0.003] | -0.0123 [-0.1290, 0.0907] |
| LC1/full | delta_1 | n/a | n/a |
| LC1/full | delta_5 | 0.536 [0.400, 0.710] | -0.0285 [-0.1277, 0.0563] |
| LC1/full | delta_15 | 0.862 [0.606, 1.251] | -0.0254 [-0.1121, 0.0610] |
| LC1/full | alpha_continuous | n/a | n/a |
| LC1/full | alpha_micro | n/a | n/a |
| LC1/full | alpha_macro | 0.895 [0.895, 0.895] | -0.0222 [-0.0222, -0.0222] |
| LC1/full | all | n/a | n/a |
| LC1/full | Δ1 vs Δ15 (ratio of ratios; AUC difference) | n/a | n/a |
| LC3/request | delta_1 | 0.211 [0.128, 0.355] | -0.0140 [-0.0941, 0.0643] |
| LC3/request | delta_5 | 0.481 [0.385, 0.577] | -0.0233 [-0.0902, 0.0419] |
| LC3/request | delta_15 | 0.623 [0.507, 0.763] | 0.0549 [-0.0110, 0.1257] |
| LC3/request | alpha_continuous | 0.451 [0.363, 0.560] | 0.0032 [-0.0756, 0.0831] |
| LC3/request | alpha_micro | 0.282 [0.209, 0.381] | -0.0039 [-0.0520, 0.0448] |
| LC3/request | alpha_macro | 0.496 [0.388, 0.614] | 0.0183 [-0.0300, 0.0673] |
| LC3/request | all | 0.398 [0.332, 0.481] | 0.0059 [-0.0504, 0.0621] |
| LC3/request | Δ1 vs Δ15 (ratio of ratios; AUC difference) | 0.338 [0.190, 0.619] | -0.0689 [-0.1668, 0.0233] |
| LC3/full | delta_1 | 0.127 [0.070, 0.217] | 0.0323 [-0.0519, 0.1134] |
| LC3/full | delta_5 | 0.590 [0.480, 0.714] | -0.0545 [-0.1447, 0.0260] |
| LC3/full | delta_15 | 0.905 [0.706, 1.152] | -0.0022 [-0.0726, 0.0734] |
| LC3/full | alpha_continuous | 0.603 [0.472, 0.783] | -0.0293 [-0.1075, 0.0437] |
| LC3/full | alpha_micro | 0.194 [0.140, 0.265] | 0.0044 [-0.0744, 0.0769] |
| LC3/full | alpha_macro | 0.581 [0.416, 0.796] | 0.0005 [-0.0621, 0.0648] |
| LC3/full | all | 0.408 [0.334, 0.498] | -0.0081 [-0.0776, 0.0583] |
| LC3/full | Δ1 vs Δ15 (ratio of ratios; AUC difference) | 0.141 [0.068, 0.270] | 0.0345 [-0.0623, 0.1243] |

## Item 3. Training dynamics of the joint model

Pairwise cosine of per-request gradients on the shared trunk (training probe batch, mean over members and probes after update 0):

| recipe | phase | same Δ (mean / share < 0) | same description | different both |
| --- | --- | --- | --- | --- |
| B | all | 0.872 / 5.54e-04 | 0.930 / 0.001 | 0.844 / 5.14e-04 |
| B | 0-3000 | 0.945 / 0.000 | 0.939 / 0.000 | 0.903 / 0.000 |
| B | 3000-6000 | 0.896 / 0.000 | 0.926 / 0.000 | 0.867 / 0.000 |
| B | 6000-9000 | 0.773 / 0.002 | 0.926 / 0.004 | 0.761 / 0.002 |
| LC1 | all | 0.809 / 0.026 | 0.560 / 0.036 | 0.496 / 0.056 |
| LC1 | 0-3000 | 0.878 / 0.001 | 0.502 / 0.004 | 0.469 / 0.005 |
| LC1 | 3000-6000 | 0.771 / 0.031 | 0.585 / 0.042 | 0.499 / 0.070 |
| LC1 | 6000-9000 | 0.776 / 0.048 | 0.598 / 0.066 | 0.522 / 0.095 |
| LC3 | all | 0.823 / 0.019 | 0.631 / 0.023 | 0.560 / 0.041 |
| LC3 | 0-3000 | 0.890 / 0.001 | 0.577 / 0.006 | 0.542 / 0.006 |
| LC3 | 3000-6000 | 0.799 / 0.016 | 0.661 / 0.022 | 0.575 / 0.041 |
| LC3 | 6000-9000 | 0.779 / 0.040 | 0.655 / 0.042 | 0.564 / 0.074 |

Pre-clip gradient norm per update (median over members of the member median; largest member maximum):

| request | B J / S req / S full | LC1 J / S req / S full | LC3 J / S req / S full |
| --- | --- | --- | --- |
| F-1-continuous | 0.215 (16.9) / 0.121 (16.9) / 0.075 (16.9) | 22.771 (11267654423960092672) / 4105 (42185) / 67.646 (12136243321135693824) | 10.539 (1202245992448) / 70.508 (5854476107776) / 12.694 (482381761427275776) |
| F-1-micro | 0.214 (13.1) / 0.127 (16.8) / 0.078 (16.8) | 23.251 (13068774317024608256) / 2884 (1198480) / 89.931 (6744607184673308672) | 10.086 (13611156111360) / 111.271 (348575039488) / 563.935 (22964490052042752) |
| F-1-macro | 0.283 (14.0) / 0.186 (15.8) / 0.268 (15.8) | 20.543 (3403614659838738432) / 11260 (46653) / 64.004 (15798689065466855424) | 9.689 (1194029219840) / 249.857 (657282629632) / 4.652 (711705608847360) |
| F-5-continuous | 0.177 (12.5) / 0.125 (16.9) / 0.082 (16.9) | 8.170 (23175) / 24.246 (452.1) / 7.807 (368.7) | 6.559 (156658361696256) / 24.246 (452.1) / 7.807 (368.7) |
| F-5-micro | 0.191 (10.5) / 0.132 (16.8) / 0.086 (16.8) | 8.321 (418924) / 27.490 (367.2) / 9.452 (3164925430071296) | 6.935 (1103360688128) / 27.345 (547.3) / 9.452 (3164925430071296) |
| F-5-macro | 0.252 (10.8) / 0.182 (15.8) / 0.286 (15.8) | 7.507 (2997) / 21.472 (621.7) / 7.179 (1385) | 6.308 (202304847872) / 21.472 (621.7) / 7.254 (2058) |
| F-15-continuous | 0.197 (8.3) / 0.122 (16.9) / 0.077 (16.9) | 2.080 (25.3) / 3.867 (33.8) / 2.671 (1099) | 1.897 (134678) / 3.867 (33.8) / 2.671 (1099) |
| F-15-micro | 0.172 (7.7) / 0.127 (16.8) / 0.081 (16.8) | 2.033 (626.1) / 5.189 (126.7) / 2.771 (25154) | 1.911 (288406) / 5.191 (84.0) / 2.771 (25154) |
| F-15-macro | 0.241 (7.9) / 0.168 (15.8) / 0.265 (15.8) | 1.797 (21.1) / 3.781 (31.7) / 2.501 (4781) | 1.738 (236619) / 3.781 (31.7) / 2.501 (4781) |

Validation loss on the left-out lineage at the last probe (mean over LOLO members; local / long-horizon at 225), joint vs per-Δ and specialist at both budgets:

| recipe | request | J | D req | D full | S req | S full |
| --- | --- | --- | --- | --- | --- | --- |
| B | F-1-continuous | 0.001 / 680.213 | 0.002 / 2158 | 0.001 / 1276 | 0.002 / 2685 | 0.001 / 1433 |
| B | F-1-micro | 0.007 / 698.529 | 0.012 / 2182 | 0.007 / 1325 | 0.012 / 2806 | 0.012 / 1515 |
| B | F-1-macro | 0.134 / 753.370 | 0.134 / 2003 | 0.132 / 1121 | 0.135 / 2909 | 0.131 / 1224 |
| B | F-5-continuous | 0.003 / 0.058 | 0.003 / 3.102 | 0.002 / 0.085 | 0.004 / 8.805 | 0.003 / 0.285 |
| B | F-5-micro | 0.009 / 0.061 | 0.014 / 3.453 | 0.009 / 0.102 | 0.014 / 9.890 | 0.015 / 0.264 |
| B | F-5-macro | 0.124 / 0.068 | 0.126 / 1.326 | 0.122 / 0.052 | 0.126 / 12.330 | 0.122 / 0.055 |
| B | F-15-continuous | 0.002 / 0.005 | 0.002 / 0.011 | 0.002 / 0.004 | 0.003 / 0.018 | 0.003 / 0.006 |
| B | F-15-micro | 0.009 / 0.005 | 0.014 / 0.011 | 0.009 / 0.004 | 0.014 / 0.019 | 0.016 / 0.006 |
| B | F-15-macro | 0.096 / 0.004 | 0.098 / 0.009 | 0.094 / 0.004 | 0.099 / 0.028 | 0.099 / 0.006 |
| LC1 | F-1-continuous | 0.020 / 0.085 | 0.385 / 47.894 | 0.213 / 88.974 | 0.753 / 10.892 | 0.221 / 118.967 |
| LC1 | F-1-micro | 0.045 / 0.138 | 0.450 / 47.251 | 0.265 / 89.304 | 0.872 / 10.177 | 0.451 / 116.205 |
| LC1 | F-1-macro | 0.163 / 0.089 | 0.584 / 46.777 | 0.360 / 86.684 | 1.392 / 22.129 | 0.535 / 204.278 |
| LC1 | F-5-continuous | 0.016 / 0.055 | 0.004 / 0.024 | 0.005 / 0.017 | 0.006 / 0.033 | 0.008 / 0.016 |
| LC1 | F-5-micro | 0.042 / 0.098 | 0.016 / 0.024 | 0.012 / 0.017 | 0.020 / 0.049 | 0.257 / 0.584 |
| LC1 | F-5-macro | 0.144 / 0.060 | 0.126 / 0.023 | 0.127 / 0.016 | 0.128 / 0.032 | 0.124 / 0.017 |
| LC1 | F-15-continuous | 0.015 / 0.032 | 0.003 / 0.011 | 0.003 / 0.008 | 0.003 / 0.014 | 0.003 / 0.009 |
| LC1 | F-15-micro | 0.041 / 0.059 | 0.017 / 0.011 | 0.011 / 0.008 | 0.019 / 0.020 | 0.012 / 0.011 |
| LC1 | F-15-macro | 0.112 / 0.034 | 0.100 / 0.010 | 0.097 / 0.009 | 0.101 / 0.013 | 0.095 / 0.010 |
| LC3 | F-1-continuous | 0.009 / 0.048 | 0.173 / 0.489 | 0.270 / 0.807 | 0.093 / 0.439 | 0.173 / 0.441 |
| LC3 | F-1-micro | 0.015 / 0.050 | 0.168 / 0.389 | 0.274 / 0.793 | 0.239 / 0.789 | 0.398 / 2.233 |
| LC3 | F-1-macro | 0.144 / 0.056 | 0.298 / 0.393 | 0.408 / 0.816 | 0.180 / 0.501 | 0.335 / 1.282 |
| LC3 | F-5-continuous | 0.009 / 0.026 | 0.004 / 0.024 | 0.005 / 0.017 | 0.006 / 0.033 | 0.008 / 0.016 |
| LC3 | F-5-micro | 0.015 / 0.025 | 0.016 / 0.024 | 0.012 / 0.017 | 0.021 / 0.050 | 0.257 / 0.584 |
| LC3 | F-5-macro | 0.131 / 0.028 | 0.126 / 0.023 | 0.127 / 0.016 | 0.128 / 0.032 | 0.125 / 0.018 |
| LC3 | F-15-continuous | 0.007 / 0.012 | 0.003 / 0.011 | 0.003 / 0.009 | 0.003 / 0.014 | 0.003 / 0.009 |
| LC3 | F-15-micro | 0.015 / 0.012 | 0.017 / 0.011 | 0.011 / 0.009 | 0.019 / 0.020 | 0.012 / 0.011 |
| LC3 | F-15-macro | 0.105 / 0.012 | 0.100 / 0.010 | 0.097 / 0.009 | 0.101 / 0.013 | 0.095 / 0.010 |

Retired members and nonfinite updates, per unit:

| unit | retired / members | retired by request | members with skips | max skips |
| --- | --- | --- | --- | --- |
| LC1/S-1-continuous/full | 39/39 | {'F-1-continuous': 39} | 0 | 0 |
| LC1/S-1-macro/full | 38/39 | {'F-1-macro': 38} | 0 | 0 |
| LC3/D1/full | 0/39 | {} | 1 | 1 |
| LC1/D1/full | 39/39 | {'F-1-macro': 10, 'F-1-micro': 11, 'F-1-continuous': 18} | 0 | 0 |
| LC1/S-1-micro/full | 39/39 | {'F-1-micro': 39} | 0 | 0 |
| LC1/J/full | 5/39 | {'F-1-macro': 1, 'F-1-micro': 2, 'F-1-continuous': 2} | 0 | 0 |
| LC1/D1/request | 38/39 | {'F-1-continuous': 15, 'F-1-macro': 9, 'F-1-micro': 14} | 0 | 0 |

First nonfinite update of every affected member (this run):

| unit | fold | seed | update (step) | request | loss finite | norm finite | horizon | retired |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LC1/S-1-continuous/full | 001 | 20260908 | 1363 (1363) | F-1-continuous | True | False | 90 | True |
| LC1/S-1-continuous/full | 001 | 20260909 | 1787 (1787) | F-1-continuous | True | False | 105 | True |
| LC1/S-1-continuous/full | 001 | 20260910 | 3930 (3930) | F-1-continuous | True | False | 165 | True |
| LC1/S-1-continuous/full | 002 | 20260908 | 2160 (2160) | F-1-continuous | True | False | 120 | True |
| LC1/S-1-continuous/full | 002 | 20260909 | 1434 (1434) | F-1-continuous | True | False | 90 | True |
| LC1/S-1-continuous/full | 002 | 20260910 | 4368 (4368) | F-1-continuous | True | False | 180 | True |
| LC1/S-1-continuous/full | 003 | 20260908 | 1525 (1525) | F-1-continuous | True | False | 105 | True |
| LC1/S-1-continuous/full | 003 | 20260909 | 3355 (3355) | F-1-continuous | True | False | 150 | True |
| LC1/S-1-continuous/full | 003 | 20260910 | 4863 (4863) | F-1-continuous | True | False | 195 | True |
| LC1/S-1-continuous/full | 004 | 20260908 | 1591 (1591) | F-1-continuous | True | False | 105 | True |
| LC1/S-1-continuous/full | 004 | 20260909 | 1455 (1455) | F-1-continuous | True | False | 90 | True |
| LC1/S-1-continuous/full | 004 | 20260910 | 4172 (4172) | F-1-continuous | True | False | 180 | True |
| LC1/S-1-continuous/full | 005 | 20260908 | 2571 (2571) | F-1-continuous | True | False | 135 | True |
| LC1/S-1-continuous/full | 005 | 20260909 | 1986 (1986) | F-1-continuous | True | False | 105 | True |
| LC1/S-1-continuous/full | 005 | 20260910 | 3356 (3356) | F-1-continuous | True | False | 150 | True |
| LC1/S-1-continuous/full | 006 | 20260908 | 1532 (1532) | F-1-continuous | True | False | 105 | True |
| LC1/S-1-continuous/full | 006 | 20260909 | 1446 (1446) | F-1-continuous | True | False | 90 | True |
| LC1/S-1-continuous/full | 006 | 20260910 | 5982 (5982) | F-1-continuous | True | False | 225 | True |
| LC1/S-1-continuous/full | 009 | 20260908 | 2305 (2305) | F-1-continuous | True | False | 120 | True |
| LC1/S-1-continuous/full | 009 | 20260909 | 2200 (2200) | F-1-continuous | True | False | 120 | True |
| LC1/S-1-continuous/full | 009 | 20260910 | 2322 (2322) | F-1-continuous | True | False | 120 | True |
| LC1/S-1-continuous/full | 010 | 20260908 | 1412 (1412) | F-1-continuous | True | False | 90 | True |
| LC1/S-1-continuous/full | 010 | 20260909 | 1629 (1629) | F-1-continuous | True | False | 105 | True |
| LC1/S-1-continuous/full | 010 | 20260910 | 4120 (4120) | F-1-continuous | True | False | 180 | True |
| LC1/S-1-continuous/full | 011 | 20260908 | 2318 (2318) | F-1-continuous | True | False | 120 | True |
| LC1/S-1-continuous/full | 011 | 20260909 | 1443 (1443) | F-1-continuous | True | False | 90 | True |
| LC1/S-1-continuous/full | 011 | 20260910 | 4444 (4444) | F-1-continuous | True | False | 180 | True |
| LC1/S-1-continuous/full | 012 | 20260908 | 1863 (1863) | F-1-continuous | True | False | 105 | True |
| LC1/S-1-continuous/full | 012 | 20260909 | 1616 (1616) | F-1-continuous | True | False | 105 | True |
| LC1/S-1-continuous/full | 012 | 20260910 | 4121 (4121) | F-1-continuous | True | False | 180 | True |
| LC1/S-1-continuous/full | 013 | 20260908 | 2468 (2468) | F-1-continuous | True | False | 120 | True |
| LC1/S-1-continuous/full | 013 | 20260909 | 1522 (1522) | F-1-continuous | True | False | 105 | True |
| LC1/S-1-continuous/full | 013 | 20260910 | 3567 (3567) | F-1-continuous | True | False | 165 | True |
| LC1/S-1-continuous/full | 015 | 20260908 | 2938 (2938) | F-1-continuous | True | False | 135 | True |
| LC1/S-1-continuous/full | 015 | 20260909 | 1840 (1840) | F-1-continuous | True | False | 105 | True |
| LC1/S-1-continuous/full | 015 | 20260910 | 4383 (4383) | F-1-continuous | True | False | 180 | True |
| LC1/S-1-continuous/full | all | 20260908 | 1649 (1649) | F-1-continuous | True | False | 105 | True |
| LC1/S-1-continuous/full | all | 20260909 | 2488 (2488) | F-1-continuous | True | False | 120 | True |
| LC1/S-1-continuous/full | all | 20260910 | 4488 (4488) | F-1-continuous | True | False | 180 | True |
| LC1/S-1-macro/full | 001 | 20260908 | 4837 (4837) | F-1-macro | True | False | 195 | True |
| LC1/S-1-macro/full | 001 | 20260909 | 1469 (1469) | F-1-macro | True | False | 90 | True |
| LC1/S-1-macro/full | 001 | 20260910 | 4165 (4165) | F-1-macro | True | False | 180 | True |
| LC1/S-1-macro/full | 002 | 20260908 | 6286 (6286) | F-1-macro | True | False | 225 | True |
| LC1/S-1-macro/full | 002 | 20260909 | 1584 (1584) | F-1-macro | True | False | 105 | True |
| LC1/S-1-macro/full | 002 | 20260910 | 4657 (4657) | F-1-macro | True | False | 195 | True |
| LC1/S-1-macro/full | 003 | 20260908 | 5448 (5448) | F-1-macro | True | False | 210 | True |
| LC1/S-1-macro/full | 003 | 20260909 | 3232 (3232) | F-1-macro | True | False | 150 | True |
| LC1/S-1-macro/full | 003 | 20260910 | 4321 (4321) | F-1-macro | True | False | 180 | True |
| LC1/S-1-macro/full | 004 | 20260908 | 7468 (7468) | F-1-macro | True | False | 225 | True |
| LC1/S-1-macro/full | 004 | 20260909 | 1985 (1985) | F-1-macro | True | False | 105 | True |
| LC1/S-1-macro/full | 004 | 20260910 | 4478 (4478) | F-1-macro | True | False | 180 | True |
| LC1/S-1-macro/full | 005 | 20260908 | 7118 (7118) | F-1-macro | True | False | 225 | True |
| LC1/S-1-macro/full | 005 | 20260909 | 2548 (2548) | F-1-macro | True | False | 135 | True |
| LC1/S-1-macro/full | 005 | 20260910 | 3663 (3663) | F-1-macro | True | False | 165 | True |
| LC1/S-1-macro/full | 006 | 20260908 | 6203 (6203) | F-1-macro | True | False | 225 | True |
| LC1/S-1-macro/full | 006 | 20260909 | 6681 (6681) | F-1-macro | True | False | 225 | True |
| LC1/S-1-macro/full | 006 | 20260910 | 3958 (3958) | F-1-macro | True | False | 165 | True |
| LC1/S-1-macro/full | 009 | 20260908 | 5760 (5760) | F-1-macro | True | False | 225 | True |
| LC1/S-1-macro/full | 009 | 20260909 | 1839 (1839) | F-1-macro | True | False | 105 | True |
| LC1/S-1-macro/full | 009 | 20260910 | 4587 (4587) | F-1-macro | True | False | 195 | True |
| LC1/S-1-macro/full | 010 | 20260908 | 5957 (5957) | F-1-macro | True | False | 225 | True |
| LC1/S-1-macro/full | 010 | 20260909 | 2486 (2486) | F-1-macro | True | False | 120 | True |
| LC1/S-1-macro/full | 010 | 20260910 | 3649 (3649) | F-1-macro | True | False | 165 | True |
| LC1/S-1-macro/full | 011 | 20260908 | 6770 (6770) | F-1-macro | True | False | 225 | True |
| LC1/S-1-macro/full | 011 | 20260909 | 3224 (3224) | F-1-macro | True | False | 150 | True |
| LC1/S-1-macro/full | 011 | 20260910 | 4006 (4006) | F-1-macro | True | False | 180 | True |
| LC1/S-1-macro/full | 012 | 20260909 | 2802 (2802) | F-1-macro | True | False | 135 | True |
| LC1/S-1-macro/full | 012 | 20260910 | 4992 (4992) | F-1-macro | True | False | 195 | True |
| LC1/S-1-macro/full | 013 | 20260908 | 6459 (6459) | F-1-macro | True | False | 225 | True |
| LC1/S-1-macro/full | 013 | 20260909 | 1459 (1459) | F-1-macro | True | False | 90 | True |
| LC1/S-1-macro/full | 013 | 20260910 | 4088 (4088) | F-1-macro | True | False | 180 | True |
| LC1/S-1-macro/full | 015 | 20260908 | 5321 (5321) | F-1-macro | True | False | 210 | True |
| LC1/S-1-macro/full | 015 | 20260909 | 1836 (1836) | F-1-macro | True | False | 105 | True |
| LC1/S-1-macro/full | 015 | 20260910 | 4154 (4154) | F-1-macro | True | False | 180 | True |
| LC1/S-1-macro/full | all | 20260908 | 6316 (6316) | F-1-macro | True | False | 225 | True |
| LC1/S-1-macro/full | all | 20260909 | 1659 (1659) | F-1-macro | True | False | 105 | True |
| LC1/S-1-macro/full | all | 20260910 | 4025 (4025) | F-1-macro | True | False | 180 | True |
| LC3/D1/full | all | 20260909 | 7546 (7546) | F-1-micro | True | False | 225 | False |
| LC1/D1/full | 001 | 20260908 | 1619 (1619) | F-1-macro | True | False | 105 | True |
| LC1/D1/full | 001 | 20260909 | 3997 (3997) | F-1-micro | True | False | 165 | True |
| LC1/D1/full | 001 | 20260910 | 3714 (3714) | F-1-continuous | True | False | 165 | True |
| LC1/D1/full | 002 | 20260908 | 1425 (1425) | F-1-continuous | True | False | 90 | True |
| LC1/D1/full | 002 | 20260909 | 1335 (1335) | F-1-continuous | True | False | 90 | True |
| LC1/D1/full | 002 | 20260910 | 3627 (3627) | F-1-continuous | True | False | 165 | True |
| LC1/D1/full | 003 | 20260908 | 1701 (1701) | F-1-continuous | True | False | 105 | True |
| LC1/D1/full | 003 | 20260909 | 1454 (1454) | F-1-macro | True | False | 90 | True |
| LC1/D1/full | 003 | 20260910 | 3614 (3614) | F-1-macro | True | False | 165 | True |
| LC1/D1/full | 004 | 20260908 | 1732 (1732) | F-1-micro | True | False | 105 | True |
| LC1/D1/full | 004 | 20260909 | 1399 (1399) | F-1-micro | True | False | 90 | True |
| LC1/D1/full | 004 | 20260910 | 4159 (4159) | F-1-micro | True | False | 180 | True |
| LC1/D1/full | 005 | 20260908 | 1787 (1787) | F-1-macro | True | False | 105 | True |
| LC1/D1/full | 005 | 20260909 | 1269 (1269) | F-1-continuous | True | False | 90 | True |
| LC1/D1/full | 005 | 20260910 | 4088 (4088) | F-1-macro | True | False | 180 | True |
| LC1/D1/full | 006 | 20260908 | 1646 (1646) | F-1-macro | True | False | 105 | True |
| LC1/D1/full | 006 | 20260909 | 1522 (1522) | F-1-micro | True | False | 105 | True |
| LC1/D1/full | 006 | 20260910 | 3114 (3114) | F-1-continuous | True | False | 150 | True |
| LC1/D1/full | 009 | 20260908 | 1443 (1443) | F-1-continuous | True | False | 90 | True |
| LC1/D1/full | 009 | 20260909 | 1398 (1398) | F-1-continuous | True | False | 90 | True |
| LC1/D1/full | 009 | 20260910 | 4228 (4228) | F-1-micro | True | False | 180 | True |
| LC1/D1/full | 010 | 20260908 | 1294 (1294) | F-1-micro | True | False | 90 | True |
| LC1/D1/full | 010 | 20260909 | 1702 (1702) | F-1-micro | True | False | 105 | True |
| LC1/D1/full | 010 | 20260910 | 3055 (3055) | F-1-micro | True | False | 150 | True |
| LC1/D1/full | 011 | 20260908 | 1230 (1230) | F-1-continuous | True | False | 90 | True |
| LC1/D1/full | 011 | 20260909 | 1767 (1767) | F-1-continuous | True | False | 105 | True |
| LC1/D1/full | 011 | 20260910 | 3663 (3663) | F-1-continuous | True | False | 165 | True |
| LC1/D1/full | 012 | 20260908 | 2132 (2132) | F-1-macro | True | False | 120 | True |
| LC1/D1/full | 012 | 20260909 | 1323 (1323) | F-1-continuous | True | False | 90 | True |
| LC1/D1/full | 012 | 20260910 | 3477 (3477) | F-1-continuous | True | False | 150 | True |
| LC1/D1/full | 013 | 20260908 | 1290 (1290) | F-1-continuous | True | False | 90 | True |
| LC1/D1/full | 013 | 20260909 | 1515 (1515) | F-1-continuous | True | False | 105 | True |
| LC1/D1/full | 013 | 20260910 | 3337 (3337) | F-1-micro | True | False | 150 | True |
| LC1/D1/full | 015 | 20260908 | 1171 (1171) | F-1-micro | True | False | 90 | True |
| LC1/D1/full | 015 | 20260909 | 1733 (1733) | F-1-macro | True | False | 105 | True |
| LC1/D1/full | 015 | 20260910 | 2472 (2472) | F-1-continuous | True | False | 120 | True |
| LC1/D1/full | all | 20260908 | 1601 (1601) | F-1-macro | True | False | 105 | True |
| LC1/D1/full | all | 20260909 | 1538 (1538) | F-1-macro | True | False | 105 | True |
| LC1/D1/full | all | 20260910 | 3588 (3588) | F-1-continuous | True | False | 165 | True |
| LC1/S-1-micro/full | 001 | 20260908 | 1920 (1920) | F-1-micro | True | False | 105 | True |
| LC1/S-1-micro/full | 001 | 20260909 | 5022 (5022) | F-1-micro | True | False | 210 | True |
| LC1/S-1-micro/full | 001 | 20260910 | 3935 (3935) | F-1-micro | True | False | 165 | True |
| LC1/S-1-micro/full | 002 | 20260908 | 1536 (1536) | F-1-micro | True | False | 105 | True |
| LC1/S-1-micro/full | 002 | 20260909 | 5021 (5021) | F-1-micro | True | False | 210 | True |
| LC1/S-1-micro/full | 002 | 20260910 | 3701 (3701) | F-1-micro | True | False | 165 | True |
| LC1/S-1-micro/full | 003 | 20260908 | 1618 (1618) | F-1-micro | True | False | 105 | True |
| LC1/S-1-micro/full | 003 | 20260909 | 1896 (1896) | F-1-micro | True | False | 105 | True |
| LC1/S-1-micro/full | 003 | 20260910 | 3719 (3719) | F-1-micro | True | False | 165 | True |
| LC1/S-1-micro/full | 004 | 20260908 | 1742 (1742) | F-1-micro | True | False | 105 | True |
| LC1/S-1-micro/full | 004 | 20260909 | 3598 (3598) | F-1-micro | True | False | 165 | True |
| LC1/S-1-micro/full | 004 | 20260910 | 3820 (3820) | F-1-micro | True | False | 165 | True |
| LC1/S-1-micro/full | 005 | 20260908 | 1661 (1661) | F-1-micro | True | False | 105 | True |
| LC1/S-1-micro/full | 005 | 20260909 | 5346 (5346) | F-1-micro | True | False | 210 | True |
| LC1/S-1-micro/full | 005 | 20260910 | 4106 (4106) | F-1-micro | True | False | 180 | True |
| LC1/S-1-micro/full | 006 | 20260908 | 1440 (1440) | F-1-micro | True | False | 90 | True |
| LC1/S-1-micro/full | 006 | 20260909 | 5886 (5886) | F-1-micro | True | False | 225 | True |
| LC1/S-1-micro/full | 006 | 20260910 | 3959 (3959) | F-1-micro | True | False | 165 | True |
| LC1/S-1-micro/full | 009 | 20260908 | 1585 (1585) | F-1-micro | True | False | 105 | True |
| LC1/S-1-micro/full | 009 | 20260909 | 6310 (6310) | F-1-micro | True | False | 225 | True |
| LC1/S-1-micro/full | 009 | 20260910 | 4024 (4024) | F-1-micro | True | False | 180 | True |
| LC1/S-1-micro/full | 010 | 20260908 | 1806 (1806) | F-1-micro | True | False | 105 | True |
| LC1/S-1-micro/full | 010 | 20260909 | 5017 (5017) | F-1-micro | True | False | 210 | True |
| LC1/S-1-micro/full | 010 | 20260910 | 3897 (3897) | F-1-micro | True | False | 165 | True |
| LC1/S-1-micro/full | 011 | 20260908 | 1444 (1444) | F-1-micro | True | False | 90 | True |
| LC1/S-1-micro/full | 011 | 20260909 | 5040 (5040) | F-1-micro | True | False | 210 | True |
| LC1/S-1-micro/full | 011 | 20260910 | 4158 (4158) | F-1-micro | True | False | 180 | True |
| LC1/S-1-micro/full | 012 | 20260908 | 1775 (1775) | F-1-micro | True | False | 105 | True |
| LC1/S-1-micro/full | 012 | 20260909 | 1753 (1753) | F-1-micro | True | False | 105 | True |
| LC1/S-1-micro/full | 012 | 20260910 | 4257 (4257) | F-1-micro | True | False | 180 | True |
| LC1/S-1-micro/full | 013 | 20260908 | 1867 (1867) | F-1-micro | True | False | 105 | True |
| LC1/S-1-micro/full | 013 | 20260909 | 1857 (1857) | F-1-micro | True | False | 105 | True |
| LC1/S-1-micro/full | 013 | 20260910 | 4067 (4067) | F-1-micro | True | False | 180 | True |
| LC1/S-1-micro/full | 015 | 20260908 | 1905 (1905) | F-1-micro | True | False | 105 | True |
| LC1/S-1-micro/full | 015 | 20260909 | 4230 (4230) | F-1-micro | True | False | 180 | True |
| LC1/S-1-micro/full | 015 | 20260910 | 3926 (3926) | F-1-micro | True | False | 165 | True |
| LC1/S-1-micro/full | all | 20260908 | 2061 (2061) | F-1-micro | True | False | 120 | True |
| LC1/S-1-micro/full | all | 20260909 | 5303 (5303) | F-1-micro | True | False | 210 | True |
| LC1/S-1-micro/full | all | 20260910 | 3947 (3947) | F-1-micro | True | False | 165 | True |
| LC1/J/full | 004 | 20260909 | 8075 (8075) | F-1-macro | True | False | 225 | True |
| LC1/J/full | 005 | 20260908 | 5653 (5653) | F-1-micro | True | False | 225 | True |
| LC1/J/full | 005 | 20260910 | 5229 (5229) | F-1-continuous | True | False | 210 | True |
| LC1/J/full | 010 | 20260909 | 8029 (8029) | F-1-micro | True | False | 225 | True |
| LC1/J/full | all | 20260910 | 7083 (7083) | F-1-continuous | True | False | 225 | True |
| LC1/D1/request | 001 | 20260908 | 1878 (5634) | F-1-continuous | True | False | 225 | True |
| LC1/D1/request | 001 | 20260909 | 1107 (3321) | F-1-continuous | True | False | 150 | True |
| LC1/D1/request | 001 | 20260910 | 2880 (8640) | F-1-continuous | True | False | 225 | True |
| LC1/D1/request | 002 | 20260908 | 1040 (3116) | F-1-macro | True | False | 150 | True |
| LC1/D1/request | 002 | 20260909 | 1025 (3071) | F-1-macro | True | False | 150 | True |
| LC1/D1/request | 002 | 20260910 | 2829 (8487) | F-1-continuous | True | False | 225 | True |
| LC1/D1/request | 003 | 20260908 | 2597 (7787) | F-1-macro | True | False | 225 | True |
| LC1/D1/request | 003 | 20260909 | 1051 (3151) | F-1-micro | True | False | 150 | True |
| LC1/D1/request | 003 | 20260910 | 2680 (8038) | F-1-micro | True | False | 225 | True |
| LC1/D1/request | 004 | 20260908 | 1262 (3782) | F-1-macro | True | False | 165 | True |
| LC1/D1/request | 004 | 20260909 | 1272 (3816) | F-1-continuous | True | False | 165 | True |
| LC1/D1/request | 004 | 20260910 | 2965 (8893) | F-1-micro | True | False | 225 | True |
| LC1/D1/request | 005 | 20260908 | 1461 (4383) | F-1-continuous | True | False | 180 | True |
| LC1/D1/request | 005 | 20260909 | 1689 (5067) | F-1-continuous | True | False | 210 | True |
| LC1/D1/request | 006 | 20260908 | 1171 (3511) | F-1-micro | True | False | 165 | True |
| LC1/D1/request | 006 | 20260909 | 1114 (3340) | F-1-micro | True | False | 150 | True |
| LC1/D1/request | 006 | 20260910 | 2824 (8470) | F-1-micro | True | False | 225 | True |
| LC1/D1/request | 009 | 20260908 | 987 (2961) | F-1-continuous | True | False | 135 | True |
| LC1/D1/request | 009 | 20260909 | 1234 (3700) | F-1-micro | True | False | 165 | True |
| LC1/D1/request | 009 | 20260910 | 2937 (8811) | F-1-continuous | True | False | 225 | True |
| LC1/D1/request | 010 | 20260908 | 1205 (3611) | F-1-macro | True | False | 165 | True |
| LC1/D1/request | 010 | 20260909 | 1772 (5312) | F-1-macro | True | False | 210 | True |
| LC1/D1/request | 010 | 20260910 | 2808 (8424) | F-1-continuous | True | False | 225 | True |
| LC1/D1/request | 011 | 20260908 | 1707 (5121) | F-1-continuous | True | False | 210 | True |
| LC1/D1/request | 011 | 20260909 | 1270 (3808) | F-1-micro | True | False | 165 | True |
| LC1/D1/request | 011 | 20260910 | 2803 (8407) | F-1-micro | True | False | 225 | True |
| LC1/D1/request | 012 | 20260908 | 1048 (3142) | F-1-micro | True | False | 150 | True |
| LC1/D1/request | 012 | 20260909 | 1959 (5877) | F-1-continuous | True | False | 225 | True |
| LC1/D1/request | 012 | 20260910 | 2852 (8552) | F-1-macro | True | False | 225 | True |
| LC1/D1/request | 013 | 20260908 | 1465 (4393) | F-1-micro | True | False | 180 | True |
| LC1/D1/request | 013 | 20260909 | 1228 (3682) | F-1-micro | True | False | 165 | True |
| LC1/D1/request | 013 | 20260910 | 2867 (8597) | F-1-macro | True | False | 225 | True |
| LC1/D1/request | 015 | 20260908 | 1233 (3699) | F-1-continuous | True | False | 165 | True |
| LC1/D1/request | 015 | 20260909 | 1074 (3222) | F-1-continuous | True | False | 150 | True |
| LC1/D1/request | 015 | 20260910 | 2960 (8876) | F-1-macro | True | False | 225 | True |
| LC1/D1/request | all | 20260908 | 1954 (5860) | F-1-micro | True | False | 225 | True |
| LC1/D1/request | all | 20260909 | 1588 (4762) | F-1-micro | True | False | 195 | True |
| LC1/D1/request | all | 20260910 | 2778 (8334) | F-1-continuous | True | False | 225 | True |

#112 v1 published retirements by request: {'L:continuous:C-F-1': 3, 'L:hybrid:F-1-micro': 2, 'LC:continuous:C-F-1': 17, 'LC:hybrid:F-1-macro': 1, 'LC:hybrid:F-1-micro': 2, 'LC:hybrid:F-1-continuous': 3}.

### End-of-training excursions (POST HOC (added after every outcome; no token or trigger reads it))

A member ends in an excursion when, for some request, the median #77 local loss of its last 3 updates exceeds 5× the median of that request's updates in the [0.6 n, 0.95 n) window; the spike is the first update of its last 500 with a pre-clip norm above 1000.

| recipe | J | D1 req / full | D5 req / full | D15 req / full | S-1-* req / full | S-5-* req / full | S-15-* req / full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B | 0/39 | 0/39 / 0/39 | 0/39 / 0/39 | 0/39 / 0/39 | 0/117 / 1/117 | 0/117 / 1/117 | 0/117 / 0/117 |
| LC1 | 3/39 | 14/39 / 0/39 | 0/39 / 0/39 | 0/39 / 0/39 | 0/117 / 1/117 | 0/117 / 13/117 | 0/117 / 0/117 |
| LC3 | 1/39 | 4/39 / 4/39 | 0/39 / 0/39 | 0/39 / 0/39 | 6/117 / 18/117 | 0/117 / 13/117 | 0/117 / 0/117 |

Joint members that end in an excursion:

| unit | fold | seed | requests elevated ≥ 5× | first spike (update, request, norm) | requests with spikes |
| --- | --- | --- | --- | --- | --- |
| LC1/J/full | issue-77-n1-003 | 20260909 | 3/9 | 8982, F-1-continuous, 1.81e+08 | 1-continuous, 1-macro, 1-micro, 5-continuous, 5-macro |
| LC1/J/full | issue-77-n1-004 | 20260909 | 6/9 | none |  |
| LC1/J/full | issue-77-n1-010 | 20260909 | 1/9 | none |  |
| LC3/J/full | all | 20260908 | 6/9 | 8605, F-1-micro, 4.84e+04 | 1-continuous, 1-macro, 1-micro, 15-continuous, 5-continuous, 5-macro, 5-micro |

Held-out rollout error at t = 225 of each joint fold-'all' member (the Gate-B checkpoint type; mean over the held-out lineages):

| recipe | seed | 1-continuous | 1-micro | 1-macro | 5-continuous | 5-micro | 5-macro | 15-continuous | 15-micro | 15-macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B | 20260908 | 3429 | 3472 | 5214 | 0.1992 | 0.1989 | 0.2661 | 0.0496 | 0.0491 | 0.0372 |
| B | 20260909 | 4250 | 3983 | 5308 | 0.5885 | 0.6968 | 0.6836 | 0.0699 | 0.0731 | 0.0697 |
| B | 20260910 | 5093 | 4746 | 4792 | 0.3126 | 0.4030 | 0.3788 | 0.0555 | 0.0541 | 0.0457 |
| LC1 | 20260908 | 0.0173 | 0.0173 | 0.0148 | 0.0193 | 0.0202 | 0.0218 | 0.0335 | 0.0371 | 0.0402 |
| LC1 | 20260909 | 0.0264 | 0.0288 | 0.0302 | 0.0309 | 0.0366 | 0.0461 | 0.0425 | 0.0460 | 0.0498 |
| LC1 | 20260910 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| LC3 | 20260908 | 1.3179 | 1.0314 | 1.4101 | 0.5260 | 0.5622 | 0.6143 | 1.0653 | 0.9074 | 1.0503 |
| LC3 | 20260909 | 0.0205 | 0.0202 | 0.0224 | 0.0265 | 0.0231 | 0.0423 | 0.0334 | 0.0320 | 0.0420 |
| LC3 | 20260910 | 0.0319 | 0.0308 | 0.0350 | 0.0393 | 0.0393 | 0.0412 | 0.0516 | 0.0505 | 0.0556 |

## Cross-hardware replication of the #112 v3 joint arms (LOLO grid AUC / rollout at 225)

| request | B here | B #112 v3 | LC3 here | LC #112 v3 |
| --- | --- | --- | --- | --- |
| F-1-continuous | 0.5598 / 3841 | 0.5681 / 3622 | 0.5047 / 0.0607 | 0.5316 / 0.0565 |
| F-1-micro | 0.5505 / 3793 | 0.5141 / 3775 | 0.5113 / 0.0607 | 0.5233 / 0.0596 |
| F-1-macro | 0.5304 / 4039 | 0.5377 / 3881 | 0.5248 / 0.0744 | 0.5483 / 0.0613 |
| F-5-continuous | 0.4763 / 0.2587 | 0.4808 / 0.3027 | 0.5921 / 0.0619 | 0.5930 / 0.0582 |
| F-5-micro | 0.4944 / 0.2842 | 0.5175 / 0.3279 | 0.5859 / 0.0612 | 0.6272 / 0.0574 |
| F-5-macro | 0.4803 / 0.2858 | 0.5057 / 0.3230 | 0.6310 / 0.0679 | 0.6452 / 0.0608 |
| F-15-continuous | 0.5294 / 0.0347 | 0.5028 / 0.0368 | 0.6861 / 0.0382 | 0.7180 / 0.0379 |
| F-15-micro | 0.5211 / 0.0345 | 0.5093 / 0.0358 | 0.6918 / 0.0389 | 0.7294 / 0.0384 |
| F-15-macro | 0.4861 / 0.0271 | 0.4960 / 0.0266 | 0.7245 / 0.0398 | 0.7153 / 0.0396 |

## Held-out (007/008/014/016; scored once after every LOLO record; DESCRIPTIVE)

| recipe / budget | grid AUC J − S (request-mean) | rollout J / S at 225 (all requests) |
| --- | --- | --- |
| B/request | -0.0403 [-0.0691, 0.0025] | 0.146 [0.123, 0.176] |
| B/full | 0.0379 [-0.0099, 0.0914] | 1.111 [1.048, 1.171] |
| LC1/request | 0.2049 [0.1444, 0.2481] | 0.039 [0.030, 0.059] |
| LC1/full | n/a | n/a |
| LC3/request | 0.0914 [0.0691, 0.1259] | 1.854 [1.424, 2.314] |
| LC3/full | 0.1251 [0.0593, 0.1630] | 2.828 [2.194, 3.646] |

## Compute

GPU seconds 80179; wall seconds (summed over workers) 80251; engine seconds 0.

Claim boundary: NovPhy only (items 1-3 of #117; CLEVRER and Physion-Dominoes are item 5); E99 carriers; decision-only ranking of retained engine-verdicted executions; fit lineages by LOLO and the 4 exposed held-out lineages once, descriptively; no engine access, rendering or capture; no closed-loop or competence claim.
