# Issue #119: remaining-frames input saturation (#104 sealed #109 evaluation split)

Primary claim: "the kappa cap costs decision value: an unsaturated remaining-frames input improves J - F_hindsight by >= delta on the primary inventory".

- v* = **b** (min(kappa, 225) / 225), selected on the policy split before any score (seed-mean policy-split AUC: b 0.7383, c 0.7361, d 0.7349; control a 0.7208).
- (J(v*) - F_hindsight) - (J - F_hindsight) = J(b, L60) - J(a, L60), grid: **0.0042 [-0.0005, 0.0088]** over 933 cells (311 levels); F_hindsight = F-15-continuous. Token: **not_supported_by_this_experiment**.
- Rule: (J(v*) - F_hindsight) - (J - F_hindsight) = J(v*, L60) - J(a, L60) paired grid AUC per (level, seed), F_hindsight common to both (the in-inventory best fixed request of the #101 records); supported iff mean >= 0.02 with lower bound > 0; not_supported_by_this_experiment iff upper < 0.02; otherwise readiness_or_precision_insufficient; any failed guard (including the variant-a replication guard) -> readiness_or_precision_insufficient

Intervals: member (level)-clustered percentile bootstrap, 10000 draws, PCG64 7201, DESCRIPTIVE; δ = 0.02. Tokens follow the δ rule only: supported iff mean ≥ δ and lower > 0; not_supported_by_this_experiment iff upper < δ; otherwise readiness_or_precision_insufficient. Offset, angle and power are DESCRIPTIVE.

## Per-variant table (grid)

| arm | input | labels | label acc. L60 | label acc. L225 | own rows | AUC | − F_hindsight | − J | − prior | MACs / candidate | policy MACs / state | right − wrong anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| J(a, L60) = J | a | pe, L60 | 47.2% | 37.4% | 48.9% | 0.7057 [0.6846, 0.7267] | -0.0565 [-0.0677, -0.0458] | — | -0.0146 [-0.0388, 0.0088] | 184.0M | 32128 | 0.0336 [0.0299, 0.0373] |
| J(b, L60) | b | pe, L60 | 47.3% | 38.0% | 47.6% | 0.7099 [0.6891, 0.7304] | -0.0524 [-0.0617, -0.0431] | 0.0042 [-0.0005, 0.0088] | -0.0104 [-0.0368, 0.0151] | 164.9M | 32128 | 0.0331 [0.0290, 0.0373] |
| J(c, L60) | c | pe, L60 | 47.5% | 43.8% | 49.9% | 0.7094 [0.6879, 0.7308] | -0.0528 [-0.0633, -0.0430] | 0.0037 [-0.0006, 0.0082] | -0.0109 [-0.0375, 0.0147] | 227.5M | 32128 | 0.0322 [0.0284, 0.0360] |
| J(d, L60) | d | pe, L60 | 47.4% | 41.2% | 49.1% | 0.7052 [0.6841, 0.7261] | -0.0571 [-0.0674, -0.0472] | -0.0006 [-0.0062, 0.0050] | -0.0151 [-0.0409, 0.0099] | 193.4M | 32256 | 0.0335 [0.0275, 0.0394] |
| J(a, L225) | a | pe, L225 | 47.0% | 53.5% | 54.5% | 0.6802 [0.6579, 0.7022] | -0.0821 [-0.0939, -0.0705] | -0.0255 [-0.0316, -0.0194] | -0.0401 [-0.0680, -0.0129] | 379.2M | 32128 | 0.0014 [0.0008, 0.0021] |
| J(b, L225) | b | pe, L225 | 47.5% | 53.7% | 54.9% | 0.6833 [0.6609, 0.7052] | -0.0790 [-0.0913, -0.0672] | -0.0225 [-0.0280, -0.0170] | -0.0370 [-0.0646, -0.0102] | 376.6M | 32128 | 0.0047 [0.0032, 0.0061] |
| J_dec(b) | b | decision (λ 1e-1) | 9.1% | 6.6% | 77.7% | 0.7418 [0.7243, 0.7595] | -0.0204 [-0.0249, -0.0159] | 0.0361 [0.0266, 0.0457] | 0.0215 [-0.0039, 0.0465] | 22.7M | 32128 | 0.1251 [0.1151, 0.1352] |
| J_dec(a) (#101 J_dec) | a | decision (λ 1e-1) | 9.1% | 5.2% | 78.2% | 0.7460 [0.7277, 0.7637] | -0.0163 [-0.0203, -0.0122] | 0.0403 [0.0295, 0.0516] | 0.0257 [-0.0002, 0.0512] | 22.7M | 32128 | 0.1279 [0.1172, 0.1387] (#101) |

Label accuracy is argmax agreement with the DP labels on the policy split (training data, in-sample for a policy's own label kind): L60 = round-0 labels of the first 60-frame windows, L225 = round-0 labels of the 225-frame windows (both from the variant-a groups of the same seed), own = the policy's aggregated training rows (both DAgger rounds). Seed means.

## Contrasts (AUC difference paired by level and seed)

| contrast | grid | offset | angle | power |
| --- | --- | --- | --- | --- |
| primary: J(v*)-J | 0.0042 [-0.0005, 0.0088] not_supported_by_this_experiment | 0.0328 [0.0136, 0.0521] supported | 0.0321 [0.0187, 0.0451] supported | -0.0299 [-0.0433, -0.0175] not_supported_by_this_experiment |
| J(b,L60)-J(a,L60) | 0.0042 [-0.0005, 0.0088] not_supported_by_this_experiment | 0.0328 [0.0136, 0.0521] supported | 0.0321 [0.0187, 0.0451] supported | -0.0299 [-0.0433, -0.0175] not_supported_by_this_experiment |
| J(c,L60)-J(a,L60) | 0.0037 [-0.0006, 0.0082] not_supported_by_this_experiment | 0.0183 [-0.0082, 0.0441] readiness_or_precision_insufficient | 0.0276 [0.0077, 0.0465] supported | -0.0339 [-0.0480, -0.0208] not_supported_by_this_experiment |
| J(d,L60)-J(a,L60) | -0.0006 [-0.0062, 0.0050] not_supported_by_this_experiment | 0.0058 [-0.0187, 0.0307] readiness_or_precision_insufficient | 0.0130 [-0.0114, 0.0368] readiness_or_precision_insufficient | -0.0174 [-0.0286, -0.0071] not_supported_by_this_experiment |
| J(a,L225)-J(a,L60) | -0.0255 [-0.0316, -0.0194] not_supported_by_this_experiment | -0.0109 [-0.0307, 0.0092] not_supported_by_this_experiment | -0.0070 [-0.0230, 0.0103] not_supported_by_this_experiment | -0.0140 [-0.0251, -0.0045] not_supported_by_this_experiment |
| J(b,L225)-J(b,L60) | -0.0266 [-0.0307, -0.0226] not_supported_by_this_experiment | -0.0636 [-0.0899, -0.0368] not_supported_by_this_experiment | -0.0432 [-0.0595, -0.0265] not_supported_by_this_experiment | 0.0248 [0.0134, 0.0372] supported |
| J(b,L225)-J(a,L60) | -0.0225 [-0.0280, -0.0170] not_supported_by_this_experiment | -0.0308 [-0.0461, -0.0145] not_supported_by_this_experiment | -0.0111 [-0.0280, 0.0065] not_supported_by_this_experiment | -0.0050 [-0.0101, 0.0000] not_supported_by_this_experiment |
| J(b,L225)-J(a,L225) | 0.0031 [0.0013, 0.0048] not_supported_by_this_experiment | -0.0199 [-0.0351, -0.0035] not_supported_by_this_experiment | -0.0041 [-0.0206, 0.0118] not_supported_by_this_experiment | 0.0089 [0.0003, 0.0194] not_supported_by_this_experiment |
| interaction input x window | -0.0011 [-0.0067, 0.0045] not_supported_by_this_experiment | -0.0527 [-0.0798, -0.0256] not_supported_by_this_experiment | -0.0363 [-0.0518, -0.0213] not_supported_by_this_experiment | 0.0388 [0.0199, 0.0607] supported |
| J_dec(v*)-J_dec(a) | -0.0041 [-0.0078, -0.0006] not_supported_by_this_experiment | 0.0165 [0.0039, 0.0292] readiness_or_precision_insufficient | 0.0077 [-0.0049, 0.0205] readiness_or_precision_insufficient | -0.0001 [-0.0105, 0.0129] not_supported_by_this_experiment |
| J_dec(v*)-J(v*,L60) | 0.0319 [0.0228, 0.0411] supported | -0.0781 [-0.1228, -0.0325] not_supported_by_this_experiment | -0.1020 [-0.1415, -0.0607] not_supported_by_this_experiment | 0.0165 [-0.0003, 0.0341] readiness_or_precision_insufficient |
| J-a-L60-F_hindsight | -0.0565 [-0.0677, -0.0458] not_supported_by_this_experiment | -0.0899 [-0.1259, -0.0553] not_supported_by_this_experiment | -0.0793 [-0.1055, -0.0540] not_supported_by_this_experiment | -0.0061 [-0.0135, 0.0007] not_supported_by_this_experiment |
| J-b-L60-F_hindsight | -0.0524 [-0.0617, -0.0431] not_supported_by_this_experiment | -0.0571 [-0.0886, -0.0272] not_supported_by_this_experiment | -0.0472 [-0.0749, -0.0210] not_supported_by_this_experiment | -0.0359 [-0.0502, -0.0220] not_supported_by_this_experiment |
| Jdec-b-F_hindsight | -0.0204 [-0.0249, -0.0159] not_supported_by_this_experiment | -0.1352 [-0.1605, -0.1105] not_supported_by_this_experiment | -0.1492 [-0.1729, -0.1255] not_supported_by_this_experiment | -0.0195 [-0.0298, -0.0093] not_supported_by_this_experiment |
| Jdec-a-F_hindsight | -0.0163 [-0.0203, -0.0122] not_supported_by_this_experiment | -0.1517 [-0.1762, -0.1275] not_supported_by_this_experiment | -0.1569 [-0.1802, -0.1348] not_supported_by_this_experiment | -0.0194 [-0.0353, -0.0064] not_supported_by_this_experiment |

## Per-inventory AUC

| arm | grid | offset | angle | power | MACs / candidate (grid) |
| --- | --- | --- | --- | --- | --- |
| F-1-continuous | 0.6772 [0.6548, 0.6994] | 0.6833 [0.6491, 0.7158] | 0.7260 [0.6959, 0.7548] | 0.7930 [0.7552, 0.8319] | 322.3M |
| F-1-micro | 0.6500 [0.6283, 0.6716] | 0.7083 [0.6833, 0.7325] | 0.7243 [0.7046, 0.7426] | 0.7912 [0.7524, 0.8302] | 404.0M |
| F-1-macro | 0.6661 [0.6434, 0.6893] | 0.5058 [0.4583, 0.5527] | 0.5823 [0.5399, 0.6247] | 0.7818 [0.7485, 0.8166] | 329.6M |
| F-5-continuous | 0.7486 [0.7286, 0.7683] | 0.8186 [0.7952, 0.8406] | 0.8347 [0.8137, 0.8553] | 0.8103 [0.7704, 0.8509] | 64.5M |
| F-5-micro | 0.7446 [0.7250, 0.7642] | 0.8421 [0.8193, 0.8636] (F_hindsight) | 0.8557 [0.8373, 0.8735] (F_hindsight) | 0.8137 [0.7761, 0.8525] (F_hindsight) | 80.8M |
| F-5-macro | 0.7426 [0.7238, 0.7620] | 0.6095 [0.5838, 0.6329] | 0.6593 [0.6398, 0.6776] | 0.8131 [0.7829, 0.8441] | 65.9M |
| F-15-continuous | 0.7622 [0.7429, 0.7811] (F_hindsight) | 0.8179 [0.7974, 0.8373] | 0.7979 [0.7804, 0.8154] | 0.8043 [0.7650, 0.8437] | 21.5M |
| F-15-micro | 0.7601 [0.7413, 0.7787] | 0.7922 [0.7720, 0.8115] | 0.7765 [0.7588, 0.7934] | 0.8078 [0.7713, 0.8457] | 26.9M |
| F-15-macro | 0.7444 [0.7269, 0.7620] | 0.6670 [0.6503, 0.6827] | 0.6791 [0.6663, 0.6912] | 0.8058 [0.7729, 0.8396] | 22.0M |
| J(a, L60) = J | 0.7057 [0.6846, 0.7267] | 0.7522 [0.7039, 0.7978] | 0.7765 [0.7402, 0.8115] | 0.8076 [0.7695, 0.8463] | 184.0M |
| J(b, L60) | 0.7099 [0.6891, 0.7304] | 0.7850 [0.7442, 0.8232] | 0.8086 [0.7717, 0.8429] | 0.7777 [0.7299, 0.8253] | 164.9M |
| J(c, L60) | 0.7094 [0.6879, 0.7308] | 0.7705 [0.7307, 0.8074] | 0.8041 [0.7685, 0.8379] | 0.7737 [0.7297, 0.8196] | 227.5M |
| J(d, L60) | 0.7052 [0.6841, 0.7261] | 0.7580 [0.7224, 0.7913] | 0.7894 [0.7592, 0.8176] | 0.7902 [0.7507, 0.8313] | 193.4M |
| J(a, L225) | 0.6802 [0.6579, 0.7022] | 0.7413 [0.6904, 0.7900] | 0.7695 [0.7248, 0.8121] | 0.7936 [0.7497, 0.8368] | 379.2M |
| J(b, L225) | 0.6833 [0.6609, 0.7052] | 0.7214 [0.6662, 0.7745] | 0.7653 [0.7226, 0.8059] | 0.8026 [0.7648, 0.8406] | 376.6M |
| J_dec(b) | 0.7418 [0.7243, 0.7595] | 0.7069 [0.6882, 0.7245] | 0.7066 [0.6905, 0.7222] | 0.7942 [0.7551, 0.8338] | 22.7M |
| J_dec(a) (#101 J_dec) | 0.7460 [0.7277, 0.7637] | 0.6904 [0.6737, 0.7062] | 0.6988 [0.6827, 0.7147] | 0.7943 [0.7555, 0.8331] | 22.7M |
| no-model action prior | 0.7203 [0.6965, 0.7442] | 0.5000 [0.5000, 0.5000] | 0.8119 [0.7778, 0.8450] | 0.5000 [0.5000, 0.5000] | n/a |

## Request mix against the true remaining frames κ (grid; share of policy steps)

| arm | κ bin | steps | Δ=1 | Δ=5 | Δ=15 | continuous | micro | macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| J(a, L60) = J | 166–225 | 401972 | 90.8% | 0.0% | 9.2% | 13.9% | 14.9% | 71.2% |
| J(a, L60) = J | 106–165 | 428910 | 92.3% | 0.0% | 7.7% | 12.4% | 7.8% | 79.8% |
| J(a, L60) = J | 61–105 | 363585 | 94.0% | 0.0% | 6.0% | 13.7% | 6.1% | 80.2% |
| J(a, L60) = J | 31–60 | 278377 | 95.7% | 0.0% | 4.3% | 10.4% | 4.3% | 85.3% |
| J(a, L60) = J | 1–30 | 335098 | 96.3% | 2.2% | 1.5% | 10.2% | 4.5% | 85.3% |
| J(b, L60) | 166–225 | 276963 | 83.4% | 0.0% | 16.6% | 48.0% | 22.8% | 29.2% |
| J(b, L60) | 106–165 | 348194 | 88.9% | 0.0% | 11.1% | 36.1% | 11.9% | 51.9% |
| J(b, L60) | 61–105 | 372607 | 94.3% | 0.0% | 5.7% | 23.6% | 5.8% | 70.7% |
| J(b, L60) | 31–60 | 288966 | 96.1% | 0.0% | 3.9% | 15.9% | 3.9% | 80.3% |
| J(b, L60) | 1–30 | 333546 | 96.3% | 2.3% | 1.5% | 16.5% | 5.0% | 78.4% |
| J(c, L60) | 166–225 | 547235 | 95.1% | 0.2% | 4.7% | 14.2% | 9.3% | 76.5% |
| J(c, L60) | 106–165 | 595396 | 96.4% | 0.1% | 3.5% | 14.7% | 3.9% | 81.4% |
| J(c, L60) | 61–105 | 463159 | 96.8% | 0.0% | 3.2% | 16.8% | 3.2% | 80.0% |
| J(c, L60) | 31–60 | 308774 | 96.8% | 0.0% | 3.2% | 16.0% | 3.2% | 80.7% |
| J(c, L60) | 1–30 | 336226 | 96.3% | 2.2% | 1.5% | 19.1% | 5.0% | 75.9% |
| J(d, L60) | 166–225 | 449975 | 91.4% | 1.4% | 7.2% | 11.0% | 13.7% | 75.3% |
| J(d, L60) | 106–165 | 496633 | 94.3% | 0.0% | 5.7% | 9.3% | 4.3% | 86.3% |
| J(d, L60) | 61–105 | 391917 | 94.9% | 0.0% | 5.1% | 9.9% | 3.9% | 86.2% |
| J(d, L60) | 31–60 | 268445 | 95.4% | 0.0% | 4.6% | 8.9% | 3.9% | 87.2% |
| J(d, L60) | 1–30 | 298112 | 94.9% | 3.0% | 2.2% | 9.9% | 5.0% | 85.1% |
| J(a, L225) | 166–225 | 870032 | 99.8% | 0.0% | 0.2% | 11.1% | 54.4% | 34.5% |
| J(a, L225) | 106–165 | 895680 | 100.0% | 0.0% | 0.0% | 0.6% | 64.9% | 34.5% |
| J(a, L225) | 61–105 | 671760 | 100.0% | 0.0% | 0.0% | 0.0% | 65.5% | 34.5% |
| J(a, L225) | 31–60 | 447840 | 100.0% | 0.0% | 0.0% | 0.0% | 63.4% | 36.6% |
| J(a, L225) | 1–30 | 447840 | 100.0% | 0.0% | 0.0% | 0.0% | 58.7% | 41.3% |
| J(b, L225) | 166–225 | 872958 | 99.8% | 0.0% | 0.2% | 12.6% | 53.2% | 34.3% |
| J(b, L225) | 106–165 | 895680 | 100.0% | 0.0% | 0.0% | 1.8% | 61.6% | 36.6% |
| J(b, L225) | 61–105 | 671760 | 100.0% | 0.0% | 0.0% | 0.0% | 59.8% | 40.2% |
| J(b, L225) | 31–60 | 447840 | 100.0% | 0.0% | 0.0% | 0.0% | 56.3% | 43.7% |
| J(b, L225) | 1–30 | 447840 | 100.0% | 0.0% | 0.0% | 0.0% | 53.9% | 46.1% |
| J_dec(b) | 166–225 | 59743 | 0.0% | 0.0% | 99.9% | 55.7% | 0.0% | 44.3% |
| J_dec(b) | 106–165 | 59712 | 0.0% | 0.0% | 100.0% | 58.1% | 0.0% | 41.9% |
| J_dec(b) | 61–105 | 44784 | 0.0% | 0.0% | 100.0% | 55.1% | 8.3% | 36.6% |
| J_dec(b) | 31–60 | 29856 | 0.0% | 0.0% | 100.0% | 46.0% | 23.8% | 30.2% |
| J_dec(b) | 1–30 | 29923 | 0.1% | 0.2% | 99.7% | 42.4% | 32.8% | 24.8% |
| J_dec(a) (#101 J_dec) | 166–225 | 59718 | 0.0% | 0.0% | 100.0% | 53.9% | 0.0% | 46.1% |
| J_dec(a) (#101 J_dec) | 106–165 | 59712 | 0.0% | 0.0% | 100.0% | 57.5% | 0.0% | 42.5% |
| J_dec(a) (#101 J_dec) | 61–105 | 44784 | 0.0% | 0.0% | 100.0% | 52.4% | 11.3% | 36.2% |
| J_dec(a) (#101 J_dec) | 31–60 | 29856 | 0.0% | 0.0% | 100.0% | 44.9% | 23.0% | 32.1% |
| J_dec(a) (#101 J_dec) | 1–30 | 29880 | 0.1% | 0.0% | 99.9% | 43.5% | 33.8% | 22.6% |

Late − early shift (κ 1–60 minus κ 166–225, grid; percentage points): 

| arm | Δ=1 | Δ=5 | Δ=15 | continuous | micro | macro |
| --- | --- | --- | --- | --- | --- | --- |
| J(a, L60) = J | +5.3 | +1.2 | -6.5 | -3.6 | -10.5 | +14.2 |
| J(b, L60) | +12.8 | +1.2 | -14.0 | -31.8 | -18.3 | +50.1 |
| J(c, L60) | +1.5 | +1.0 | -2.4 | +3.4 | -5.1 | +1.7 |
| J(d, L60) | +3.7 | +0.2 | -3.9 | -1.6 | -9.3 | +10.9 |
| J(a, L225) | +0.2 | +0.0 | -0.2 | -11.1 | +6.6 | +4.4 |
| J(b, L225) | +0.2 | +0.0 | -0.2 | -12.6 | +1.9 | +10.6 |
| J_dec(b) | +0.0 | +0.1 | -0.1 | -11.5 | +28.3 | -16.8 |
| J_dec(a) (#101 J_dec) | +0.0 | +0.0 | -0.0 | -9.7 | +28.4 | -18.7 |

## Label accuracy by κ (L225 round-0 labels, seed means)

| arm | κ 166–225 | κ 106–165 | κ 61–105 | κ 31–60 | κ 1–30 |
| --- | --- | --- | --- | --- | --- |
| J(a, L60) = J | 33.5% | 34.4% | 34.6% | 41.4% | 51.1% |
| J(b, L60) | 25.9% | 37.6% | 41.8% | 43.3% | 51.2% |
| J(c, L60) | 41.6% | 42.2% | 43.3% | 44.3% | 51.4% |
| J(d, L60) | 37.9% | 40.5% | 41.7% | 41.4% | 48.4% |
| J(a, L225) | 58.0% | 54.0% | 52.0% | 48.0% | 51.4% |
| J(b, L225) | 58.2% | 54.1% | 52.3% | 48.4% | 51.7% |
| J_dec(b) | 4.9% | 5.6% | 6.0% | 5.8% | 13.4% |
| J_dec(a) (#101 J_dec) | 3.8% | 3.7% | 3.4% | 4.1% | 14.4% |

## Wrong-anchor control (grid, 480 fit anchors)

| arm | right | wrong | right − wrong | rank stability | token |
| --- | --- | --- | --- | --- | --- |
| J(a, L60) = J | 0.7057 [0.6846, 0.7267] | 0.6721 [0.6506, 0.6935] | 0.0336 [0.0299, 0.0373] | 0.9159 [0.9129, 0.9188] | supported |
| J(b, L60) | 0.7099 [0.6891, 0.7304] | 0.6768 [0.6556, 0.6978] | 0.0331 [0.0290, 0.0373] | 0.9171 [0.9145, 0.9198] | supported |
| J(c, L60) | 0.7094 [0.6879, 0.7308] | 0.6773 [0.6556, 0.6986] | 0.0322 [0.0284, 0.0360] | 0.9125 [0.9095, 0.9154] | supported |
| J(d, L60) | 0.7052 [0.6841, 0.7261] | 0.6717 [0.6506, 0.6924] | 0.0335 [0.0275, 0.0394] | 0.8322 [0.8221, 0.8416] | supported |
| J(a, L225) | 0.6802 [0.6579, 0.7022] | 0.6787 [0.6563, 0.7009] | 0.0014 [0.0008, 0.0021] | 0.9983 [0.9983, 0.9984] | not_supported_by_this_experiment |
| J(b, L225) | 0.6833 [0.6609, 0.7052] | 0.6786 [0.6562, 0.7007] | 0.0047 [0.0032, 0.0061] | 0.9944 [0.9943, 0.9946] | not_supported_by_this_experiment |
| J_dec(b) | 0.7418 [0.7243, 0.7595] | 0.6167 [0.6009, 0.6323] | 0.1251 [0.1151, 0.1352] | 0.6001 [0.5893, 0.6111] | supported |

## Guards

- G1_isolation: pass
- G2_scoring_policy: pass
- G3_chronology: pass
- G4_checkpoint_binding: pass
- G5_replication_101_J: pass
- G6_reused_101_records: pass
- G7_selection: pass
  - G5 detail: labels True, round checkpoints True, policy True, 1536 scored blocks True.
- Typed failures: none.

## Compute

- GPU 7889 s, process wall 7895 s, engine 0 s (4 x RTX 5090 (one process per GPU)).
- 225-frame label windows: 3480 shots, 3326 reach frame 225 (mean length 222.9, min 140), 775520 round-0 label rows per group.
