# Issue #114: horizon-only versus joint request selection at matched compute (#104 sealed #109 evaluation split)

Plan frozen 2026-10-14T11:29:59Z after a passed smoke, before training and scoring. Retained scores use processes that did not train, `deterministic_scoring()` and tie-free cost. Validation: `python -u -m scripts.run_horizon_vs_joint --validate`.

**Execution deviation:** accidentally duplicated fixed-arm scoring workers were terminated. Literal scored-once compliance failed; deterministic recomputation is not new independent evidence. Numerical dispositions below use the untouched frozen G1–G6 and token rules, not a post hoc relaxed rule. See `execution-incident.json`. Timings include concurrent-worker contention and are not isolated-device latency.

**Disposition — adaptive alpha adds decision value beyond adaptive Delta at matched compute: supported** (J - J-cont (grid AUC), frontiers interpolated at the MAC budgets of J's and C-J's lambda = 1e-4 points; δ = 0.02).

| budget (MACs / candidate) | J − J-cont | token | J − C-J | token |
| --- | --- | --- | --- | --- |
| J_lambda_1e-04 (184.0M) | 0.0201 [0.0154, 0.0248] | supported | 0.0111 [0.0050, 0.0174] | not_supported_by_this_experiment |
| CJ_lambda_1e-04 (234.7M) | 0.0201 [0.0154, 0.0248] | supported | 0.0058 [0.0000, 0.0117] | not_supported_by_this_experiment |
| control: no-model action prior AUC | 0.7203 [0.6965, 0.7442] | reference | — | — |
| control: wrong-anchor J AUC (grid) | 0.6721 [0.6506, 0.6935] | reference | — | — |

Secondary — the hybrid J keeps its advantage over the horizon-only adaptive C-J at matched compute: **not_supported_by_this_experiment**.

## Per-inventory AUC (tie-free; level-clustered, DESCRIPTIVE)

| arm | grid | offset | angle | power |
| --- | --- | --- | --- | --- |
| J λ=1e-04 | 0.7057 [0.6846, 0.7267] | 0.7522 [0.7039, 0.7978] | 0.7765 [0.7402, 0.8115] | 0.8076 [0.7695, 0.8463] |
| J-cont λ=1e-04 | 0.6856 [0.6647, 0.7061] | 0.7624 [0.7290, 0.7936] | 0.7677 [0.7399, 0.7955] | 0.7804 [0.7394, 0.8227] |
| T-continuous | 0.7025 [0.6813, 0.7238] | 0.7553 [0.7289, 0.7789] | 0.7762 [0.7555, 0.7949] | 0.7955 [0.7573, 0.8350] |
| T-micro | 0.7044 [0.6827, 0.7258] | 0.7625 [0.7333, 0.7886] | 0.7809 [0.7594, 0.8010] | 0.7906 [0.7503, 0.8318] |
| T-macro | 0.6776 [0.6555, 0.7002] | 0.5373 [0.4921, 0.5804] | 0.6244 [0.5858, 0.6617] | 0.7851 [0.7522, 0.8194] |
| D-1 | 0.6851 [0.6631, 0.7069] | 0.7101 [0.6500, 0.7684] | 0.7449 [0.7000, 0.7888] | 0.7651 [0.7140, 0.8165] |
| D-5 | 0.7378 [0.7193, 0.7565] | 0.7777 [0.7487, 0.8058] | 0.7860 [0.7650, 0.8059] | 0.8023 [0.7652, 0.8406] |
| D-15 | 0.7619 [0.7433, 0.7802] | 0.8010 [0.7776, 0.8237] | 0.7850 [0.7660, 0.8029] | 0.7964 [0.7553, 0.8383] |
| C-J λ=1e-04 | 0.6999 [0.6827, 0.7170] | 0.7750 [0.7618, 0.7877] | 0.7693 [0.7561, 0.7825] | 0.7969 [0.7560, 0.8376] |
| J-as-cont λ=1e-04 | 0.7021 [0.6805, 0.7231] | 0.7197 [0.6952, 0.7417] | 0.7516 [0.7301, 0.7717] | 0.7987 [0.7617, 0.8371] |
| F-1-continuous | 0.6772 [0.6548, 0.6994] | 0.6833 [0.6491, 0.7158] | 0.7260 [0.6959, 0.7548] | 0.7930 [0.7552, 0.8319] |
| F-1-micro | 0.6500 [0.6283, 0.6716] | 0.7083 [0.6833, 0.7325] | 0.7243 [0.7046, 0.7426] | 0.7912 [0.7524, 0.8302] |
| F-1-macro | 0.6661 [0.6434, 0.6893] | 0.5058 [0.4583, 0.5527] | 0.5823 [0.5399, 0.6247] | 0.7818 [0.7485, 0.8166] |
| F-5-continuous | 0.7486 [0.7286, 0.7683] | 0.8186 [0.7952, 0.8406] | 0.8347 [0.8137, 0.8553] | 0.8103 [0.7704, 0.8509] |
| F-5-micro | 0.7446 [0.7250, 0.7642] | 0.8421 [0.8193, 0.8636] | 0.8557 [0.8373, 0.8735] | 0.8137 [0.7761, 0.8525] |
| F-5-macro | 0.7426 [0.7238, 0.7620] | 0.6095 [0.5838, 0.6329] | 0.6593 [0.6398, 0.6776] | 0.8131 [0.7829, 0.8441] |
| F-15-continuous | 0.7622 [0.7429, 0.7811] | 0.8179 [0.7974, 0.8373] | 0.7979 [0.7804, 0.8154] | 0.8043 [0.7650, 0.8437] |
| F-15-micro | 0.7601 [0.7413, 0.7787] | 0.7922 [0.7720, 0.8115] | 0.7765 [0.7588, 0.7934] | 0.8078 [0.7713, 0.8457] |
| F-15-macro | 0.7444 [0.7269, 0.7620] | 0.6670 [0.6503, 0.6827] | 0.6791 [0.6663, 0.6912] | 0.8058 [0.7729, 0.8396] |
| C-F-1 | 0.6316 [0.6141, 0.6490] | 0.7385 [0.7140, 0.7614] | 0.7347 [0.7137, 0.7546] | 0.7753 [0.7299, 0.8206] |
| C-F-5 | 0.6909 [0.6762, 0.7055] | 0.7216 [0.6939, 0.7491] | 0.7212 [0.6986, 0.7425] | 0.8054 [0.7713, 0.8406] |
| C-F-15 | 0.7834 [0.7676, 0.7992] | 0.7748 [0.7524, 0.7962] | 0.7590 [0.7422, 0.7752] | 0.8068 [0.7711, 0.8440] |
| prior | 0.7203 [0.6965, 0.7442] | 0.5000 [0.5000, 0.5000] | 0.8119 [0.7778, 0.8450] | 0.5000 [0.5000, 0.5000] |
| wrong-anchor J λ=1e-04 (grid only) | 0.6721 [0.6506, 0.6935] | — | — | — |

## Contrasts (paired by level and seed; δ rule)

| contrast | grid | offset | angle | power |
| --- | --- | --- | --- | --- |
| J-1e-04 - Jcont-1e-04 | 0.0201 [0.0154, 0.0248] supported | -0.0102 [-0.0383, 0.0183] not_supported_by_this_experiment | 0.0087 [-0.0153, 0.0323] readiness_or_precision_insufficient | 0.0272 [0.0178, 0.0366] supported |
| J-1e-04 - T-continuous | 0.0032 [-0.0075, 0.0137] not_supported_by_this_experiment | -0.0031 [-0.0491, 0.0417] readiness_or_precision_insufficient | 0.0003 [-0.0338, 0.0337] readiness_or_precision_insufficient | 0.0121 [0.0046, 0.0191] not_supported_by_this_experiment |
| J-1e-04 - CJ-1e-04 | 0.0058 [0.0000, 0.0117] not_supported_by_this_experiment | -0.0228 [-0.0644, 0.0168] not_supported_by_this_experiment | 0.0072 [-0.0235, 0.0358] readiness_or_precision_insufficient | 0.0107 [0.0017, 0.0196] not_supported_by_this_experiment |
| J-1e-04 - Jascont-1e-04 | 0.0037 [-0.0021, 0.0095] not_supported_by_this_experiment | 0.0325 [0.0013, 0.0632] supported | 0.0248 [0.0018, 0.0480] supported | 0.0089 [0.0029, 0.0145] not_supported_by_this_experiment |
| J-1e-04 - T-micro | 0.0014 [-0.0086, 0.0112] not_supported_by_this_experiment | -0.0103 [-0.0449, 0.0228] readiness_or_precision_insufficient | -0.0044 [-0.0349, 0.0264] readiness_or_precision_insufficient | 0.0170 [0.0063, 0.0297] readiness_or_precision_insufficient |
| J-1e-04 - T-macro | 0.0281 [0.0227, 0.0337] supported | 0.2149 [0.1867, 0.2421] supported | 0.1521 [0.1259, 0.1790] supported | 0.0224 [0.0101, 0.0342] supported |
| J-1e-04 - D-1 | 0.0207 [0.0173, 0.0241] supported | 0.0421 [0.0246, 0.0588] supported | 0.0315 [0.0170, 0.0459] supported | 0.0424 [0.0223, 0.0634] supported |
| J-1e-04 - D-5 | -0.0320 [-0.0416, -0.0230] not_supported_by_this_experiment | -0.0255 [-0.0829, 0.0315] readiness_or_precision_insufficient | -0.0096 [-0.0506, 0.0302] readiness_or_precision_insufficient | 0.0053 [-0.0066, 0.0167] not_supported_by_this_experiment |
| J-1e-04 - D-15 | -0.0562 [-0.0685, -0.0441] not_supported_by_this_experiment | -0.0488 [-0.1079, 0.0099] not_supported_by_this_experiment | -0.0085 [-0.0531, 0.0331] readiness_or_precision_insufficient | 0.0112 [-0.0032, 0.0269] readiness_or_precision_insufficient |
| Jcont-1e-04 - T-continuous | -0.0169 [-0.0260, -0.0082] not_supported_by_this_experiment | 0.0071 [-0.0311, 0.0439] readiness_or_precision_insufficient | -0.0084 [-0.0416, 0.0250] readiness_or_precision_insufficient | -0.0151 [-0.0238, -0.0066] not_supported_by_this_experiment |
| J-1e-04 - prior | -0.0146 [-0.0388, 0.0088] not_supported_by_this_experiment | 0.2522 [0.2039, 0.2978] supported | -0.0354 [-0.0605, -0.0089] not_supported_by_this_experiment | 0.3076 [0.2695, 0.3463] supported |
| Jcont-1e-04 - prior | -0.0347 [-0.0604, -0.0099] not_supported_by_this_experiment | 0.2624 [0.2290, 0.2936] supported | -0.0441 [-0.0709, -0.0168] not_supported_by_this_experiment | 0.2804 [0.2394, 0.3227] supported |
| CJ-1e-04 - prior | -0.0204 [-0.0425, 0.0013] not_supported_by_this_experiment | 0.2750 [0.2618, 0.2877] supported | -0.0426 [-0.0704, -0.0142] not_supported_by_this_experiment | 0.2969 [0.2560, 0.3376] supported |
| control: no-model action prior AUC | 0.7203 [0.6965, 0.7442] | 0.5000 [0.5000, 0.5000] | 0.8119 [0.7778, 0.8450] | 0.5000 [0.5000, 0.5000] |
| control: wrong-anchor J AUC | 0.6721 [0.6506, 0.6935] | grid only | grid only | grid only |

## λ-swept frontiers (grid)

| policy | λ | AUC | top-3 | MACs / candidate | calls / candidate | GPU s / state | tied cells | tied pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| J | 1e-05 | 0.7041 [0.6843, 0.7237] | 0.5070 [0.4641, 0.5509] | 181.2M | 112.0 | 0.292 | 0 | 0/18945 |
| J | 1e-04 | 0.7057 [0.6846, 0.7267] | 0.5102 [0.4620, 0.5606] | 184.0M | 121.1 | 0.339 | 0 | 0/18945 |
| J | 3e-04 | 0.7129 [0.6925, 0.7338] | 0.5198 [0.4737, 0.5681] | 85.5M | 55.2 | 0.203 | 0 | 0/18945 |
| J | 1e-03 | 0.7342 [0.7139, 0.7545] | 0.5884 [0.5413, 0.6377] | 34.4M | 21.4 | 0.056 | 0 | 0/18945 |
| J | 3e-03 | 0.7251 [0.7049, 0.7450] | 0.5627 [0.5188, 0.6077] | 33.6M | 21.6 | 0.057 | 0 | 0/18945 |
| J | 1e-02 | 0.7129 [0.6931, 0.7331] | 0.5198 [0.4780, 0.5627] | 49.5M | 33.1 | 0.099 | 0 | 0/18945 |
| J | 3e-02 | 0.7459 [0.7269, 0.7645] | 0.5713 [0.5273, 0.6152] | 25.2M | 17.1 | 0.043 | 0 | 0/18945 |
| J | 1e-01 | 0.7700 [0.7513, 0.7884] | 0.6056 [0.5595, 0.6538] | 22.1M | 15.0 | 0.037 | 0 | 0/18945 |
| J-cont | 1e-05 | 0.6877 [0.6663, 0.7090] | 0.4673 [0.4234, 0.5134] | 118.1M | 80.6 | 0.208 | 0 | 0/18945 |
| J-cont | 1e-04 | 0.6856 [0.6647, 0.7061] | 0.4662 [0.4223, 0.5123] | 122.7M | 83.8 | 0.227 | 0 | 0/18945 |
| J-cont | 3e-04 | 0.6858 [0.6646, 0.7067] | 0.4523 [0.4094, 0.4962] | 80.3M | 54.8 | 0.200 | 0 | 0/18945 |
| J-cont | 1e-03 | 0.6872 [0.6655, 0.7087] | 0.4877 [0.4416, 0.5348] | 91.9M | 62.7 | 0.210 | 0 | 0/18945 |
| J-cont | 3e-03 | 0.6906 [0.6704, 0.7107] | 0.4834 [0.4394, 0.5284] | 71.5M | 48.8 | 0.173 | 0 | 0/18945 |
| J-cont | 1e-02 | 0.6976 [0.6764, 0.7187] | 0.5145 [0.4705, 0.5606] | 33.1M | 22.6 | 0.059 | 0 | 0/18945 |
| J-cont | 3e-02 | 0.7571 [0.7378, 0.7764] | 0.5884 [0.5402, 0.6388] | 35.3M | 24.1 | 0.059 | 0 | 0/18945 |
| J-cont | 1e-01 | 0.7574 [0.7383, 0.7762] | 0.6152 [0.5681, 0.6635] | 22.1M | 15.1 | 0.037 | 0 | 0/18945 |
| C-J | 1e-05 | 0.7258 [0.7102, 0.7418] | 0.4930 [0.4566, 0.5305] | 48.0M | 25.5 | 0.037 | 0 | 0/18945 |
| C-J | 1e-04 | 0.6999 [0.6827, 0.7170] | 0.4469 [0.4148, 0.4802] | 234.7M | 124.5 | 0.168 | 0 | 0/18945 |
| C-J | 3e-04 | 0.7114 [0.6937, 0.7291] | 0.4748 [0.4384, 0.5123] | 245.7M | 130.4 | 0.172 | 0 | 0/18945 |
| C-J | 1e-03 | 0.6933 [0.6765, 0.7097] | 0.4930 [0.4555, 0.5306] | 172.9M | 91.8 | 0.123 | 0 | 0/18945 |
| C-J | 3e-03 | 0.6817 [0.6671, 0.6964] | 0.4448 [0.4126, 0.4780] | 82.2M | 43.6 | 0.070 | 0 | 0/18945 |
| C-J | 1e-02 | 0.7119 [0.6975, 0.7263] | 0.4962 [0.4609, 0.5316] | 49.6M | 26.3 | 0.044 | 0 | 0/18945 |
| C-J | 3e-02 | 0.6899 [0.6774, 0.7030] | 0.4330 [0.3998, 0.4673] | 45.2M | 24.0 | 0.035 | 0 | 0/18945 |
| C-J | 1e-01 | 0.6972 [0.6835, 0.7109] | 0.4759 [0.4405, 0.5113] | 44.2M | 23.5 | 0.035 | 0 | 0/18945 |
| no-model action prior | — | 0.7203 [0.6965, 0.7442] | 0.4180 [0.3633, 0.4727] | 0 | 0 | 0 | 0 | 42/18945 |
| wrong-anchor J λ=1e-04 | 1e-04 | 0.6721 [0.6506, 0.6935] | 0.4231 [0.3774, 0.4704] | 181.5M | — | 0.243 | — | — |
| wrong-anchor J-cont λ=1e-04 | 1e-04 | 0.6704 [0.6496, 0.6909] | 0.4224 [0.3802, 0.4661] | 123.2M | — | 0.112 | — | — |
| wrong-anchor C-J λ=1e-04 | 1e-04 | 0.6247 [0.6076, 0.6418] | 0.3648 [0.3371, 0.3923] | 231.1M | — | 0.149 | — | — |

Top-3 chance on grid: 0.254. Frontier figure: `frontier-grid.png`; machine-readable points: `frontier.json`.

Matched-compute interpolation (grid AUC at each budget; † = budget above the frontier's largest MACs, the largest-MAC point is used):

- J_lambda_1e-04 (184.0M): J 0.7057 [0.6846, 0.7267] (1.000·J λ=1e-04); J-cont† 0.6856 [0.6647, 0.7061] (1.000·J-cont λ=1e-04); C-J 0.6946 [0.6779, 0.7110] (0.797·C-J λ=1e-03 + 0.203·C-J λ=1e-04).
- CJ_lambda_1e-04 (234.7M): J† 0.7057 [0.6846, 0.7267] (1.000·J λ=1e-04); J-cont† 0.6856 [0.6647, 0.7061] (1.000·J-cont λ=1e-04); C-J 0.6999 [0.6827, 0.7170] (1.000·C-J λ=1e-04).

An endpoint clamp is an **at-most-budget contrast, not exact equal expenditure**. Both declared budgets can reuse the same endpoint pair; equal estimates at the two budgets are not independent replications. The frozen interpolation passes through every lambda point without outcome-based pruning; it is not the best-AUC-at-or-below-budget envelope and does not establish Pareto dominance. The numerical token must be read with these limitations and the execution-integrity disclosure above.

## Compute decomposition (grid, MACs per candidate and state; first − second)

| λ | contrast | Δ MACs | (a) calls (horizon) | (b) per-call α | (b') trunk | (c) policy |
| --- | --- | --- | --- | --- | --- | --- |
| 1e-05 | J λ=1e-05 − J-cont λ=1e-05 | 63.1M | 45.0M | 17.1M | 0.0M | 1.0M |
| 1e-05 | J λ=1e-05 − C-J λ=1e-05 | 133.1M | 160.3M | 17.1M | -47.0M | 2.8M |
| 1e-05 | J λ=1e-05 − J-as-cont λ=1e-05 | 20.7M | 0.0M | 17.1M | 0.0M | 3.6M |
| 1e-04 | J λ=1e-04 − J-cont λ=1e-04 | 61.2M | 53.5M | 6.6M | 0.0M | 1.2M |
| 1e-04 | J λ=1e-04 − C-J λ=1e-04 | -50.7M | -6.4M | 6.6M | -50.8M | -0.1M |
| 1e-04 | J λ=1e-04 − J-as-cont λ=1e-04 | 10.5M | 0.0M | 6.6M | 0.0M | 3.9M |
| 3e-04 | J λ=3e-04 − J-cont λ=3e-04 | 5.2M | 0.6M | 4.6M | 0.0M | 0.0M |
| 3e-04 | J λ=3e-04 − C-J λ=3e-04 | -160.2M | -139.2M | 4.6M | -23.2M | -2.4M |
| 3e-04 | J λ=3e-04 − J-as-cont λ=3e-04 | 6.4M | 0.0M | 4.6M | 0.0M | 1.8M |
| 1e-03 | J λ=1e-03 − J-cont λ=1e-03 | -57.5M | -59.2M | 3.0M | 0.0M | -1.3M |
| 1e-03 | J λ=1e-03 − C-J λ=1e-03 | -138.5M | -130.3M | 3.0M | -9.0M | -2.3M |
| 1e-03 | J λ=1e-03 − J-as-cont λ=1e-03 | 3.6M | 0.0M | 3.0M | 0.0M | 0.7M |
| 3e-03 | J λ=3e-03 − J-cont λ=3e-03 | -38.0M | -39.0M | 1.9M | 0.0M | -0.9M |
| 3e-03 | J λ=3e-03 − C-J λ=3e-03 | -48.7M | -40.7M | 1.9M | -9.1M | -0.7M |
| 3e-03 | J λ=3e-03 − J-as-cont λ=3e-03 | 2.6M | 0.0M | 1.9M | 0.0M | 0.7M |
| 1e-02 | J λ=1e-02 − J-cont λ=1e-02 | 16.4M | 15.1M | 0.9M | 0.0M | 0.3M |
| 1e-02 | J λ=1e-02 − C-J λ=1e-02 | -0.1M | 12.6M | 0.9M | -13.9M | 0.2M |
| 1e-02 | J λ=1e-02 − J-as-cont λ=1e-02 | 2.0M | 0.0M | 0.9M | 0.0M | 1.1M |
| 3e-02 | J λ=3e-02 − J-cont λ=3e-02 | -10.1M | -10.1M | 0.2M | 0.0M | -0.2M |
| 3e-02 | J λ=3e-02 − C-J λ=3e-02 | -20.0M | -12.9M | 0.2M | -7.2M | -0.2M |
| 3e-02 | J λ=3e-02 − J-as-cont λ=3e-02 | 0.8M | 0.0M | 0.2M | 0.0M | 0.5M |
| 1e-01 | J λ=1e-01 − J-cont λ=1e-01 | -0.0M | -0.2M | 0.2M | 0.0M | -0.0M |
| 1e-01 | J λ=1e-01 − C-J λ=1e-01 | -22.1M | -15.7M | 0.2M | -6.3M | -0.3M |
| 1e-01 | J λ=1e-01 − J-as-cont λ=1e-01 | 0.6M | 0.0M | 0.2M | 0.0M | 0.5M |
| control | no-model action prior: AUC 0.7203 [0.6965, 0.7442], 0 model MACs | — | — | — | — | — |
| control | wrong-anchor J: AUC 0.6721 [0.6506, 0.6935] | — | — | — | — | — |

Counterfactual J-as-cont (J's schedule with every α replaced by continuous, same calls, no policy calls), grid:

| λ | J AUC | J-as-cont AUC | J − J-as-cont | J MACs | J-as-cont MACs |
| --- | --- | --- | --- | --- | --- |
| 1e-05 | 0.7041 [0.6843, 0.7237] | 0.7103 [0.6893, 0.7310] | -0.0062 [-0.0115, -0.0008] not_supported_by_this_experiment | 181.2M | 160.5M |
| 1e-04 | 0.7057 [0.6846, 0.7267] | 0.7021 [0.6805, 0.7231] | 0.0037 [-0.0021, 0.0095] not_supported_by_this_experiment | 184.0M | 173.5M |
| 3e-04 | 0.7129 [0.6925, 0.7338] | 0.6953 [0.6741, 0.7167] | 0.0176 [0.0124, 0.0227] readiness_or_precision_insufficient | 85.5M | 79.1M |
| 1e-03 | 0.7342 [0.7139, 0.7545] | 0.7130 [0.6923, 0.7338] | 0.0212 [0.0170, 0.0254] supported | 34.4M | 30.7M |
| 3e-03 | 0.7251 [0.7049, 0.7450] | 0.7082 [0.6874, 0.7290] | 0.0168 [0.0128, 0.0208] readiness_or_precision_insufficient | 33.6M | 31.0M |
| 1e-02 | 0.7129 [0.6931, 0.7331] | 0.7034 [0.6826, 0.7246] | 0.0095 [0.0037, 0.0154] not_supported_by_this_experiment | 49.5M | 47.5M |
| 3e-02 | 0.7459 [0.7269, 0.7645] | 0.7483 [0.7290, 0.7673] | -0.0024 [-0.0057, 0.0009] not_supported_by_this_experiment | 25.2M | 24.4M |
| 1e-01 | 0.7700 [0.7513, 0.7884] | 0.7622 [0.7429, 0.7811] | 0.0078 [0.0053, 0.0102] not_supported_by_this_experiment | 22.1M | 21.5M |
| control | no-model action prior 0.7203 [0.6965, 0.7442] | — | — | 0 | — |
| control | wrong-anchor J 0.6721 [0.6506, 0.6935] | — | — | — | — |

## Schedules (grid, pooled over seeds)

| arm | modal request | non-modal step share | cells with candidate-dependent schedules |
| --- | --- | --- | --- |
| J λ=1e-04 | 1-macro | 0.210 | 760/933 |
| J-cont λ=1e-04 | 1-continuous | 0.176 | 722/933 |
| C-J λ=1e-04 | 1-continuous | 0.061 | 426/933 |
| T-continuous | 1-continuous | 0.050 | 403/933 |
| T-micro | 1-micro | 0.111 | 429/933 |
| T-macro | 1-macro | 0.020 | 461/933 |
| D-1 | 1-macro | 0.410 | 520/933 |
| D-5 | 5-micro | 0.558 | 789/933 |
| D-15 | 15-micro | 0.370 | 604/933 |
| control: no-model action prior AUC 0.7203 [0.6965, 0.7442] | no predictor | — | — |
| control: wrong-anchor J AUC 0.6721 [0.6506, 0.6935] | wrong-anchor diagnostic | — | — |

## Controls

| policy (λ = 1e-4) | right anchor | wrong anchors (480 fit) | right − wrong | token | stability |
| --- | --- | --- | --- | --- | --- |
| J λ=1e-04 | 0.7057 [0.6846, 0.7267] | 0.6721 [0.6506, 0.6935] | 0.0336 [0.0299, 0.0373] | supported | 0.9159 [0.9129, 0.9188] |
| J-cont λ=1e-04 | 0.6856 [0.6647, 0.7061] | 0.6704 [0.6496, 0.6909] | 0.0152 [0.0112, 0.0192] | not_supported_by_this_experiment | 0.9055 [0.9028, 0.9081] |
| C-J λ=1e-04 | 0.6999 [0.6827, 0.7170] | 0.6247 [0.6076, 0.6418] | 0.0753 [0.0686, 0.0821] | supported | 0.8156 [0.8105, 0.8207] |
| no-model action prior | 0.7203 [0.6965, 0.7442] | anchor independent | — | reference | — |

No-model action prior (grid): 0.7203 [0.6965, 0.7442].

## Guards

- G1_isolation: True
- G2_scoring_policy: True
- G3_chronology: True
- G4_checkpoint_binding: True
- G5_replication_101_gate_b: True
- G6_restricted_schedules: True

Compute: 9.57 GPU h, 9.57 process wall h (RTX 5090 (GPUs 0 and 1, two processes per GPU)); reused #101 J training 0.60 GPU h; 0 engine seconds.
Discarded-run compute is an explicitly labelled conservative allocated-process upper bound in `ledger/operational-discarded-upper-bound.json`, not measured CUDA active time.
