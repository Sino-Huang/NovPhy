# Issue #101: decision-aware request policy on the #104 sealed #109 evaluation split

Plan frozen 2026-10-14T10:04:00Z after a passed smoke, before any training or evaluation score; scored once in processes that did not train, under `deterministic_scoring()`. Validation: `python -u -m scripts.run_decision_aware_policy --validate`.

**Primary (grid): J_dec − F_hindsight = -0.0163 [-0.0203, -0.0122] over 933 cells, 311 levels; not_supported_by_this_experiment** (δ = 0.02; F_hindsight = F-15-continuous). J_dec = `J-dec-1e-01`, J (ICLR) = `J-pe-1e-04`, J_pg = `J-pg-1e-01`.

## Contrasts per inventory (paired AUC differences, level-clustered, DESCRIPTIVE)

| contrast | grid | offset | angle | power |
| --- | --- | --- | --- | --- |
| J_dec − F_hindsight | -0.0163 [-0.0203, -0.0122] not_supported_by_this_experiment | -0.1517 [-0.1762, -0.1275] not_supported_by_this_experiment | -0.1569 [-0.1802, -0.1348] not_supported_by_this_experiment | -0.0194 [-0.0353, -0.0064] not_supported_by_this_experiment |
| J_dec − F* | 0.0014 [-0.0050, 0.0079] not_supported_by_this_experiment | -0.1517 [-0.1762, -0.1275] not_supported_by_this_experiment | -0.1569 [-0.1802, -0.1348] not_supported_by_this_experiment | -0.0194 [-0.0353, -0.0064] not_supported_by_this_experiment |
| J_dec − M_dec | 0.0005 [-0.0005, 0.0018] not_supported_by_this_experiment | 0.0011 [-0.0039, 0.0055] not_supported_by_this_experiment | 0.0033 [-0.0019, 0.0091] not_supported_by_this_experiment | 0.0004 [-0.0007, 0.0015] not_supported_by_this_experiment |
| J_dec − OL_dec | -0.0031 [-0.0064, 0.0003] not_supported_by_this_experiment | -0.0231 [-0.0324, -0.0144] not_supported_by_this_experiment | -0.0225 [-0.0317, -0.0134] not_supported_by_this_experiment | -0.0120 [-0.0248, -0.0028] not_supported_by_this_experiment |
| J_dec − J | 0.0403 [0.0295, 0.0516] supported | -0.0617 [-0.1144, -0.0091] not_supported_by_this_experiment | -0.0776 [-0.1179, -0.0370] not_supported_by_this_experiment | -0.0133 [-0.0314, 0.0023] not_supported_by_this_experiment |
| J − F_hindsight | -0.0565 [-0.0677, -0.0458] not_supported_by_this_experiment | -0.0899 [-0.1259, -0.0553] not_supported_by_this_experiment | -0.0793 [-0.1055, -0.0540] not_supported_by_this_experiment | -0.0061 [-0.0135, 0.0007] not_supported_by_this_experiment |
| J − M | -0.0006 [-0.0032, 0.0019] not_supported_by_this_experiment | 0.0385 [0.0145, 0.0625] supported | -0.0022 [-0.0182, 0.0145] not_supported_by_this_experiment | 0.0315 [0.0145, 0.0502] supported |
| J − OL | 0.0127 [0.0098, 0.0156] not_supported_by_this_experiment | 0.0798 [0.0610, 0.0982] supported | 0.0495 [0.0336, 0.0652] supported | 0.0113 [0.0028, 0.0192] not_supported_by_this_experiment |
| J_pg − F_hindsight | -0.0011 [-0.0031, 0.0009] not_supported_by_this_experiment | -0.0326 [-0.0596, -0.0058] not_supported_by_this_experiment | -0.0635 [-0.0872, -0.0393] not_supported_by_this_experiment | -0.0078 [-0.0149, 0.0000] not_supported_by_this_experiment |
| J_pg − J_dec | 0.0151 [0.0108, 0.0195] not_supported_by_this_experiment | 0.1191 [0.1075, 0.1307] supported | 0.0934 [0.0806, 0.1065] supported | 0.0116 [-0.0000, 0.0261] readiness_or_precision_insufficient |
| J_dec − prior | 0.0257 [-0.0002, 0.0512] readiness_or_precision_insufficient | 0.1904 [0.1737, 0.2062] supported | -0.1131 [-0.1510, -0.0746] not_supported_by_this_experiment | 0.2943 [0.2555, 0.3331] supported |
| J_dec − C* | -0.0374 [-0.0439, -0.0309] not_supported_by_this_experiment | -0.0843 [-0.1028, -0.0662] not_supported_by_this_experiment | -0.0602 [-0.0793, -0.0420] not_supported_by_this_experiment | -0.0125 [-0.0281, -0.0002] not_supported_by_this_experiment |

Targets: grid: F_hindsight F-15-continuous, F* (cross-fit r†) F-5-micro; offset: F_hindsight F-5-micro, F* (cross-fit r†) F-5-micro; angle: F_hindsight F-5-micro, F* (cross-fit r†) F-5-micro; power: F_hindsight F-5-micro, F* (cross-fit r†) F-5-micro.

F* is cross-inventory selection (the #96 rule: best fixed request on the other three inventories), not member-disjoint cross-fitting: the same levels appear in several inventories, so selection optimism is not removed, and every F* and F_hindsight interval is conditional on the selection (the selection is not repeated inside the bootstrap).

## Rankers (grid)

| arm | AUC | MACs / candidate | tied pair share | nonfinite share |
| --- | --- | --- | --- | --- |
| F-1-continuous | 0.6772 [0.6548, 0.6994] | 322.3M | 0.0000 | 0.0000 |
| F-1-micro | 0.6500 [0.6283, 0.6716] | 404.0M | 0.0000 | 0.0000 |
| F-1-macro | 0.6661 [0.6434, 0.6893] | 329.6M | 0.0000 | 0.0000 |
| F-5-continuous | 0.7486 [0.7286, 0.7683] | 64.5M | 0.0000 | 0.0000 |
| F-5-micro | 0.7446 [0.7250, 0.7642] | 80.8M | 0.0000 | 0.0000 |
| F-5-macro | 0.7426 [0.7238, 0.7620] | 65.9M | 0.0000 | 0.0000 |
| F-15-continuous | 0.7622 [0.7429, 0.7811] | 21.5M | 0.0000 | 0.0000 |
| F-15-micro | 0.7601 [0.7413, 0.7787] | 26.9M | 0.0000 | 0.0000 |
| F-15-macro | 0.7444 [0.7269, 0.7620] | 22.0M | 0.0000 | 0.0000 |
| C-F-15 | 0.7834 [0.7676, 0.7992] | 27.8M | 0.0000 | 0.0000 |
| J-pe-1e-04 | 0.7057 [0.6846, 0.7267] | 184.0M | 0.0000 | 0.0000 |
| M-pe | 0.7063 [0.6851, 0.7278] | 175.4M | 0.0000 | 0.0000 |
| OL-pe | 0.6931 [0.6715, 0.7145] | 185.1M | 0.0000 | 0.0000 |
| J-dec-1e-01 | 0.7460 [0.7277, 0.7637] | 22.7M | 0.0000 | 0.0000 |
| M-dec | 0.7454 [0.7273, 0.7631] | 22.7M | 0.0000 | 0.0000 |
| OL-dec | 0.7491 [0.7309, 0.7674] | 22.2M | 0.0000 | 0.0000 |
| J-pg-1e-01 | 0.7611 [0.7419, 0.7801] | 22.1M | 0.0000 | 0.0000 |
| prior | 0.7203 [0.6965, 0.7442] | n/a | 0.0022 | 0.0000 |

## λ sweep (grid): MAC vs AUC for each policy type

| type | λ | AUC | MACs / candidate | − F_hindsight | policy-split AUC (seed mean) |
| --- | --- | --- | --- | --- | --- |
| pe | 1e-05 | 0.7041 [0.6843, 0.7237] | 181.2M | -0.0581 [-0.0661, -0.0502] | 0.7098 |
| pe | 1e-04 | 0.7057 [0.6846, 0.7267] | 184.0M | -0.0565 [-0.0677, -0.0458] | 0.7208 |
| pe | 1e-03 | 0.7342 [0.7139, 0.7545] | 34.4M | -0.0280 [-0.0324, -0.0237] | 0.7802 |
| pe | 1e-02 | 0.7129 [0.6931, 0.7331] | 49.5M | -0.0493 [-0.0588, -0.0398] | 0.7219 |
| dec | 1e-04 | 0.6807 [0.6622, 0.6996] | 46.3M | -0.0816 [-0.0948, -0.0686] | 0.6619 |
| dec | 1e-03 | 0.6898 [0.6706, 0.7091] | 39.3M | -0.0725 [-0.0819, -0.0629] | 0.6651 |
| dec | 1e-02 | 0.7060 [0.6864, 0.7256] | 47.4M | -0.0562 [-0.0626, -0.0497] | 0.6828 |
| dec | 1e-01 | 0.7460 [0.7277, 0.7637] | 22.7M | -0.0163 [-0.0203, -0.0122] | 0.7336 |
| pg | 1e-04 | 0.7643 [0.7453, 0.7834] | 45.2M | 0.0021 [-0.0030, 0.0070] | 0.7765 |
| pg | 1e-03 | 0.7530 [0.7336, 0.7721] | 25.7M | -0.0093 [-0.0121, -0.0065] | 0.7751 |
| pg | 1e-02 | 0.7611 [0.7419, 0.7801] | 22.1M | -0.0011 [-0.0031, 0.0009] | 0.7844 |
| pg | 1e-01 | 0.7611 [0.7419, 0.7801] | 22.1M | -0.0011 [-0.0031, 0.0009] | 0.7844 |

## Schedules (grid; pooled over seeds)

| arm | modal request | non-modal step share | cells with candidate-dependent schedules |
| --- | --- | --- | --- |
| J-pe-1e-04 | 1-macro | 0.210 | 760/933 |
| J-dec-1e-01 | 15-continuous | 0.480 | 732/933 |
| J-pg-1e-01 | 15-continuous | 0.333 | 0/933 |
| M-pe | 1-macro | 0.262 | 863/933 |
| M-dec | 15-continuous | 0.485 | 753/933 |

Wrong-anchor control of J_dec (480 fit-split anchors): right 0.7460 [0.7277, 0.7637], wrong 0.6181 [0.6023, 0.6337], right − wrong 0.1279 [0.1172, 0.1387] (supported); stability 0.5980 [0.5865, 0.6095].

## Guards

- G1_isolation: True
- G2_scoring_policy: True
- G3_chronology: True
- G4_checkpoint_binding: True
- G5_gate_b_replication: True

Compute: 2.96 GPU h, 2.96 process wall h (4 x RTX 5090 (one process per GPU)); 0 engine seconds.
