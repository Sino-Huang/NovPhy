# Issue #103: engine-truth compute–quality frontier of the request policies

Frozen 2026-10-14T15:00:15Z. Validation: `python -u -m scripts.run_engine_truth_frontier --validate`. Guards all pass. Zero engine seconds. Intervals are member (level)-clustered percentile bootstraps (10 000 draws, PCG64 7201), DESCRIPTIVE. δ = 0.02: `supported` iff mean ≥ δ and lower > 0; `not_supported_by_this_experiment` iff upper < δ; otherwise `readiness_or_precision_insufficient`. A positive difference below δ is not a win.

## The ICLR sentence on engine truth

Sentence: “matches a horizon-only adaptive baseline at a third of its compute”. The #74 proxy basis: hybrid adaptive 118.7M vs continuous adaptive 356.0M MACs per candidate-state (ratio 0.33), regret on the construct-invalidated count cost.

- **primary** (Gate-B grid): J-1e-04 / CJ-1e-04 MAC ratio 0.784, wall ratio 16.738; AUC J-1e-04 − CJ-1e-04 = 0.0058 [0.0000, 0.0117]; a third: False; non-inferior (lower > −δ): True → **`not_supported_by_this_experiment`**.
  - Frontier status: J-1e-04 dominated by C-F-15 (33 dominating; C-F-15 - J-1e-04 = 0.0777 [0.0655, 0.0902], supported; bootstrap frontier share 0.00); CJ-1e-04 dominated by C-F-15 (39 dominating; C-F-15 - CJ-1e-04 = 0.0834 [0.0729, 0.0941], supported; bootstrap frontier share 0.00).
  - Other λ (descriptive): the cheapest J point non-inferior to CJ-1e-04 is J-1e-01 at MAC ratio 0.094 (AUC difference 0.0701 [0.0598, 0.0801]).
- **secondary** (#93 grid): J-1e-04 / CJ-1e-04 MAC ratio 0.786, wall ratio 1.657; AUC J-1e-04 − CJ-1e-04 = 0.0219 [-0.0042, 0.0524]; a third: False; non-inferior (lower > −δ): True → **`not_supported_by_this_experiment`**.
  - Frontier status: J-1e-04 dominated by C-F-15 (30 dominating; C-F-15 - J-1e-04 = 0.0738 [0.0197, 0.1358], supported; bootstrap frontier share 0.00); CJ-1e-04 dominated by C-F-15 (42 dominating; C-F-15 - CJ-1e-04 = 0.0957 [0.0482, 0.1477], supported; bootstrap frontier share 0.00).
  - Other λ (descriptive): the cheapest J point non-inferior to CJ-1e-04 is J-1e-01 at MAC ratio 0.095 (AUC difference 0.0837 [0.0460, 0.1228]).

Rule (frozen): 'matches a horizon-only adaptive baseline at a third of its compute' on the primary grid with the transferred ICLR operating recipe: J = J-1e-04 against the horizon-only adaptive C-J = CJ-1e-04. 'a third' holds iff J's MACs per candidate-state <= 0.4 x C-J's; 'matches' holds iff the paired J - C-J AUC lower bound > -0.02 (non-inferiority at delta). supported iff both; not_supported_by_this_experiment iff 'a third' fails or the upper bound < -0.02; else readiness_or_precision_insufficient; any failed guard -> readiness_or_precision_insufficient. The other J lambdas and the frontier status are reported DESCRIPTIVE. The same rule is reported on the secondary set (DESCRIPTIVE). If the primary token is not 'supported', a MANUSCRIPT ticket is opened (#103 acceptance).

## Primary set: #109 evaluation split, Gate-B grid (311 levels × 3 seeds)

MACs: linear MACs per candidate rollout to step 225 (predictor + policy calls), retained records, all seeds. Wall: synchronized rollout seconds per state (all candidates batched), the frozen uniform timing pass, seed 20260908, one process per RTX 5090, warm-up cell discarded (the post-freeze three-seed re-timing is post-hoc DESCRIPTIVE, below). Frontier column: on/off for AUC×MACs (with the bootstrap frontier share), AUC×wall, top-3×MACs, top-3×wall. Verdict: the strongest dominator (highest AUC among policies with no more MACs), dominator − policy AUC, δ token, bootstrap frontier share.

| Policy | Group | AUC | Top-3 | MACs / cand. | Wall s / state | All-tied cells | Tied pairs | Frontier (AUC×MACs, share; AUC×wall; top-3×MACs; top-3×wall) | Verdict (AUC × MACs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F-1-continuous | fixed request (hybrid) | 0.6772 [0.6548, 0.6994] | 0.4330 [0.3880, 0.4791] | 322.3M | 0.147 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.1062 [0.0915, 0.1215], supported; 0.00) |
| F-1-micro | fixed request (hybrid) | 0.6500 [0.6283, 0.6716] | 0.4223 [0.3783, 0.4684] | 404.0M | 0.187 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.1333 [0.1153, 0.1508], supported; 0.00) |
| F-1-macro | fixed request (hybrid) | 0.6661 [0.6434, 0.6893] | 0.4437 [0.3891, 0.4984] | 329.6M | 0.165 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.1173 [0.1025, 0.1327], supported; 0.00) |
| F-5-continuous | fixed request (hybrid) | 0.7486 [0.7286, 0.7683] | 0.5863 [0.5370, 0.6367] | 64.5M | 0.030 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0348 [0.0259, 0.0439], supported; 0.00) |
| F-5-micro | fixed request (hybrid) | 0.7446 [0.7250, 0.7642] | 0.5659 [0.5198, 0.6141] | 80.8M | 0.037 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0388 [0.0296, 0.0479], supported; 0.00) |
| F-5-macro | fixed request (hybrid) | 0.7426 [0.7238, 0.7620] | 0.5348 [0.4877, 0.5841] | 65.9M | 0.033 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0408 [0.0335, 0.0482], supported; 0.00) |
| F-15-continuous | fixed request (hybrid) | 0.7622 [0.7429, 0.7811] | 0.6184 [0.5723, 0.6667] | 21.5M | 0.010 | 0/933 | 0/18945 | **on** 1.00; off; **on**; **on** | **on the frontier** (1.00) |
| F-15-micro | fixed request (hybrid) | 0.7601 [0.7413, 0.7787] | 0.5927 [0.5466, 0.6409] | 26.9M | 0.013 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by J-1e-01 (0.0099 [0.0067, 0.0130], not_supported_by_this_experiment; 0.00) |
| F-15-macro | fixed request (hybrid) | 0.7444 [0.7269, 0.7620] | 0.5488 [0.5048, 0.5938] | 22.0M | 0.011 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by F-15-continuous (0.0178 [0.0121, 0.0236], readiness_or_precision_insufficient; 0.00) |
| C-F-1 | continuous-only fixed Delta (C*) | 0.6316 [0.6141, 0.6490] | 0.3526 [0.3280, 0.3783] | 416.7M | 0.113 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.1518 [0.1358, 0.1679], supported; 0.00) |
| C-F-5 | continuous-only fixed Delta (C*) | 0.6909 [0.6762, 0.7055] | 0.4480 [0.4148, 0.4812] | 83.3M | 0.023 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0924 [0.0843, 0.1007], supported; 0.00) |
| C-F-15 | continuous-only fixed Delta (C*) | 0.7834 [0.7676, 0.7992] | 0.6024 [0.5616, 0.6431] | 27.8M | 0.008 | 0/933 | 0/18945 | **on** 1.00; **on**; off; **on** | **on the frontier** (1.00) |
| CJ-1e-05 | horizon-only adaptive C-J (C*) | 0.7258 [0.7102, 0.7418] | 0.4930 [0.4566, 0.5305] | 48.0M | 0.015 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0576 [0.0486, 0.0668], supported; 0.00) |
| CJ-1e-04 | horizon-only adaptive C-J (C*) | 0.6999 [0.6827, 0.7170] | 0.4469 [0.4148, 0.4802] | 234.7M | 0.014 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0834 [0.0729, 0.0941], supported; 0.00) |
| CJ-3e-04 | horizon-only adaptive C-J (C*) | 0.7114 [0.6937, 0.7291] | 0.4748 [0.4384, 0.5123] | 245.7M | 0.014 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0719 [0.0612, 0.0827], supported; 0.00) |
| CJ-1e-03 | horizon-only adaptive C-J (C*) | 0.6933 [0.6765, 0.7097] | 0.4930 [0.4555, 0.5306] | 172.9M | 0.014 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0901 [0.0787, 0.1017], supported; 0.00) |
| CJ-3e-03 | horizon-only adaptive C-J (C*) | 0.6817 [0.6671, 0.6964] | 0.4448 [0.4126, 0.4780] | 82.2M | 0.014 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.1017 [0.0920, 0.1114], supported; 0.00) |
| CJ-1e-02 | horizon-only adaptive C-J (C*) | 0.7119 [0.6975, 0.7263] | 0.4962 [0.4609, 0.5316] | 49.6M | 0.011 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0715 [0.0632, 0.0799], supported; 0.00) |
| CJ-3e-02 | horizon-only adaptive C-J (C*) | 0.6899 [0.6774, 0.7030] | 0.4330 [0.3998, 0.4673] | 45.2M | 0.011 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0935 [0.0842, 0.1031], supported; 0.00) |
| CJ-1e-01 | horizon-only adaptive C-J (C*) | 0.6972 [0.6835, 0.7109] | 0.4759 [0.4405, 0.5113] | 44.2M | 0.011 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0862 [0.0768, 0.0954], supported; 0.00) |
| J-1e-05 | J (prediction-error DP) | 0.7041 [0.6843, 0.7237] | 0.5070 [0.4641, 0.5509] | 181.2M | 0.183 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0792 [0.0700, 0.0888], supported; 0.00) |
| J-1e-04 | J (prediction-error DP) | 0.7057 [0.6846, 0.7267] | 0.5102 [0.4620, 0.5606] | 184.0M | 0.242 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0777 [0.0655, 0.0902], supported; 0.00) |
| J-3e-04 | J (prediction-error DP) | 0.7129 [0.6925, 0.7338] | 0.5198 [0.4737, 0.5681] | 85.5M | 0.028 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0705 [0.0588, 0.0820], supported; 0.00) |
| J-1e-03 | J (prediction-error DP) | 0.7342 [0.7139, 0.7545] | 0.5884 [0.5413, 0.6377] | 34.4M | 0.021 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0491 [0.0387, 0.0595], supported; 0.00) |
| J-3e-03 | J (prediction-error DP) | 0.7251 [0.7049, 0.7450] | 0.5627 [0.5188, 0.6077] | 33.6M | 0.021 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0583 [0.0468, 0.0694], supported; 0.00) |
| J-1e-02 | J (prediction-error DP) | 0.7129 [0.6931, 0.7331] | 0.5198 [0.4780, 0.5627] | 49.5M | 0.035 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0704 [0.0580, 0.0826], supported; 0.00) |
| J-3e-02 | J (prediction-error DP) | 0.7459 [0.7269, 0.7645] | 0.5713 [0.5273, 0.6152] | 25.2M | 0.017 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by J-1e-01 (0.0241 [0.0200, 0.0282], supported; 0.00) |
| J-1e-01 | J (prediction-error DP) | 0.7700 [0.7513, 0.7884] | 0.6056 [0.5595, 0.6538] | 22.1M | 0.013 | 0/933 | 0/18945 | **on** 1.00; off; off; off | **on the frontier** (1.00) |
| Jcont-1e-05 | J-cont (continuous requests, hybrid) | 0.6877 [0.6663, 0.7090] | 0.4673 [0.4234, 0.5134] | 118.1M | 0.033 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0957 [0.0836, 0.1082], supported; 0.00) |
| Jcont-1e-04 | J-cont (continuous requests, hybrid) | 0.6856 [0.6647, 0.7061] | 0.4662 [0.4223, 0.5123] | 122.7M | 0.048 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0978 [0.0851, 0.1114], supported; 0.00) |
| Jcont-3e-04 | J-cont (continuous requests, hybrid) | 0.6858 [0.6646, 0.7067] | 0.4523 [0.4094, 0.4962] | 80.3M | 0.053 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0976 [0.0852, 0.1104], supported; 0.00) |
| Jcont-1e-03 | J-cont (continuous requests, hybrid) | 0.6872 [0.6655, 0.7087] | 0.4877 [0.4416, 0.5348] | 91.9M | 0.038 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0962 [0.0830, 0.1098], supported; 0.00) |
| Jcont-3e-03 | J-cont (continuous requests, hybrid) | 0.6906 [0.6704, 0.7107] | 0.4834 [0.4394, 0.5284] | 71.5M | 0.021 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0928 [0.0811, 0.1047], supported; 0.00) |
| Jcont-1e-02 | J-cont (continuous requests, hybrid) | 0.6976 [0.6764, 0.7187] | 0.5145 [0.4705, 0.5606] | 33.1M | 0.029 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0858 [0.0737, 0.0979], supported; 0.00) |
| Jcont-3e-02 | J-cont (continuous requests, hybrid) | 0.7571 [0.7378, 0.7764] | 0.5884 [0.5402, 0.6388] | 35.3M | 0.037 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0263 [0.0184, 0.0343], supported; 0.00) |
| Jcont-1e-01 | J-cont (continuous requests, hybrid) | 0.7574 [0.7383, 0.7762] | 0.6152 [0.5681, 0.6635] | 22.1M | 0.013 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by J-1e-01 (0.0126 [0.0095, 0.0158], not_supported_by_this_experiment; 0.00) |
| T-continuous | J restricted (T-alpha / D-Delta) | 0.7025 [0.6813, 0.7238] | 0.5348 [0.4909, 0.5809] | 196.1M | 0.139 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0809 [0.0677, 0.0941], supported; 0.00) |
| T-micro | J restricted (T-alpha / D-Delta) | 0.7044 [0.6827, 0.7258] | 0.4952 [0.4480, 0.5434] | 167.1M | 0.251 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0790 [0.0658, 0.0923], supported; 0.00) |
| T-macro | J restricted (T-alpha / D-Delta) | 0.6776 [0.6555, 0.7002] | 0.4630 [0.4105, 0.5166] | 280.0M | 0.150 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.1058 [0.0922, 0.1199], supported; 0.00) |
| D-1 | J restricted (T-alpha / D-Delta) | 0.6851 [0.6631, 0.7069] | 0.4362 [0.3837, 0.4898] | 343.0M | 0.303 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0983 [0.0841, 0.1130], supported; 0.00) |
| D-5 | J restricted (T-alpha / D-Delta) | 0.7378 [0.7193, 0.7565] | 0.5498 [0.5048, 0.5959] | 73.6M | 0.056 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0456 [0.0356, 0.0556], supported; 0.00) |
| D-15 | J restricted (T-alpha / D-Delta) | 0.7619 [0.7433, 0.7802] | 0.5916 [0.5445, 0.6409] | 25.5M | 0.017 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by J-1e-01 (0.0081 [0.0052, 0.0110], not_supported_by_this_experiment; 0.00) |
| J-dec-1e-04 | J_dec (decision-valued DP) | 0.6807 [0.6622, 0.6996] | 0.3976 [0.3580, 0.4394] | 46.3M | 0.041 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.1027 [0.0897, 0.1160], supported; 0.00) |
| J-dec-1e-03 | J_dec (decision-valued DP) | 0.6898 [0.6706, 0.7091] | 0.4062 [0.3655, 0.4469] | 39.3M | 0.041 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0936 [0.0816, 0.1059], supported; 0.00) |
| J-dec-1e-02 | J_dec (decision-valued DP) | 0.7060 [0.6864, 0.7256] | 0.4962 [0.4577, 0.5338] | 47.4M | 0.072 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0773 [0.0669, 0.0875], supported; 0.00) |
| J-dec-1e-01 | J_dec (decision-valued DP) | 0.7460 [0.7277, 0.7637] | 0.5756 [0.5305, 0.6217] | 22.7M | 0.013 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by J-1e-01 (0.0240 [0.0208, 0.0273], supported; 0.00) |
| J-pg-1e-04 | J_pg (policy gradient) | 0.7643 [0.7453, 0.7834] | 0.5938 [0.5445, 0.6442] | 45.2M | 0.037 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0191 [0.0120, 0.0262], readiness_or_precision_insufficient; 0.00) |
| J-pg-1e-03 | J_pg (policy gradient) | 0.7530 [0.7336, 0.7721] | 0.5949 [0.5488, 0.6420] | 25.7M | 0.015 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by J-1e-01 (0.0171 [0.0128, 0.0213], readiness_or_precision_insufficient; 0.00) |
| J-pg-1e-02 | J_pg (policy gradient) | 0.7611 [0.7419, 0.7801] | 0.6109 [0.5648, 0.6581] | 22.1M | 0.013 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by J-1e-01 (0.0089 [0.0055, 0.0123], not_supported_by_this_experiment; 0.00) |
| J-pg-1e-01 | J_pg (policy gradient) | 0.7611 [0.7419, 0.7801] | 0.6109 [0.5648, 0.6581] | 22.1M | 0.014 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by J-1e-01 (0.0089 [0.0055, 0.0123], not_supported_by_this_experiment; 0.00) |
| M-pe | factorized / open-loop (#101) | 0.7063 [0.6851, 0.7278] | 0.5005 [0.4512, 0.5520] | 175.4M | 0.254 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0770 [0.0645, 0.0899], supported; 0.00) |
| M-dec | factorized / open-loop (#101) | 0.7454 [0.7273, 0.7631] | 0.5798 [0.5348, 0.6259] | 22.7M | 0.015 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by J-1e-01 (0.0246 [0.0213, 0.0279], supported; 0.00) |
| OL-pe | factorized / open-loop (#101) | 0.6931 [0.6715, 0.7145] | 0.4877 [0.4362, 0.5402] | 185.1M | 0.147 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0903 [0.0780, 0.1031], supported; 0.00) |
| OL-dec | factorized / open-loop (#101) | 0.7491 [0.7309, 0.7674] | 0.5798 [0.5359, 0.6249] | 22.2M | 0.011 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by J-1e-01 (0.0209 [0.0177, 0.0243], supported; 0.00) |
| J_diag | #105 tied-axes restriction | 0.7125 [0.6922, 0.7329] | 0.5788 [0.5295, 0.6292] | 110.5M | 0.123 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0708 [0.0599, 0.0817], supported; 0.00) |
| J_monotone | #105 tied-axes restriction | 0.7029 [0.6818, 0.7241] | 0.4920 [0.4427, 0.5423] | 186.7M | 0.269 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0804 [0.0683, 0.0930], supported; 0.00) |
| J_anti | #105 tied-axes restriction | 0.7001 [0.6790, 0.7212] | 0.4973 [0.4469, 0.5488] | 221.3M | 0.206 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0833 [0.0719, 0.0950], supported; 0.00) |
| J_dec_diag | #105 tied-axes restriction | 0.6946 [0.6786, 0.7111] | 0.4920 [0.4577, 0.5273] | 26.6M | 0.033 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by J-1e-01 (0.0754 [0.0592, 0.0912], supported; 0.00) |
| J_dec_monotone | #105 tied-axes restriction | 0.7459 [0.7277, 0.7637] | 0.5756 [0.5305, 0.6217] | 22.7M | 0.015 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by J-1e-01 (0.0241 [0.0208, 0.0274], supported; 0.00) |
| J_dec_anti | #105 tied-axes restriction | 0.7622 [0.7429, 0.7811] | 0.6184 [0.5723, 0.6667] | 22.1M | 0.014 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by F-15-continuous (0.0000 [0.0000, 0.0000], not_supported_by_this_experiment; 0.00) |
| H-1-continuous | #105 hierarchical level | 0.5080 [0.4953, 0.5205] | 0.2154 [0.1897, 0.2412] | 410.8M | 0.224 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.2754 [0.2565, 0.2950], supported; 0.00) |
| H-5-micro | #105 hierarchical level | 0.7063 [0.6875, 0.7251] | 0.4941 [0.4491, 0.5402] | 58.5M | 0.036 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.0770 [0.0642, 0.0893], supported; 0.00) |
| H-15-macro | #105 hierarchical level | 0.4159 [0.4021, 0.4299] | 0.0686 [0.0536, 0.0836] | 7.9M | 0.006 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by prior (0.3044 [0.2725, 0.3365], supported; 0.00) |
| TAWM-1 | #78 port (#71 carrier) | 0.4587 [0.4414, 0.4763] | 0.1329 [0.1104, 0.1565] | 416.7M | 0.111 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.3246 [0.2958, 0.3542], supported; 0.00) |
| TAWM-5 | #78 port (#71 carrier) | 0.6007 [0.5757, 0.6260] | 0.3687 [0.3215, 0.4159] | 83.3M | 0.022 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.1827 [0.1627, 0.2022], supported; 0.00) |
| TAWM-15 | #78 port (#71 carrier) | 0.6103 [0.5846, 0.6361] | 0.4116 [0.3569, 0.4662] | 27.8M | 0.008 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.1731 [0.1527, 0.1927], supported; 0.00) |
| VLWM-1 | #78 port (#71 carrier) | 0.4480 [0.4257, 0.4695] | 0.1404 [0.1158, 0.1661] | 416.7M | 0.113 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.3354 [0.3047, 0.3662], supported; 0.00) |
| VLWM-5 | #78 port (#71 carrier) | 0.5520 [0.5423, 0.5618] | 0.2379 [0.2111, 0.2658] | 83.3M | 0.023 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.2313 [0.2151, 0.2479], supported; 0.00) |
| VLWM-15 | #78 port (#71 carrier) | 0.6422 [0.6187, 0.6658] | 0.4523 [0.4009, 0.5048] | 27.8M | 0.008 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.1412 [0.1256, 0.1564], supported; 0.00) |
| TAWM-adaptive | #78 port (#71 carrier) | 0.4497 [0.4304, 0.4691] | 0.1543 [0.1297, 0.1801] | 236.0M | 0.078 | 0/933 | 0/18945 | off 0.00; off; off; off | dominated by C-F-15 (0.3337 [0.3040, 0.3638], supported; 0.00) |
| prior | no-model action prior | 0.7203 [0.6965, 0.7442] | 0.4180 [0.3633, 0.4727] | 0 | 0 | 0/933 | 42/18945 | **on** 1.00; **on**; **on**; **on** | **on the frontier** (1.00) |

## Frontier sets per inventory and axis

| Set | Inventory | Cells | AUC × MACs | AUC × wall | Top-3 × MACs | Top-3 × wall |
| --- | --- | --- | --- | --- | --- | --- |
| primary | grid | 311 × 3 | F-15-continuous, C-F-15, J-1e-01, prior | C-F-15, prior | F-15-continuous, prior | F-15-continuous, C-F-15, prior |
| primary | offset | 76 × 3 | F-5-micro, F-15-continuous, J-1e-03, J-3e-03, H-15-macro, prior | F-5-micro, F-15-continuous, C-F-15, J-1e-03, H-15-macro, prior | F-15-continuous, J-1e-03, prior | F-15-continuous, C-F-15, J-1e-03, prior |
| primary | angle | 78 × 3 | F-5-micro, J-1e-03, J-3e-03, prior | F-5-micro, J-1e-03, prior | J-1e-03, Jcont-1e-02, prior | J-1e-03, prior |
| primary | power | 47 × 3 | F-15-continuous, F-15-macro, J-1e-01, J-pg-1e-04, prior | F-5-macro, F-15-micro, C-F-15, J-1e-01, J-pg-1e-04, TAWM-15, VLWM-15, prior | F-15-continuous, C-F-15, CJ-3e-03, prior | C-F-15, CJ-3e-03, VLWM-15, prior |
| secondary | grid | 14 × 3 | F-15-continuous, C-F-15, J-1e-01, prior | C-F-15, prior | F-15-continuous, C-F-15, prior | C-F-15, prior |
| secondary | offset | 9 × 3 | F-15-continuous, Jcont-1e-02, H-15-macro, prior | F-15-continuous, C-F-15, Jcont-1e-02, H-15-macro, VLWM-15, prior | F-15-continuous, Jcont-1e-02, prior | F-15-continuous, C-F-15, Jcont-1e-02, prior |
| secondary | angle | 7 × 3 | F-15-continuous, J-1e-03, J-3e-02, prior | F-15-continuous, C-F-15, J-1e-03, J-3e-02, prior | F-15-continuous, J-1e-03, J-3e-02, prior | F-15-continuous, C-F-15, CJ-1e-05, J-1e-03, J-3e-02, prior |
| secondary | power | 8 × 3 | F-15-continuous, CJ-3e-03, J-1e-03, J-1e-01, J-pg-1e-04, prior | F-15-continuous, C-F-15, CJ-3e-03, J-1e-03, J-1e-01, J-pg-1e-04, J-pg-1e-01, prior | F-15-continuous, CJ-3e-03, M-dec, J_dec_diag, prior | C-F-15, CJ-3e-03, J-dec-1e-01, J_dec_diag, prior |

## Declared secondary set: #93 / #94 / #87 inventories, #93 grid (14 members × 3 seeds; DESCRIPTIVE)

Wall: this scoring's synchronized rollout seconds per state, all seeds. OL-pe / OL-dec are primary-only.

| Policy | Group | AUC | Top-3 | MACs / cand. | Wall s / state | All-tied cells | Tied pairs | Frontier (AUC×MACs, share; AUC×wall; top-3×MACs; top-3×wall) | Verdict (AUC × MACs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F-1-continuous | fixed request (hybrid) | 0.6765 [0.5639, 0.7765] | 0.3095 [0.1429, 0.5000] | 322.3M | 0.147 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.1124 [0.0410, 0.1897], supported; 0.00) |
| F-1-micro | fixed request (hybrid) | 0.6218 [0.5069, 0.7254] | 0.2619 [0.1190, 0.4286] | 404.0M | 0.186 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.1671 [0.0755, 0.2617], supported; 0.00) |
| F-1-macro | fixed request (hybrid) | 0.6845 [0.5854, 0.7788] | 0.3571 [0.1429, 0.6429] | 329.6M | 0.164 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.1044 [0.0517, 0.1736], supported; 0.00) |
| F-5-continuous | fixed request (hybrid) | 0.7582 [0.6632, 0.8432] | 0.5000 [0.2619, 0.7381] | 64.5M | 0.030 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0307 [-0.0109, 0.0797], readiness_or_precision_insufficient; 0.00) |
| F-5-micro | fixed request (hybrid) | 0.7536 [0.6614, 0.8350] | 0.4524 [0.2619, 0.6429] | 80.8M | 0.038 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0353 [-0.0051, 0.0841], readiness_or_precision_insufficient; 0.00) |
| F-5-macro | fixed request (hybrid) | 0.7625 [0.6854, 0.8394] | 0.4286 [0.1905, 0.6667] | 65.9M | 0.033 | 0/42 | 0/723 | off 0.05; off; off; off | dominated by C-F-15 (0.0264 [-0.0065, 0.0629], readiness_or_precision_insufficient; 0.05) |
| F-15-continuous | fixed request (hybrid) | 0.7643 [0.6755, 0.8481] | 0.5238 [0.3095, 0.7381] | 21.5M | 0.010 | 0/42 | 0/723 | **on** 0.74; off; **on**; off | **on the frontier** (0.74) |
| F-15-micro | fixed request (hybrid) | 0.7598 [0.6723, 0.8411] | 0.5000 [0.2857, 0.7143] | 26.9M | 0.013 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by J-1e-01 (0.0171 [0.0010, 0.0347], readiness_or_precision_insufficient; 0.00) |
| F-15-macro | fixed request (hybrid) | 0.7613 [0.6887, 0.8351] | 0.4524 [0.2381, 0.6667] | 22.0M | 0.011 | 0/42 | 0/723 | off 0.26; off; off; off | dominated by F-15-continuous (0.0030 [-0.0315, 0.0313], readiness_or_precision_insufficient; 0.26) |
| C-F-1 | continuous-only fixed Delta (C*) | 0.6191 [0.5313, 0.7016] | 0.3095 [0.2381, 0.3810] | 416.7M | 0.110 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.1698 [0.0951, 0.2448], supported; 0.00) |
| C-F-5 | continuous-only fixed Delta (C*) | 0.7011 [0.6390, 0.7608] | 0.5000 [0.3571, 0.6190] | 83.3M | 0.022 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0878 [0.0488, 0.1265], supported; 0.00) |
| C-F-15 | continuous-only fixed Delta (C*) | 0.7889 [0.7275, 0.8494] | 0.5714 [0.4286, 0.7143] | 27.8M | 0.008 | 0/42 | 0/723 | **on** 0.69; **on**; **on**; **on** | **on the frontier** (0.69) |
| CJ-1e-05 | horizon-only adaptive C-J (C*) | 0.7195 [0.6489, 0.7886] | 0.4762 [0.3333, 0.6429] | 48.6M | 0.020 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0694 [0.0257, 0.1132], supported; 0.00) |
| CJ-1e-04 | horizon-only adaptive C-J (C*) | 0.6932 [0.6113, 0.7679] | 0.4048 [0.2619, 0.5238] | 231.7M | 0.083 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0957 [0.0482, 0.1477], supported; 0.00) |
| CJ-3e-04 | horizon-only adaptive C-J (C*) | 0.7086 [0.6242, 0.7882] | 0.4048 [0.2619, 0.5238] | 241.9M | 0.084 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0803 [0.0321, 0.1355], supported; 0.00) |
| CJ-1e-03 | horizon-only adaptive C-J (C*) | 0.6896 [0.6099, 0.7636] | 0.4048 [0.2619, 0.5238] | 172.2M | 0.061 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0993 [0.0506, 0.1510], supported; 0.00) |
| CJ-3e-03 | horizon-only adaptive C-J (C*) | 0.6783 [0.6130, 0.7392] | 0.4048 [0.2619, 0.5238] | 81.6M | 0.040 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.1105 [0.0714, 0.1503], supported; 0.00) |
| CJ-1e-02 | horizon-only adaptive C-J (C*) | 0.7178 [0.6586, 0.7734] | 0.4524 [0.3095, 0.5952] | 50.0M | 0.024 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0711 [0.0382, 0.1058], supported; 0.00) |
| CJ-3e-02 | horizon-only adaptive C-J (C*) | 0.6903 [0.6361, 0.7444] | 0.4048 [0.2857, 0.5238] | 45.3M | 0.018 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0986 [0.0615, 0.1341], supported; 0.00) |
| CJ-1e-01 | horizon-only adaptive C-J (C*) | 0.7026 [0.6466, 0.7575] | 0.4762 [0.3571, 0.5714] | 44.4M | 0.018 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0863 [0.0439, 0.1315], supported; 0.00) |
| J-1e-05 | J (prediction-error DP) | 0.7154 [0.6251, 0.7982] | 0.3810 [0.2143, 0.5476] | 195.5M | 0.120 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0735 [0.0317, 0.1231], supported; 0.00) |
| J-1e-04 | J (prediction-error DP) | 0.7151 [0.6164, 0.8054] | 0.4048 [0.1905, 0.6190] | 182.1M | 0.137 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0738 [0.0197, 0.1358], supported; 0.00) |
| J-3e-04 | J (prediction-error DP) | 0.7189 [0.6237, 0.8046] | 0.4048 [0.1905, 0.6190] | 82.1M | 0.079 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0700 [0.0185, 0.1268], supported; 0.00) |
| J-1e-03 | J (prediction-error DP) | 0.7495 [0.6569, 0.8329] | 0.5000 [0.2619, 0.7381] | 34.4M | 0.023 | 0/42 | 0/723 | off 0.01; off; off; off | dominated by C-F-15 (0.0394 [-0.0020, 0.0863], readiness_or_precision_insufficient; 0.01) |
| J-3e-03 | J (prediction-error DP) | 0.7111 [0.6120, 0.7995] | 0.4286 [0.2381, 0.6190] | 33.2M | 0.023 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0778 [0.0272, 0.1338], supported; 0.00) |
| J-1e-02 | J (prediction-error DP) | 0.7224 [0.6274, 0.8036] | 0.4286 [0.2619, 0.6190] | 49.1M | 0.042 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0665 [0.0133, 0.1244], supported; 0.00) |
| J-3e-02 | J (prediction-error DP) | 0.7450 [0.6571, 0.8262] | 0.4762 [0.2857, 0.6667] | 25.3M | 0.016 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by J-1e-01 (0.0320 [0.0101, 0.0578], supported; 0.00) |
| J-1e-01 | J (prediction-error DP) | 0.7769 [0.6970, 0.8548] | 0.5238 [0.3095, 0.7381] | 22.1M | 0.013 | 0/42 | 0/723 | **on** 0.75; off; off; off | **on the frontier** (0.75) |
| Jcont-1e-05 | J-cont (continuous requests, hybrid) | 0.6900 [0.5876, 0.7809] | 0.2857 [0.1190, 0.4524] | 120.4M | 0.079 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0989 [0.0426, 0.1615], supported; 0.00) |
| Jcont-1e-04 | J-cont (continuous requests, hybrid) | 0.6836 [0.5857, 0.7743] | 0.2857 [0.1190, 0.4524] | 123.5M | 0.084 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.1053 [0.0403, 0.1743], supported; 0.00) |
| Jcont-3e-04 | J-cont (continuous requests, hybrid) | 0.7050 [0.6092, 0.7924] | 0.3333 [0.1667, 0.5238] | 77.7M | 0.077 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0838 [0.0304, 0.1411], supported; 0.00) |
| Jcont-1e-03 | J-cont (continuous requests, hybrid) | 0.6990 [0.5985, 0.7925] | 0.3810 [0.1667, 0.5952] | 89.6M | 0.081 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0899 [0.0335, 0.1517], supported; 0.00) |
| Jcont-3e-03 | J-cont (continuous requests, hybrid) | 0.6855 [0.5828, 0.7755] | 0.3333 [0.1429, 0.5238] | 69.9M | 0.066 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.1034 [0.0464, 0.1678], supported; 0.00) |
| Jcont-1e-02 | J-cont (continuous requests, hybrid) | 0.6948 [0.5896, 0.7904] | 0.4048 [0.2143, 0.5952] | 33.1M | 0.021 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0941 [0.0367, 0.1580], supported; 0.00) |
| Jcont-3e-02 | J-cont (continuous requests, hybrid) | 0.7704 [0.6796, 0.8535] | 0.5238 [0.3095, 0.7381] | 35.5M | 0.021 | 0/42 | 0/723 | off 0.13; off; off; off | dominated by C-F-15 (0.0185 [-0.0198, 0.0619], readiness_or_precision_insufficient; 0.13) |
| Jcont-1e-01 | J-cont (continuous requests, hybrid) | 0.7528 [0.6693, 0.8320] | 0.5238 [0.3095, 0.7381] | 22.2M | 0.013 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by J-1e-01 (0.0241 [0.0077, 0.0422], supported; 0.00) |
| T-continuous | J restricted (T-alpha / D-Delta) | 0.6912 [0.5785, 0.7875] | 0.4286 [0.2381, 0.6190] | 200.8M | 0.128 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0977 [0.0324, 0.1729], supported; 0.00) |
| T-micro | J restricted (T-alpha / D-Delta) | 0.6846 [0.5597, 0.7920] | 0.3571 [0.1667, 0.5714] | 167.6M | 0.105 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.1043 [0.0345, 0.1862], supported; 0.00) |
| T-macro | J restricted (T-alpha / D-Delta) | 0.6922 [0.5974, 0.7810] | 0.3571 [0.1429, 0.5952] | 288.7M | 0.200 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0967 [0.0481, 0.1591], supported; 0.00) |
| D-1 | J restricted (T-alpha / D-Delta) | 0.6956 [0.5950, 0.7880] | 0.3333 [0.1190, 0.5714] | 344.2M | 0.261 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0933 [0.0337, 0.1599], supported; 0.00) |
| D-5 | J restricted (T-alpha / D-Delta) | 0.7379 [0.6517, 0.8161] | 0.4524 [0.2619, 0.6190] | 73.7M | 0.056 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0510 [0.0057, 0.1007], supported; 0.00) |
| D-15 | J restricted (T-alpha / D-Delta) | 0.7676 [0.6888, 0.8428] | 0.4762 [0.2619, 0.6905] | 25.4M | 0.018 | 0/42 | 0/723 | off 0.04; off; off; off | dominated by J-1e-01 (0.0093 [-0.0028, 0.0222], readiness_or_precision_insufficient; 0.04) |
| J-dec-1e-04 | J_dec (decision-valued DP) | 0.6903 [0.6114, 0.7706] | 0.3810 [0.1905, 0.5714] | 45.7M | 0.046 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0986 [0.0480, 0.1550], supported; 0.00) |
| J-dec-1e-03 | J_dec (decision-valued DP) | 0.7004 [0.6119, 0.7841] | 0.3333 [0.1667, 0.5238] | 39.6M | 0.039 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0885 [0.0330, 0.1445], supported; 0.00) |
| J-dec-1e-02 | J_dec (decision-valued DP) | 0.6902 [0.5850, 0.7847] | 0.4524 [0.2619, 0.6429] | 46.7M | 0.048 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0986 [0.0475, 0.1650], supported; 0.00) |
| J-dec-1e-01 | J_dec (decision-valued DP) | 0.7502 [0.6773, 0.8231] | 0.4762 [0.2619, 0.6905] | 22.7M | 0.016 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by J-1e-01 (0.0267 [0.0140, 0.0394], supported; 0.00) |
| J-pg-1e-04 | J_pg (policy gradient) | 0.7743 [0.6889, 0.8556] | 0.5000 [0.2619, 0.7381] | 44.6M | 0.025 | 0/42 | 0/723 | off 0.07; off; off; off | dominated by C-F-15 (0.0146 [-0.0180, 0.0519], readiness_or_precision_insufficient; 0.07) |
| J-pg-1e-03 | J_pg (policy gradient) | 0.7527 [0.6605, 0.8374] | 0.5000 [0.2857, 0.7143] | 25.7M | 0.015 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by J-1e-01 (0.0242 [0.0031, 0.0472], supported; 0.00) |
| J-pg-1e-02 | J_pg (policy gradient) | 0.7639 [0.6730, 0.8497] | 0.5238 [0.3095, 0.7381] | 22.1M | 0.013 | 0/42 | 0/723 | off 0.08; off; off; off | dominated by J-1e-01 (0.0130 [-0.0058, 0.0323], readiness_or_precision_insufficient; 0.08) |
| J-pg-1e-01 | J_pg (policy gradient) | 0.7639 [0.6730, 0.8497] | 0.5238 [0.3095, 0.7381] | 22.1M | 0.013 | 0/42 | 0/723 | off 0.08; off; off; off | dominated by J-1e-01 (0.0130 [-0.0058, 0.0323], readiness_or_precision_insufficient; 0.08) |
| M-pe | factorized / open-loop (#101) | 0.7206 [0.6150, 0.8168] | 0.4524 [0.2381, 0.6905] | 173.9M | 0.148 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0683 [0.0067, 0.1378], supported; 0.00) |
| M-dec | factorized / open-loop (#101) | 0.7485 [0.6750, 0.8222] | 0.4524 [0.2381, 0.6667] | 22.7M | 0.018 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by J-1e-01 (0.0284 [0.0152, 0.0413], supported; 0.00) |
| J_diag | #105 tied-axes restriction | 0.7198 [0.6230, 0.8037] | 0.5000 [0.2619, 0.7381] | 119.3M | 0.083 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0690 [0.0189, 0.1256], supported; 0.00) |
| J_monotone | #105 tied-axes restriction | 0.7119 [0.6117, 0.8032] | 0.3810 [0.1667, 0.6190] | 189.1M | 0.155 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0770 [0.0218, 0.1401], supported; 0.00) |
| J_anti | #105 tied-axes restriction | 0.7280 [0.6270, 0.8213] | 0.4048 [0.1667, 0.6429] | 214.4M | 0.155 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0609 [0.0104, 0.1259], supported; 0.00) |
| J_dec_diag | #105 tied-axes restriction | 0.7168 [0.6439, 0.7880] | 0.4048 [0.2381, 0.5714] | 26.4M | 0.022 | 0/42 | 0/723 | off 0.03; off; off; off | dominated by J-1e-01 (0.0601 [-0.0032, 0.1196], readiness_or_precision_insufficient; 0.03) |
| J_dec_monotone | #105 tied-axes restriction | 0.7502 [0.6773, 0.8231] | 0.4762 [0.2619, 0.6905] | 22.7M | 0.018 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by J-1e-01 (0.0267 [0.0140, 0.0394], supported; 0.00) |
| J_dec_anti | #105 tied-axes restriction | 0.7643 [0.6755, 0.8481] | 0.5238 [0.3095, 0.7381] | 22.2M | 0.014 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by J-1e-01 (0.0126 [-0.0002, 0.0270], readiness_or_precision_insufficient; 0.00) |
| H-1-continuous | #105 hierarchical level | 0.5517 [0.5003, 0.6038] | 0.1905 [0.0952, 0.3095] | 410.8M | 0.226 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.2372 [0.1643, 0.3127], supported; 0.00) |
| H-5-micro | #105 hierarchical level | 0.6935 [0.6030, 0.7777] | 0.5238 [0.2857, 0.7381] | 58.5M | 0.035 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.0954 [0.0480, 0.1460], supported; 0.00) |
| H-15-macro | #105 hierarchical level | 0.4081 [0.3496, 0.4674] | 0.0714 [0.0000, 0.1429] | 7.9M | 0.006 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by prior (0.3171 [0.1723, 0.4636], supported; 0.00) |
| TAWM-1 | #78 port (#71 carrier) | 0.4052 [0.3362, 0.4711] | 0.0714 [0.0000, 0.1429] | 416.7M | 0.111 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.3837 [0.2717, 0.5014], supported; 0.00) |
| TAWM-5 | #78 port (#71 carrier) | 0.5868 [0.4707, 0.6936] | 0.2381 [0.0476, 0.4762] | 83.3M | 0.023 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.2021 [0.1156, 0.2918], supported; 0.00) |
| TAWM-15 | #78 port (#71 carrier) | 0.5857 [0.4607, 0.6977] | 0.2143 [0.0000, 0.4286] | 27.8M | 0.008 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.2032 [0.1103, 0.2995], supported; 0.00) |
| VLWM-1 | #78 port (#71 carrier) | 0.4887 [0.3815, 0.5973] | 0.2381 [0.0952, 0.3810] | 416.7M | 0.112 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.3002 [0.1588, 0.4345], supported; 0.00) |
| VLWM-5 | #78 port (#71 carrier) | 0.5321 [0.4776, 0.5854] | 0.2143 [0.0952, 0.3333] | 83.3M | 0.022 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.2568 [0.2067, 0.3087], supported; 0.00) |
| VLWM-15 | #78 port (#71 carrier) | 0.6226 [0.5074, 0.7267] | 0.3095 [0.1190, 0.5238] | 27.8M | 0.008 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.1663 [0.0917, 0.2454], supported; 0.00) |
| TAWM-adaptive | #78 port (#71 carrier) | 0.4204 [0.3527, 0.4844] | 0.0714 [0.0000, 0.1429] | 228.4M | 0.086 | 0/42 | 0/723 | off 0.00; off; off; off | dominated by C-F-15 (0.3685 [0.2522, 0.4857], supported; 0.00) |
| prior | no-model action prior | 0.7252 [0.6082, 0.8407] | 0.4286 [0.2143, 0.7143] | 0 | 0 | 0/42 | 3/723 | **on** 1.00; **on**; **on**; **on** | **on the frontier** (1.00) |

## Guards

- G1_reuse_fidelity: pass {"cells_checked": 35840, "mismatch_count": 0}
- G2_cross_source_identity: pass {"cells_checked": 43008, "mismatch_count": 0}
- G3_scoring_policy: pass
- G4_input_binding: pass {"inputs": 429}
- G5_chronology: pass {"first_new_record": "2026-10-14T15:01:02Z"}
- G6_secondary_inputs: pass

## Compute

- New: 7334 GPU-allocation seconds, 7334 process-wall seconds (phases: smoke 38s, anchors-secondary-f77 1s, secondary 509s, timing 2422s, timing-posthoc 4365s); engine 0 s.
- Reused retained scoring (not re-charged): #101 2283s, #114 13383s, #105 2966s, #Gate-B 49s.
- Hardware: 4 x RTX 5090 (one process per GPU) instead of the #85 RTX 3090.

## Post-freeze amendment: three-seed re-timing (post-hoc DESCRIPTIVE)

- at: 2026-10-14T15:18:02Z.
- after: the first publication of the frozen runner (all 70 seed-20260908 timing and 204 secondary records complete; compute.json, summary.json and findings.md written once).
- what: the timing pass was extended from the frozen seed 20260908 to all three seeds: 140 further timing records (seeds 20260909/10), new timing data collected after the freeze and after the first publication.
- why: judged after the first publication was seen: each seed is a separately trained policy, so seed 20260908 alone does not represent the all-seed MAC column of the same policy (grid transition calls per candidate at seed 20260908 vs pooled were 196 vs 121 for J-1e-04 and 21 vs 124.5 for CJ-1e-04), and the frozen seed-only J / C-J wall ratio of 16.7 describes seed 20260908.
- status: post-hoc DESCRIPTIVE (phase-A review B0, 2026-10-16): the frozen seed-20260908 timing pass is restored as the confirmatory wall axis (wall columns, AUC x wall and top-3 x wall frontiers, the ICLR wall ratio, G1 over the frozen seed), equal to the first publication kept in superseded/; the three-seed wall columns, wall ratio, wall frontiers and all-seed reuse check are reported only under posthoc_wall and are not a confirmatory result.
- unchanged: cells, retained and secondary records, AUC and top-3, MACs, every MAC-axis frontier, bootstrap share and verdict, and the ICLR-sentence token (its rule uses MACs).
- superseded: superseded/ keeps the first publication's findings.md, summary.json and compute.json.
- execution: one additional launch of --score on 4 GPUs (one process each) for the seed-20260909/10 timing tasks; existing records are skipped, nothing is overwritten; PIDs in logs/score-pids.txt.

Post-hoc DESCRIPTIVE wall axis (seeds 20260908, 20260909, 20260910; not confirmatory): J-1e-04 / CJ-1e-04 wall ratio 1.633 on the primary grid (frozen seed-20260908 ratio above: 16.738); all-seed reuse fidelity 107520 cells, 0 mismatches.

| Inventory | AUC × wall (three seeds) | Top-3 × wall (three seeds) |
| --- | --- | --- |
| grid | C-F-15, prior | F-15-continuous, C-F-15, VLWM-15, prior |
| offset | F-5-micro, F-15-continuous, C-F-15, J-1e-03, H-15-macro, prior | F-15-continuous, C-F-15, J-1e-03, prior |
| angle | F-5-micro, J-1e-03, prior | J-1e-03, prior |
| power | F-15-micro, C-F-15, J-1e-01, J-pg-1e-04, VLWM-15, prior | C-F-5, C-F-15, CJ-3e-03, VLWM-15, prior |

| Policy (grid) | Wall s / state, frozen seed (confirmatory) | Wall s / state, three seeds (post-hoc) |
| --- | --- | --- |
| F-1-continuous | 0.147 | 0.147 |
| F-1-micro | 0.187 | 0.190 |
| F-1-macro | 0.165 | 0.164 |
| F-5-continuous | 0.030 | 0.030 |
| F-5-micro | 0.037 | 0.037 |
| F-5-macro | 0.033 | 0.033 |
| F-15-continuous | 0.010 | 0.010 |
| F-15-micro | 0.013 | 0.013 |
| F-15-macro | 0.011 | 0.011 |
| C-F-1 | 0.113 | 0.112 |
| C-F-5 | 0.023 | 0.022 |
| C-F-15 | 0.008 | 0.008 |
| CJ-1e-05 | 0.015 | 0.020 |
| CJ-1e-04 | 0.014 | 0.085 |
| CJ-3e-04 | 0.014 | 0.086 |
| CJ-1e-03 | 0.014 | 0.062 |
| CJ-3e-03 | 0.014 | 0.039 |
| CJ-1e-02 | 0.011 | 0.024 |
| CJ-3e-02 | 0.011 | 0.018 |
| CJ-1e-01 | 0.011 | 0.018 |
| J-1e-05 | 0.183 | 0.113 |
| J-1e-04 | 0.242 | 0.138 |
| J-3e-04 | 0.028 | 0.080 |
| J-1e-03 | 0.021 | 0.022 |
| J-3e-03 | 0.021 | 0.022 |
| J-1e-02 | 0.035 | 0.043 |
| J-3e-02 | 0.017 | 0.016 |
| J-1e-01 | 0.013 | 0.013 |
| Jcont-1e-05 | 0.033 | 0.076 |
| Jcont-1e-04 | 0.048 | 0.084 |
| Jcont-3e-04 | 0.053 | 0.080 |
| Jcont-1e-03 | 0.038 | 0.083 |
| Jcont-3e-03 | 0.021 | 0.068 |
| Jcont-1e-02 | 0.029 | 0.022 |
| Jcont-3e-02 | 0.037 | 0.021 |
| Jcont-1e-01 | 0.013 | 0.013 |
| T-continuous | 0.139 | 0.125 |
| T-micro | 0.251 | 0.105 |
| T-macro | 0.150 | 0.193 |
| D-1 | 0.303 | 0.260 |
| D-5 | 0.056 | 0.055 |
| D-15 | 0.017 | 0.017 |
| J-dec-1e-04 | 0.041 | 0.047 |
| J-dec-1e-03 | 0.041 | 0.039 |
| J-dec-1e-02 | 0.072 | 0.048 |
| J-dec-1e-01 | 0.013 | 0.016 |
| J-pg-1e-04 | 0.037 | 0.025 |
| J-pg-1e-03 | 0.015 | 0.015 |
| J-pg-1e-02 | 0.013 | 0.013 |
| J-pg-1e-01 | 0.014 | 0.013 |
| M-pe | 0.254 | 0.148 |
| M-dec | 0.015 | 0.018 |
| OL-pe | 0.147 | 0.093 |
| OL-dec | 0.011 | 0.011 |
| J_diag | 0.123 | 0.079 |
| J_monotone | 0.269 | 0.155 |
| J_anti | 0.206 | 0.157 |
| J_dec_diag | 0.033 | 0.021 |
| J_dec_monotone | 0.015 | 0.018 |
| J_dec_anti | 0.014 | 0.015 |
| H-1-continuous | 0.224 | 0.224 |
| H-5-micro | 0.036 | 0.035 |
| H-15-macro | 0.006 | 0.006 |
| TAWM-1 | 0.111 | 0.112 |
| TAWM-5 | 0.022 | 0.022 |
| TAWM-15 | 0.008 | 0.008 |
| VLWM-1 | 0.113 | 0.111 |
| VLWM-5 | 0.023 | 0.023 |
| VLWM-15 | 0.008 | 0.008 |
| TAWM-adaptive | 0.078 | 0.088 |
| prior | 0 | 0 |

## Excluded

- J-as-cont (#114): a counterfactual replay of J's recorded schedules; it needs J to run first, so it is not a deployable policy.
- wrong-anchor arms (#101/#114/#105): controls scored from the wrong anchor, not policies.
- OL-pe / OL-dec on the secondary set: their open-loop streams are frozen to the primary cell order and J's primary request frequencies (#101); no secondary analogue was frozen.
