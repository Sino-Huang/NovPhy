# Issue #116: request-set scaling (findings)

Plan frozen 2026-10-17T03:02:25Z; scored once on the #104 sealed #109 evaluation split; tie-free cost; level-clustered percentile bootstraps (10 000 draws, PCG64 7201), DESCRIPTIVE; delta = 0.02; zero engine seconds. Grid is primary; offset, angle, power, rollout error, label accuracy and MACs are DESCRIPTIVE.

Guards pass: True.

## Dispositions (pre-declared)

1. **A larger or continuous Delta set improves ranking over R_9: `readiness_or_precision_insufficient`.** Rule: per set and budget the grid J-flat AUC minus R_9's J-flat (paired by level and seed) with the delta superiority token; supported iff some set is supported under both budgets; not_supported_by_this_experiment iff every set and budget is.

| set / budget | J-flat grid AUC | minus R_9 | token |
| --- | --- | --- | --- |
| R_9 (Gate-B joint, both budgets) | 0.678 [0.656, 0.700] | - | - |
| R_18/total | 0.615 [0.603, 0.627] | -0.063 [-0.077, -0.049] | not_supported_by_this_experiment |
| R_18/request | 0.638 [0.618, 0.659] | -0.042 [-0.056, -0.030] | not_supported_by_this_experiment |
| R_30/total | 0.543 [0.527, 0.559] | -0.135 [-0.154, -0.116] | not_supported_by_this_experiment |
| R_30/request | 0.749 [0.737, 0.761] | +0.071 [+0.049, +0.092] | supported |
| R_cont/total | 0.699 [0.687, 0.711] | +0.021 [-0.002, +0.042] | readiness_or_precision_insufficient |
| R_cont/request | 0.947 [0.935, 0.958] | +0.269 [+0.245, +0.293] | supported |

2. **The DP labeller stays tractable at |R| = 225: `supported`.** Rule: every usable R_cont predictor's exact round-0 pass over the policy pool (N = 3480 windows, L = 225 frames) within 3600 GPU s on one RTX 5090 and at most 2x R_9's seconds per carrier evaluation.

| R_cont predictor | GPU s | peak GB | s per evaluation / R_9 | ok |
| --- | --- | --- | --- | --- |
| R_cont/total seed 20260908 | 18.1 | 1.23 | 0.73 | True |
| R_cont/total seed 20260909 | 17.9 | 1.26 | 0.73 | True |
| R_cont/total seed 20260910 | 18.3 | 1.26 | 0.74 | True |
| R_cont/request seed 20260908 | 17.8 | 1.23 | 0.72 | True |
| R_cont/request seed 20260909 | 18.0 | 1.23 | 0.73 | True |
| R_cont/request seed 20260910 | 17.9 | 1.23 | 0.73 | True |

3. **Sub-frame Delta: `unsupported`.** one carrier frame is 50 native engine steps (the #104 / #109 capture stride, frames 30000 + 50 i); a Delta below one frame has no observed target in any existing capture, and a finer-stride capture is outside the frozen #109 / #104 plan (an owner decision, not this ticket).

## Scaling table (grid; per set and budget)

| set / budget | updates | DP round-0 GPU s | peak GB | label entropy (bits / max) | J-flat label accuracy (eval) | J-flat rollout error t=225 | J-flat grid AUC | J-flat MACs / candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R_9 | 9000 | 1.2 | 1.16 | 2.12 / 3.17 | 0.528 [0.523, 0.534] | 0.0272 [0.0262, 0.0283] | 0.678 [0.656, 0.700] | 374.9M |
| R_18/total | 9000 | 2.0 | 1.16 | 3.14 / 4.17 | 0.399 [0.390, 0.407] | 0.0389 [0.0378, 0.0400] | 0.615 [0.603, 0.627] | 290.7M |
| R_18/request | 18000 | 2.0 | 1.16 | 2.65 / 4.17 | 0.484 [0.476, 0.492] | 0.0881 [0.0852, 0.0911] | 0.638 [0.618, 0.659] | 336.7M |
| R_30/total | 9000 | 3.0 | 1.17 | 3.38 / 4.91 | 0.311 [0.308, 0.315] | 0.2746 [0.2694, 0.2798] | 0.543 [0.527, 0.559] | 295.2M |
| R_30/request | 30000 | 2.9 | 1.17 | 3.81 / 4.91 | 0.220 [0.218, 0.222] | 0.1129 [0.1047, 0.1213] | 0.749 [0.737, 0.761] | 286.4M |
| R_cont/total | 9000 | 18.1 | 1.25 | 6.91 / 7.81 | 0.136 [0.134, 0.138] | 0.2158 [0.2029, 0.2292] | 0.699 [0.687, 0.711] | 105.5M |
| R_cont/request | 225000 | 17.9 | 1.23 | 6.11 / 7.81 | 0.149 [0.146, 0.151] | 0.0842 [0.0785, 0.0899] | 0.947 [0.935, 0.958] | 220.5M |

## Ranking AUC per inventory (J heads, fixed-request means, controls)

| ranker | grid | offset | angle | power |
| --- | --- | --- | --- | --- |
| R_9 J-flat | 0.678 [0.656, 0.700] | 0.689 [0.637, 0.740] | 0.752 [0.709, 0.794] | 0.780 [0.736, 0.823] |
| R_9 J-factorised | 0.676 [0.654, 0.698] | 0.703 [0.643, 0.762] | 0.742 [0.696, 0.786] | 0.802 [0.762, 0.842] |
| R_9 fixed, R_9-request mean | 0.722 [0.703, 0.741] | 0.716 [0.700, 0.732] | 0.737 [0.723, 0.752] | 0.802 [0.767, 0.839] |
| R_9 fixed, all-request mean | 0.722 [0.703, 0.741] | 0.716 [0.700, 0.732] | 0.737 [0.723, 0.752] | 0.802 [0.767, 0.839] |
| R_18/total J-flat | 0.615 [0.603, 0.627] | 0.580 [0.568, 0.593] | 0.570 [0.560, 0.580] | 0.720 [0.680, 0.763] |
| R_18/total J-factorised | 0.601 [0.588, 0.615] | 0.565 [0.554, 0.577] | 0.554 [0.543, 0.565] | 0.742 [0.707, 0.779] |
| R_18/total fixed, R_9-request mean | 0.686 [0.673, 0.700] | 0.671 [0.652, 0.690] | 0.663 [0.647, 0.678] | 0.754 [0.716, 0.793] |
| R_18/total fixed, all-request mean | 0.671 [0.658, 0.684] | 0.661 [0.641, 0.682] | 0.650 [0.633, 0.667] | 0.751 [0.713, 0.789] |
| R_18/request J-flat | 0.638 [0.618, 0.659] | 0.722 [0.701, 0.743] | 0.725 [0.709, 0.740] | 0.764 [0.717, 0.811] |
| R_18/request J-factorised | 0.623 [0.602, 0.645] | 0.731 [0.709, 0.752] | 0.719 [0.697, 0.739] | 0.780 [0.737, 0.823] |
| R_18/request fixed, R_9-request mean | 0.709 [0.694, 0.724] | 0.758 [0.743, 0.774] | 0.748 [0.736, 0.761] | 0.768 [0.734, 0.803] |
| R_18/request fixed, all-request mean | 0.709 [0.694, 0.724] | 0.755 [0.739, 0.772] | 0.746 [0.732, 0.760] | 0.768 [0.735, 0.802] |
| R_30/total J-flat | 0.543 [0.527, 0.559] | 0.589 [0.567, 0.611] | 0.542 [0.520, 0.564] | 0.680 [0.643, 0.719] |
| R_30/total J-factorised | 0.564 [0.546, 0.582] | 0.593 [0.561, 0.626] | 0.546 [0.518, 0.575] | 0.670 [0.633, 0.709] |
| R_30/total fixed, R_9-request mean | 0.593 [0.580, 0.606] | 0.572 [0.563, 0.581] | 0.557 [0.551, 0.564] | 0.731 [0.696, 0.768] |
| R_30/total fixed, all-request mean | 0.628 [0.615, 0.641] | 0.591 [0.577, 0.606] | 0.586 [0.575, 0.597] | 0.741 [0.709, 0.775] |
| R_30/request J-flat | 0.749 [0.737, 0.761] | 0.869 [0.850, 0.886] | 0.876 [0.859, 0.892] | 0.829 [0.787, 0.866] |
| R_30/request J-factorised | 0.739 [0.727, 0.751] | 0.859 [0.839, 0.879] | 0.854 [0.835, 0.873] | 0.813 [0.775, 0.848] |
| R_30/request fixed, R_9-request mean | 0.821 [0.812, 0.831] | 0.879 [0.864, 0.892] | 0.883 [0.870, 0.896] | 0.785 [0.744, 0.822] |
| R_30/request fixed, all-request mean | 0.849 [0.839, 0.859] | 0.891 [0.877, 0.904] | 0.899 [0.885, 0.912] | 0.794 [0.754, 0.830] |
| R_cont/total J-flat | 0.699 [0.687, 0.711] | 0.733 [0.704, 0.763] | 0.716 [0.697, 0.736] | 0.720 [0.675, 0.764] |
| R_cont/total J-factorised | 0.678 [0.662, 0.693] | 0.674 [0.645, 0.703] | 0.654 [0.629, 0.678] | 0.707 [0.677, 0.740] |
| R_cont/total J-ordinal | 0.694 [0.682, 0.706] | 0.699 [0.677, 0.721] | 0.693 [0.670, 0.715] | 0.722 [0.676, 0.767] |
| R_cont/total fixed, R_9-request mean | 0.791 [0.781, 0.801] | 0.799 [0.784, 0.814] | 0.786 [0.777, 0.796] | 0.767 [0.737, 0.798] |
| R_cont/total fixed, all-request mean | 0.837 [0.826, 0.848] | 0.831 [0.816, 0.846] | 0.829 [0.819, 0.840] | 0.752 [0.721, 0.783] |
| R_cont/request J-flat | 0.947 [0.935, 0.958] | 0.904 [0.873, 0.929] | 0.954 [0.940, 0.966] | 0.777 [0.730, 0.823] |
| R_cont/request J-factorised | 0.926 [0.913, 0.937] | 0.836 [0.804, 0.865] | 0.882 [0.859, 0.904] | 0.759 [0.706, 0.811] |
| R_cont/request J-ordinal | 0.925 [0.911, 0.938] | 0.864 [0.826, 0.899] | 0.917 [0.892, 0.941] | 0.780 [0.732, 0.828] |
| R_cont/request fixed, R_9-request mean | 0.959 [0.949, 0.967] | 0.942 [0.926, 0.956] | 0.966 [0.957, 0.973] | 0.810 [0.767, 0.852] |
| R_cont/request fixed, all-request mean | 0.967 [0.958, 0.975] | 0.958 [0.942, 0.971] | 0.978 [0.971, 0.984] | 0.832 [0.793, 0.870] |
| no-model action prior | 0.720 [0.696, 0.744] | 0.500 [0.500, 0.500] | 0.812 [0.778, 0.845] | 0.500 [0.500, 0.500] |

Fixed requests follow the endpoint rule (floor(225 / Delta) transitions, then the same-alpha fallback to exactly 225). Best fixed request in hindsight (grid, member mean; selection on the evaluation split, DESCRIPTIVE):

- R_9: F-15-continuous 0.762 [0.743, 0.781]
- R_18/total: F-15-macro 0.791 [0.775, 0.809]
- R_18/request: F-15-macro 0.830 [0.815, 0.844]
- R_30/total: F-45-micro 0.744 [0.727, 0.761]
- R_30/request: F-75-micro 0.879 [0.868, 0.889]
- R_cont/total: F-6-macro 0.875 [0.862, 0.888]
- R_cont/request: F-44-micro 0.970 [0.962, 0.977]

### Contrasts against R_9 (grid; paired by level and seed)

| set / budget | J-flat - R_9 J-flat | fixed R_9-request mean - R_9 | rollout error J-flat - R_9 | rollout error fixed R_9 requests - R_9 |
| --- | --- | --- | --- | --- |
| R_18/total | -0.063 [-0.077, -0.049] | -0.035 [-0.044, -0.027] | +0.0117 [+0.0110, +0.0124] | +0.0045 [+0.0040, +0.0049] |
| R_18/request | -0.042 [-0.056, -0.030] | -0.027 [-0.040, -0.015] | +0.0609 [+0.0587, +0.0632] | +0.0285 [+0.0273, +0.0297] |
| R_30/total | -0.135 [-0.154, -0.116] | -0.128 [-0.138, -0.119] | +0.2474 [+0.2425, +0.2523] | +0.1670 [+0.1642, +0.1697] |
| R_30/request | +0.071 [+0.049, +0.092] | +0.100 [+0.081, +0.119] | +0.0857 [+0.0777, +0.0939] | +0.0577 [+0.0536, +0.0617] |
| R_cont/total | +0.021 [-0.002, +0.042] | +0.069 [+0.056, +0.082] | +0.1886 [+0.1757, +0.2021] | +0.4875 [+0.4817, +0.4933] |
| R_cont/request | +0.269 [+0.245, +0.293] | +0.237 [+0.217, +0.256] | +0.0570 [+0.0516, +0.0623] | +0.0377 [+0.0347, +0.0407] |

## Policy-head comparison (per set)

| set / budget | head | width | parameters | policy MACs / call | label accuracy exact / Delta / alpha (eval) | grid AUC | minus flat | rollout error t=225 | MACs / candidate | Delta chosen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R_9 | flat | 128 | 32265 | 32128 | 0.528 / 0.546 / 0.859 | 0.678 [0.656, 0.700] | - | 0.0272 [0.0262, 0.0283] | 374.9M | median 1 (10-90 %: 1-1), modal F-1-micro |
| R_9 | factorised | 130 | 32376 | 32240 | 0.505 / 0.558 / 0.858 | 0.676 [0.654, 0.698] | -0.002 [-0.004, +0.000] | 0.0272 [0.0262, 0.0283] | 379.9M | median 1 (10-90 %: 1-1), modal F-1-micro |
| R_18/total | flat | 128 | 33426 | 33280 | 0.399 / 0.465 / 0.779 | 0.615 [0.603, 0.627] | - | 0.0389 [0.0378, 0.0400] | 290.7M | median 1 (10-90 %: 1-1), modal F-1-macro |
| R_18/total | factorised | 133 | 33525 | 33383 | 0.393 / 0.479 / 0.790 | 0.601 [0.588, 0.615] | -0.014 [-0.019, -0.008] | 0.0367 [0.0358, 0.0376] | 347.9M | median 1 (10-90 %: 1-1), modal F-1-macro |
| R_18/request | flat | 128 | 33426 | 33280 | 0.484 / 0.508 / 0.905 | 0.638 [0.618, 0.659] | - | 0.0881 [0.0852, 0.0911] | 336.7M | median 1 (10-90 %: 1-1), modal F-1-macro |
| R_18/request | factorised | 133 | 33525 | 33383 | 0.466 / 0.494 / 0.903 | 0.623 [0.602, 0.645] | -0.015 [-0.019, -0.011] | 0.0885 [0.0855, 0.0915] | 347.8M | median 1 (10-90 %: 1-1), modal F-1-macro |
| R_30/total | flat | 128 | 34974 | 34816 | 0.311 / 0.334 / 0.872 | 0.543 [0.527, 0.559] | - | 0.2746 [0.2694, 0.2798] | 295.2M | median 1 (10-90 %: 1-1), modal F-1-micro |
| R_30/total | factorised | 137 | 35085 | 34935 | 0.304 / 0.341 / 0.869 | 0.564 [0.546, 0.582] | +0.021 [+0.008, +0.035] | 0.2676 [0.2613, 0.2740] | 249.3M | median 1 (10-90 %: 1-1), modal F-1-micro |
| R_30/request | flat | 128 | 34974 | 34816 | 0.220 / 0.242 / 0.734 | 0.749 [0.737, 0.761] | - | 0.1129 [0.1047, 0.1213] | 286.4M | median 1 (10-90 %: 1-1), modal F-1-continuous |
| R_30/request | factorised | 137 | 35085 | 34935 | 0.195 / 0.247 / 0.748 | 0.739 [0.727, 0.751] | -0.010 [-0.016, -0.004] | 0.1568 [0.1491, 0.1643] | 333.5M | median 1 (10-90 %: 1-1), modal F-1-micro |
| R_cont/total | flat | 128 | 60129 | 59776 | 0.136 / 0.169 / 0.719 | 0.699 [0.687, 0.711] | - | 0.2158 [0.2029, 0.2292] | 105.5M | median 1 (10-90 %: 1-1), modal F-1-continuous |
| R_cont/total | factorised | 187 | 60105 | 59840 | 0.145 / 0.192 / 0.780 | 0.678 [0.662, 0.693] | -0.021 [-0.037, -0.004] | 0.2316 [0.2205, 0.2432] | 135.0M | median 1 (10-90 %: 1-1), modal F-1-continuous |
| R_cont/total | ordinal | 243 | 60099 | 59778 | 0.072 / 0.101 / 0.793 | 0.694 [0.682, 0.706] | -0.004 [-0.018, +0.009] | 0.1487 [0.1397, 0.1582] | 74.8M | median 1 (10-90 %: 1-1), modal F-1-continuous |
| R_cont/request | flat | 128 | 60129 | 59776 | 0.149 / 0.166 / 0.894 | 0.947 [0.935, 0.958] | - | 0.0842 [0.0785, 0.0899] | 220.5M | median 1 (10-90 %: 1-1), modal F-1-micro |
| R_cont/request | factorised | 187 | 60105 | 59840 | 0.143 / 0.172 / 0.892 | 0.926 [0.913, 0.937] | -0.022 [-0.027, -0.017] | 0.0788 [0.0735, 0.0843] | 199.7M | median 1 (10-90 %: 1-1), modal F-1-micro |
| R_cont/request | ordinal | 243 | 60099 | 59778 | 0.077 / 0.087 / 0.890 | 0.925 [0.911, 0.938] | -0.022 [-0.030, -0.015] | 0.0685 [0.0631, 0.0740] | 122.6M | median 1 (10-90 %: 1-1), modal F-1-micro |

Predictor MACs per transition: continuous 1.4M, micro 1.8M, macro 1.5M (Delta-independent).

## Controls

| set / budget | J-flat - prior (grid) | token | right - wrong anchor | token | stability |
| --- | --- | --- | --- | --- | --- |
| R_9 | -0.042 [-0.069, -0.016] | not_supported_by_this_experiment | +0.003 [+0.001, +0.004] | not_supported_by_this_experiment | 0.99 |
| R_18/total | -0.105 [-0.131, -0.079] | not_supported_by_this_experiment | +0.030 [+0.023, +0.036] | supported | 0.90 |
| R_18/request | -0.082 [-0.115, -0.049] | not_supported_by_this_experiment | -0.001 [-0.004, +0.001] | not_supported_by_this_experiment | 0.99 |
| R_30/total | -0.177 [-0.209, -0.145] | not_supported_by_this_experiment | -0.006 [-0.014, +0.001] | not_supported_by_this_experiment | 0.81 |
| R_30/request | +0.029 [+0.001, +0.056] | supported | +0.162 [+0.150, +0.174] | supported | 0.66 |
| R_cont/total | -0.022 [-0.049, +0.006] | not_supported_by_this_experiment | +0.142 [+0.128, +0.156] | supported | 0.37 |
| R_cont/request | +0.227 [+0.202, +0.251] | supported | +0.384 [+0.371, +0.397] | supported | 0.24 |

## DP labelling cost

| set / budget | requests | round-0 s (L = 225) | L = 60 s | s per 10^6 evaluations (L = 225) | ratio to R_9 | Delta entropy (bits) |
| --- | --- | --- | --- | --- | --- | --- |
| R_9 | 9 | 1.2 | 0.3 | 0.168 | 1.00 | 1.40 |
| R_18/total | 18 | 2.0 | 0.5 | 0.144 | 0.86 | 2.30 |
| R_18/request | 18 | 2.0 | 0.5 | 0.146 | 0.87 | 2.18 |
| R_30/total | 30 | 3.0 | 0.6 | 0.137 | 0.82 | 2.77 |
| R_30/request | 30 | 2.9 | 0.6 | 0.136 | 0.81 | 3.10 |
| R_cont/total | 225 | 18.1 | 2.6 | 0.123 | 0.73 | 5.78 |
| R_cont/request | 225 | 17.9 | 2.5 | 0.122 | 0.73 | 5.78 |

Linear fit over every exact pass: 0.121 s per 10^6 carrier evaluations + 0.2 s (R^2 0.999, 40 passes); log-log slope of the round-0 seconds against |R|: 0.87.

Exact vs pruned DP (R_cont; candidate Delta restricted to the log-spaced grid [1, 2, 3, 5, 7, 11, 16, 23, 34, 51, 75] x alpha):

| predictor | label disagreement | Delta disagreement | alpha disagreement | exact label on the grid | mean relative value gap | pruned GPU s |
| --- | --- | --- | --- | --- | --- | --- |
| R_cont/total seed 20260908 | 0.813 | 0.813 | 0.076 | 0.268 | +0.0293 | 3.1 |
| R_cont/total seed 20260909 | 0.808 | 0.808 | 0.076 | 0.293 | +0.0309 | 3.3 |
| R_cont/total seed 20260910 | 0.815 | 0.815 | 0.090 | 0.265 | +0.0274 | 3.2 |
| R_cont/request seed 20260908 | 0.788 | 0.788 | 0.018 | 0.292 | +0.0540 | 3.2 |
| R_cont/request seed 20260909 | 0.850 | 0.850 | 0.254 | 0.198 | +0.0725 | 3.3 |
| R_cont/request seed 20260910 | 0.818 | 0.818 | 0.050 | 0.246 | +0.0527 | 3.1 |

## Training

| set / budget | seed | updates | GPU s | skipped (cap) | usable | typed failure |
| --- | --- | --- | --- | --- | --- | --- |
| R_9 | 20260908 | 9000 | Gate-B handoff | - | True | - |
| R_9 | 20260909 | 9000 | Gate-B handoff | - | True | - |
| R_9 | 20260910 | 9000 | Gate-B handoff | - | True | - |
| R_18/total | 20260908 | 9000 | 1082 | 0 (45) | True | - |
| R_18/total | 20260909 | 9000 | 1068 | 0 (45) | True | - |
| R_18/total | 20260910 | 9000 | 1127 | 0 (45) | True | - |
| R_18/request | 20260908 | 18000 | 2418 | 0 (90) | False | final_checkpoint_excursion |
| R_18/request | 20260909 | 18000 | 2642 | 0 (90) | True | - |
| R_18/request | 20260910 | 18000 | 2679 | 0 (90) | True | - |
| R_30/total | 20260908 | 9000 | 1601 | 0 (45) | True | - |
| R_30/total | 20260909 | 9000 | 1570 | 0 (45) | True | - |
| R_30/total | 20260910 | 9000 | 1601 | 0 (45) | True | - |
| R_30/request | 20260908 | 30000 | 3275 | 0 (150) | True | - |
| R_30/request | 20260909 | 30000 | 2806 | 0 (150) | True | - |
| R_30/request | 20260910 | 30000 | 2801 | 0 (150) | True | - |
| R_cont/total | 20260908 | 9000 | 1064 | 0 (45) | True | - |
| R_cont/total | 20260909 | 9000 | 6240 | 0 (45) | True | - |
| R_cont/total | 20260910 | 9000 | 1094 | 0 (45) | True | - |
| R_cont/request | 20260908 | 225000 | 33824 | 0 (1125) | True | - |
| R_cont/request | 20260909 | 225000 | 33930 | 0 (1125) | True | - |
| R_cont/request | 20260910 | 225000 | 33630 | 0 (1125) | True | - |

## Guards

- G1_isolation: True
- G2_scoring_policy: True
- G3_chronology: True
- G4_binding: True
- G5_replication: True
- G6_training_integrity: True
- G7_nonfinite: True

Compute: 156486 GPU s (label 1619, score 20414, train 134453); 4 x NVIDIA GeForce RTX 5090 (not the #85 RTX 3090); engine seconds 0.
