# Issue #102 - closed-loop evaluation (retry-from-start, live #104 player)

Frozen 2026-10-15T00:19:38Z. Pass = engine pig removal (`pig_removed`), shot budget 3. Every shot resets the single-bird level; the planner re-ranks the 16-launch grid from the level's decision frame and excludes candidates already executed and failed. The state never changes between shots, so replanning amounts to excluding failed candidates. Every chosen launch was executed live; one execution per distinct level x candidate is shared by the episodes choosing it (pre-declared). Intervals: level-clustered bootstrap over levels x seeds, DESCRIPTIVE; contrast tokens use the delta rule (delta = 0.02) on pass@3 differences.

Guards: FAILED - every token is readiness_or_precision_insufficient.

## N1

Levels 400, cells (level x seed) 1200; recorded-inventory ceiling 0.777.

| Arm | Group | pass@1 | pass@2 | pass@3 | shots to pass (passed) | decision s | MACs / decision | pass@3 - prior | pass@3 - random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F-1-continuous | fixed request (hybrid LC) | 0.070 [0.045, 0.095] | 0.203 [0.165, 0.242] | **0.334 [0.295, 0.375]** | 2.40 [2.30, 2.51] | 0.1483 | 5156.8M | 0.012 [-0.028, 0.052] readiness_or_precision_insufficient | 0.137 [0.098, 0.177] readiness_or_precision_insufficient |
| F-1-micro | fixed request (hybrid LC) | 0.070 [0.045, 0.095] | 0.197 [0.166, 0.228] | **0.326 [0.287, 0.366]** | 2.38 [2.28, 2.47] | 0.1894 | 6463.6M | 0.003 [-0.046, 0.052] readiness_or_precision_insufficient | 0.128 [0.092, 0.168] readiness_or_precision_insufficient |
| F-1-macro | fixed request (hybrid LC) | 0.112 [0.089, 0.135] | 0.226 [0.191, 0.262] | **0.343 [0.297, 0.390]** | 2.01 [1.92, 2.12] | 0.1647 | 5272.9M | 0.020 [-0.015, 0.055] readiness_or_precision_insufficient | 0.145 [0.098, 0.193] readiness_or_precision_insufficient |
| F-5-continuous | fixed request (hybrid LC) | 0.180 [0.151, 0.210] | 0.333 [0.293, 0.374] | **0.453 [0.408, 0.499]** | 2.03 [1.94, 2.11] | 0.0309 | 1031.4M | 0.131 [0.087, 0.176] readiness_or_precision_insufficient | 0.256 [0.212, 0.302] readiness_or_precision_insufficient |
| F-5-micro | fixed request (hybrid LC) | 0.167 [0.138, 0.197] | 0.343 [0.302, 0.384] | **0.438 [0.394, 0.481]** | 1.99 [1.90, 2.08] | 0.0387 | 1292.7M | 0.115 [0.069, 0.162] readiness_or_precision_insufficient | 0.240 [0.200, 0.282] readiness_or_precision_insufficient |
| F-5-macro | fixed request (hybrid LC) | 0.107 [0.083, 0.132] | 0.295 [0.258, 0.333] | **0.413 [0.371, 0.457]** | 2.16 [2.09, 2.24] | 0.0342 | 1054.6M | 0.091 [0.051, 0.132] readiness_or_precision_insufficient | 0.216 [0.173, 0.261] readiness_or_precision_insufficient |
| F-15-continuous | fixed request (hybrid LC) | 0.189 [0.161, 0.218] | 0.333 [0.295, 0.371] | **0.478 [0.434, 0.523]** | 2.04 [1.96, 2.11] | 0.0110 | 343.8M | 0.156 [0.110, 0.203] readiness_or_precision_insufficient | 0.281 [0.238, 0.324] readiness_or_precision_insufficient |
| F-15-micro | fixed request (hybrid LC) | 0.221 [0.192, 0.251] | 0.350 [0.313, 0.388] | **0.458 [0.416, 0.502]** | 1.83 [1.76, 1.90] | 0.0138 | 430.9M | 0.136 [0.089, 0.184] readiness_or_precision_insufficient | 0.261 [0.219, 0.303] readiness_or_precision_insufficient |
| F-15-macro | fixed request (hybrid LC) | 0.151 [0.131, 0.172] | 0.301 [0.268, 0.335] | **0.424 [0.384, 0.465]** | 1.98 [1.91, 2.05] | 0.0121 | 351.5M | 0.102 [0.057, 0.147] readiness_or_precision_insufficient | 0.227 [0.187, 0.268] readiness_or_precision_insufficient |
| C-F-15 | C* fixed (15, continuous) | 0.148 [0.126, 0.172] | 0.371 [0.332, 0.412] | **0.466 [0.427, 0.505]** | 1.96 [1.87, 2.04] | 0.0084 | 444.5M | 0.143 [0.099, 0.188] readiness_or_precision_insufficient | 0.268 [0.230, 0.309] readiness_or_precision_insufficient |
| CJ-1e-04 | horizon-only adaptive C-J (lambda 1e-4) | 0.161 [0.136, 0.187] | 0.296 [0.263, 0.329] | **0.346 [0.315, 0.378]** | 1.83 [1.74, 1.93] | 0.0859 | 3749.3M | 0.023 [-0.014, 0.062] readiness_or_precision_insufficient | 0.148 [0.116, 0.182] readiness_or_precision_insufficient |
| J-1e-04 | ICLR J (lambda 1e-4) | 0.123 [0.098, 0.150] | 0.242 [0.205, 0.282] | **0.394 [0.352, 0.438]** | 2.25 [2.15, 2.35] | 0.1400 | 2934.2M | 0.072 [0.033, 0.110] readiness_or_precision_insufficient | 0.197 [0.154, 0.241] readiness_or_precision_insufficient |
| J-dec-1e-01 | decision-aware J_dec (lambda 1e-1) | 0.139 [0.117, 0.162] | 0.298 [0.266, 0.331] | **0.445 [0.403, 0.487]** | 2.07 [2.00, 2.14] | 0.0168 | 362.6M | 0.122 [0.077, 0.168] readiness_or_precision_insufficient | 0.247 [0.207, 0.289] readiness_or_precision_insufficient |
| prior | no-model action prior | 0.140 [0.107, 0.175] | 0.268 [0.225, 0.312] | **0.323 [0.278, 0.370]** | 1.74 [1.61, 1.86] | 0.0005 | 0 | - | 0.125 [0.078, 0.172] readiness_or_precision_insufficient |
| random | uniform random | 0.068 [0.053, 0.082] | 0.140 [0.118, 0.162] | **0.198 [0.172, 0.223]** | 1.93 [1.82, 2.05] | 0.0000 | 0 | -0.125 [-0.172, -0.078] readiness_or_precision_insufficient | - |
| oracle | oracle ceiling | 0.775 [0.733, 0.815] | 0.775 [0.733, 0.815] | **0.775 [0.733, 0.815]** | 1.00 [1.00, 1.00] | 0.0000 | 0 | 0.453 [0.403, 0.502] readiness_or_precision_insufficient | 0.578 [0.539, 0.616] readiness_or_precision_insufficient |

## N1-mixed

Levels 311, cells (level x seed) 933; recorded-inventory ceiling 1.000.

| Arm | Group | pass@1 | pass@2 | pass@3 | shots to pass (passed) | decision s | MACs / decision | pass@3 - prior | pass@3 - random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F-1-continuous | fixed request (hybrid LC) | 0.090 [0.061, 0.125] | 0.260 [0.212, 0.312] | **0.430 [0.385, 0.476]** | 2.40 [2.30, 2.51] | 0.1483 | 5156.8M | 0.015 [-0.035, 0.066] readiness_or_precision_insufficient | 0.176 [0.126, 0.226] readiness_or_precision_insufficient |
| F-1-micro | fixed request (hybrid LC) | 0.090 [0.061, 0.125] | 0.253 [0.215, 0.293] | **0.419 [0.375, 0.465]** | 2.38 [2.28, 2.47] | 0.1894 | 6463.6M | 0.004 [-0.058, 0.068] readiness_or_precision_insufficient | 0.165 [0.118, 0.213] readiness_or_precision_insufficient |
| F-1-macro | fixed request (hybrid LC) | 0.144 [0.116, 0.173] | 0.290 [0.250, 0.334] | **0.441 [0.386, 0.495]** | 2.01 [1.92, 2.12] | 0.1646 | 5272.9M | 0.026 [-0.019, 0.071] readiness_or_precision_insufficient | 0.186 [0.126, 0.248] readiness_or_precision_insufficient |
| F-5-continuous | fixed request (hybrid LC) | 0.232 [0.196, 0.268] | 0.428 [0.382, 0.475] | **0.583 [0.535, 0.633]** | 2.03 [1.94, 2.11] | 0.0309 | 1031.4M | 0.168 [0.111, 0.226] readiness_or_precision_insufficient | 0.329 [0.274, 0.385] readiness_or_precision_insufficient |
| F-5-micro | fixed request (hybrid LC) | 0.214 [0.179, 0.251] | 0.441 [0.394, 0.488] | **0.563 [0.516, 0.611]** | 1.99 [1.90, 2.08] | 0.0387 | 1292.7M | 0.148 [0.089, 0.207] readiness_or_precision_insufficient | 0.309 [0.259, 0.360] readiness_or_precision_insufficient |
| F-5-macro | fixed request (hybrid LC) | 0.137 [0.107, 0.170] | 0.379 [0.335, 0.424] | **0.532 [0.484, 0.581]** | 2.16 [2.09, 2.24] | 0.0342 | 1054.6M | 0.117 [0.065, 0.168] readiness_or_precision_insufficient | 0.278 [0.224, 0.332] readiness_or_precision_insufficient |
| F-15-continuous | fixed request (hybrid LC) | 0.243 [0.209, 0.279] | 0.428 [0.384, 0.472] | **0.615 [0.569, 0.663]** | 2.04 [1.96, 2.11] | 0.0110 | 343.8M | 0.200 [0.141, 0.259] readiness_or_precision_insufficient | 0.361 [0.311, 0.413] readiness_or_precision_insufficient |
| F-15-micro | fixed request (hybrid LC) | 0.284 [0.250, 0.320] | 0.450 [0.408, 0.492] | **0.589 [0.543, 0.638]** | 1.83 [1.76, 1.90] | 0.0138 | 430.9M | 0.175 [0.115, 0.236] readiness_or_precision_insufficient | 0.335 [0.286, 0.388] readiness_or_precision_insufficient |
| F-15-macro | fixed request (hybrid LC) | 0.194 [0.169, 0.219] | 0.387 [0.348, 0.426] | **0.546 [0.502, 0.591]** | 1.98 [1.91, 2.05] | 0.0121 | 351.5M | 0.131 [0.074, 0.188] readiness_or_precision_insufficient | 0.292 [0.242, 0.341] readiness_or_precision_insufficient |
| C-F-15 | C* fixed (15, continuous) | 0.191 [0.163, 0.220] | 0.477 [0.432, 0.522] | **0.599 [0.558, 0.640]** | 1.96 [1.87, 2.04] | 0.0084 | 444.5M | 0.184 [0.126, 0.241] readiness_or_precision_insufficient | 0.345 [0.297, 0.392] readiness_or_precision_insufficient |
| CJ-1e-04 | horizon-only adaptive C-J (lambda 1e-4) | 0.207 [0.177, 0.238] | 0.380 [0.343, 0.419] | **0.445 [0.412, 0.478]** | 1.83 [1.74, 1.93] | 0.0878 | 3835.3M | 0.030 [-0.018, 0.078] readiness_or_precision_insufficient | 0.191 [0.150, 0.233] readiness_or_precision_insufficient |
| J-1e-04 | ICLR J (lambda 1e-4) | 0.159 [0.128, 0.192] | 0.312 [0.266, 0.360] | **0.507 [0.459, 0.556]** | 2.25 [2.15, 2.35] | 0.1412 | 2978.3M | 0.092 [0.044, 0.140] readiness_or_precision_insufficient | 0.253 [0.199, 0.308] readiness_or_precision_insufficient |
| J-dec-1e-01 | decision-aware J_dec (lambda 1e-1) | 0.179 [0.153, 0.207] | 0.384 [0.346, 0.421] | **0.572 [0.527, 0.618]** | 2.07 [2.00, 2.14] | 0.0169 | 362.1M | 0.158 [0.101, 0.215] readiness_or_precision_insufficient | 0.318 [0.268, 0.370] readiness_or_precision_insufficient |
| prior | no-model action prior | 0.180 [0.138, 0.225] | 0.344 [0.293, 0.399] | **0.415 [0.360, 0.469]** | 1.74 [1.61, 1.86] | 0.0006 | 0 | - | 0.161 [0.103, 0.221] readiness_or_precision_insufficient |
| random | uniform random | 0.087 [0.070, 0.105] | 0.180 [0.155, 0.207] | **0.254 [0.225, 0.285]** | 1.93 [1.82, 2.05] | 0.0000 | 0 | -0.161 [-0.221, -0.103] readiness_or_precision_insufficient | - |
| oracle | oracle ceiling | 0.997 [0.990, 1.000] | 0.997 [0.990, 1.000] | **0.997 [0.990, 1.000]** | 1.00 [1.00, 1.00] | 0.0000 | 0 | 0.582 [0.527, 0.637] readiness_or_precision_insufficient | 0.743 [0.712, 0.772] readiness_or_precision_insufficient |

## N2-appearance

Levels 40, cells (level x seed) 120; recorded-inventory ceiling 0.500.

| Arm | Group | pass@1 | pass@2 | pass@3 | shots to pass (passed) | decision s | MACs / decision | pass@3 - prior | pass@3 - random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F-1-continuous | fixed request (hybrid LC) | 0.075 [0.000, 0.175] | 0.125 [0.025, 0.225] | **0.167 [0.067, 0.283]** | 2.00 [1.38, 2.62] | 0.1480 | 5156.8M | -0.008 [-0.133, 0.117] readiness_or_precision_insufficient | 0.067 [-0.033, 0.175] readiness_or_precision_insufficient |
| F-1-micro | fixed request (hybrid LC) | 0.075 [0.000, 0.175] | 0.117 [0.033, 0.217] | **0.158 [0.058, 0.267]** | 2.02 [1.44, 2.58] | 0.1891 | 6463.6M | -0.017 [-0.150, 0.117] readiness_or_precision_insufficient | 0.058 [-0.042, 0.167] readiness_or_precision_insufficient |
| F-1-macro | fixed request (hybrid LC) | 0.058 [0.017, 0.117] | 0.117 [0.042, 0.208] | **0.175 [0.075, 0.300]** | 2.00 [1.62, 2.38] | 0.1649 | 5272.9M | 0.000 [-0.125, 0.125] readiness_or_precision_insufficient | 0.075 [-0.033, 0.192] readiness_or_precision_insufficient |
| F-5-continuous | fixed request (hybrid LC) | 0.092 [0.025, 0.175] | 0.158 [0.067, 0.267] | **0.208 [0.100, 0.333]** | 1.98 [1.55, 2.43] | 0.0310 | 1031.4M | 0.033 [-0.100, 0.167] readiness_or_precision_insufficient | 0.108 [-0.008, 0.233] readiness_or_precision_insufficient |
| F-5-micro | fixed request (hybrid LC) | 0.083 [0.017, 0.167] | 0.158 [0.058, 0.275] | **0.200 [0.092, 0.317]** | 2.00 [1.53, 2.47] | 0.0388 | 1292.7M | 0.025 [-0.108, 0.158] readiness_or_precision_insufficient | 0.100 [-0.008, 0.217] readiness_or_precision_insufficient |
| F-5-macro | fixed request (hybrid LC) | 0.092 [0.025, 0.167] | 0.142 [0.058, 0.242] | **0.192 [0.083, 0.308]** | 1.92 [1.50, 2.35] | 0.0341 | 1054.6M | 0.017 [-0.108, 0.133] readiness_or_precision_insufficient | 0.092 [-0.017, 0.217] readiness_or_precision_insufficient |
| F-15-continuous | fixed request (hybrid LC) | 0.083 [0.017, 0.167] | 0.150 [0.058, 0.250] | **0.233 [0.125, 0.358]** | 2.17 [1.75, 2.57] | 0.0110 | 343.8M | 0.058 [-0.083, 0.200] readiness_or_precision_insufficient | 0.133 [0.025, 0.250] readiness_or_precision_insufficient |
| F-15-micro | fixed request (hybrid LC) | 0.108 [0.042, 0.183] | 0.158 [0.067, 0.258] | **0.217 [0.108, 0.333]** | 1.91 [1.55, 2.27] | 0.0137 | 430.9M | 0.042 [-0.092, 0.183] readiness_or_precision_insufficient | 0.117 [0.008, 0.233] readiness_or_precision_insufficient |
| F-15-macro | fixed request (hybrid LC) | 0.067 [0.025, 0.117] | 0.125 [0.050, 0.200] | **0.217 [0.117, 0.325]** | 2.21 [1.92, 2.51] | 0.0120 | 351.5M | 0.042 [-0.092, 0.175] readiness_or_precision_insufficient | 0.117 [0.017, 0.225] readiness_or_precision_insufficient |
| C-F-15 | C* fixed (15, continuous) | 0.092 [0.033, 0.158] | 0.150 [0.067, 0.250] | **0.225 [0.125, 0.333]** | 2.09 [1.74, 2.46] | 0.0083 | 444.5M | 0.050 [-0.075, 0.175] readiness_or_precision_insufficient | 0.125 [0.025, 0.233] readiness_or_precision_insufficient |
| CJ-1e-04 | horizon-only adaptive C-J (lambda 1e-4) | 0.083 [0.025, 0.150] | 0.125 [0.050, 0.208] | **0.158 [0.083, 0.242]** | 1.86 [1.41, 2.36] | 0.0877 | 3828.4M | -0.017 [-0.133, 0.100] readiness_or_precision_insufficient | 0.058 [-0.025, 0.150] readiness_or_precision_insufficient |
| J-1e-04 | ICLR J (lambda 1e-4) | 0.075 [0.017, 0.142] | 0.125 [0.025, 0.225] | **0.183 [0.075, 0.308]** | 2.00 [1.54, 2.58] | 0.1552 | 3399.2M | 0.008 [-0.117, 0.133] readiness_or_precision_insufficient | 0.083 [-0.025, 0.200] readiness_or_precision_insufficient |
| J-dec-1e-01 | decision-aware J_dec (lambda 1e-1) | 0.042 [0.008, 0.083] | 0.117 [0.050, 0.192] | **0.217 [0.117, 0.333]** | 2.32 [2.00, 2.62] | 0.0171 | 362.8M | 0.042 [-0.092, 0.175] readiness_or_precision_insufficient | 0.117 [0.017, 0.225] readiness_or_precision_insufficient |
| prior | no-model action prior | 0.050 [0.000, 0.125] | 0.125 [0.025, 0.225] | **0.175 [0.075, 0.300]** | 2.00 [1.43, 2.57] | 0.0000 | 0 | - | 0.075 [-0.050, 0.208] readiness_or_precision_insufficient |
| random | uniform random | 0.017 [0.000, 0.042] | 0.050 [0.017, 0.092] | **0.100 [0.042, 0.175]** | 2.24 [1.72, 2.70] | 0.0000 | 0 | -0.075 [-0.208, 0.050] readiness_or_precision_insufficient | - |
| oracle | oracle ceiling | 0.500 [0.350, 0.650] | 0.500 [0.350, 0.650] | **0.500 [0.350, 0.650]** | 1.00 [1.00, 1.00] | 0.0000 | 0 | 0.325 [0.175, 0.475] readiness_or_precision_insufficient | 0.400 [0.267, 0.533] readiness_or_precision_insufficient |

## Declared contrasts (pass@3)

| Set | Contrast | Estimate | Token |
| --- | --- | --- | --- |
| N1 | J-1e-04 - CJ-1e-04 | 0.048 [0.028, 0.069] | readiness_or_precision_insufficient |
| N1 | J-dec-1e-01 - J-1e-04 | 0.051 [0.028, 0.075] | readiness_or_precision_insufficient |
| N1 | F-15-macro - C-F-15 | -0.042 [-0.060, -0.025] | readiness_or_precision_insufficient |
| N1 | prior - random | 0.125 [0.078, 0.172] | readiness_or_precision_insufficient |
| N1-mixed | J-1e-04 - CJ-1e-04 | 0.062 [0.035, 0.089] | readiness_or_precision_insufficient |
| N1-mixed | J-dec-1e-01 - J-1e-04 | 0.065 [0.036, 0.095] | readiness_or_precision_insufficient |
| N1-mixed | F-15-macro - C-F-15 | -0.054 [-0.076, -0.031] | readiness_or_precision_insufficient |
| N1-mixed | prior - random | 0.161 [0.103, 0.221] | readiness_or_precision_insufficient |
| N2-appearance | J-1e-04 - CJ-1e-04 | 0.025 [-0.033, 0.075] | readiness_or_precision_insufficient |
| N2-appearance | J-dec-1e-01 - J-1e-04 | 0.033 [-0.025, 0.100] | readiness_or_precision_insufficient |
| N2-appearance | F-15-macro - C-F-15 | -0.008 [-0.033, 0.017] | readiness_or_precision_insufficient |
| N2-appearance | prior - random | 0.075 [-0.050, 0.208] | readiness_or_precision_insufficient |

## Request mix per shot (adaptive arms, N1; share of imagined rollout steps of the chosen candidate)

| Arm | Shot | Decisions | Chosen-candidate mix | All-candidate mix |
| --- | --- | --- | --- | --- |
| CJ-1e-04 | 1 | 1200 | 1-continuous 0.94, 15-continuous 0.06, 5-continuous 0.01 | 1-continuous 0.94, 15-continuous 0.06, 5-continuous 0.01 |
| CJ-1e-04 | 2 | 1001 | 1-continuous 0.94, 15-continuous 0.05 | 1-continuous 0.94, 15-continuous 0.05 |
| CJ-1e-04 | 3 | 837 | 1-continuous 0.94, 15-continuous 0.05 | 1-continuous 0.94, 15-continuous 0.06 |
| J-1e-04 | 1 | 1200 | 1-macro 0.70, 1-continuous 0.19, 15-micro 0.07, 1-micro 0.03, 15-macro 0.01 | 1-macro 0.79, 1-continuous 0.13, 15-micro 0.05, 1-micro 0.02, 15-macro 0.01 |
| J-1e-04 | 2 | 1046 | 1-macro 0.79, 1-continuous 0.12, 15-micro 0.06, 1-micro 0.02, 15-macro 0.01 | 1-macro 0.78, 1-continuous 0.12, 15-micro 0.05, 1-micro 0.02, 15-macro 0.01 |
| J-1e-04 | 3 | 903 | 1-macro 0.67, 1-continuous 0.24, 15-micro 0.05, 1-micro 0.02, 15-macro 0.01 | 1-macro 0.79, 1-continuous 0.13, 15-micro 0.05, 1-micro 0.02, 15-macro 0.01 |
| J-dec-1e-01 | 1 | 1200 | 15-continuous 0.46, 15-macro 0.38, 15-micro 0.16 | 15-continuous 0.51, 15-macro 0.39, 15-micro 0.10 |
| J-dec-1e-01 | 2 | 1026 | 15-continuous 0.45, 15-macro 0.42, 15-micro 0.13 | 15-continuous 0.51, 15-macro 0.40, 15-micro 0.09 |
| J-dec-1e-01 | 3 | 834 | 15-continuous 0.47, 15-macro 0.41, 15-micro 0.12 | 15-continuous 0.52, 15-macro 0.40, 15-micro 0.08 |

Fixed arms request one (Delta, description) pair at every step: F-1-continuous = 1-continuous, F-1-micro = 1-micro, F-1-macro = 1-macro, F-5-continuous = 5-continuous, F-5-micro = 5-micro, F-5-macro = 5-macro, F-15-continuous = 15-continuous, F-15-micro = 15-micro, F-15-macro = 15-macro, C-F-15 = 15-continuous.

## Determinism cross-check (declared; never the result)

Live vs recorded #104 verdict of the same level and candidate: pig_removed 5106/5106, stop kind 5106/5106; typed engine failures 17. Frames vs the recorded attempt of the same candidate (byte-equal): decision frame 5024, final frame 4398, frame count 5100; renderer string as frozen 5106; native physics trace (every step and event, run identifiers masked) equal 5032.

| Set | Arm | live pass@3 | recorded-replay pass@3 | cells whose outcome differs |
| --- | --- | --- | --- | --- |
| N1 | F-1-continuous | 0.334 [0.295, 0.375] | 0.337 [0.297, 0.377] | 3 |
| N1 | F-1-micro | 0.326 [0.287, 0.366] | 0.328 [0.290, 0.369] | 3 |
| N1 | F-1-macro | 0.343 [0.297, 0.390] | 0.345 [0.297, 0.393] | 3 |
| N1 | F-5-continuous | 0.453 [0.408, 0.499] | 0.456 [0.411, 0.502] | 3 |
| N1 | F-5-micro | 0.438 [0.394, 0.481] | 0.440 [0.397, 0.483] | 3 |
| N1 | F-5-macro | 0.413 [0.371, 0.457] | 0.416 [0.373, 0.459] | 3 |
| N1 | F-15-continuous | 0.478 [0.434, 0.523] | 0.481 [0.437, 0.526] | 3 |
| N1 | F-15-micro | 0.458 [0.416, 0.502] | 0.461 [0.417, 0.505] | 3 |
| N1 | F-15-macro | 0.424 [0.384, 0.465] | 0.427 [0.386, 0.468] | 3 |
| N1 | C-F-15 | 0.466 [0.427, 0.505] | 0.468 [0.428, 0.508] | 3 |
| N1 | CJ-1e-04 | 0.346 [0.315, 0.378] | 0.347 [0.317, 0.379] | 2 |
| N1 | J-1e-04 | 0.394 [0.352, 0.438] | 0.397 [0.354, 0.441] | 3 |
| N1 | J-dec-1e-01 | 0.445 [0.403, 0.487] | 0.448 [0.405, 0.491] | 3 |
| N1 | prior | 0.323 [0.278, 0.370] | 0.325 [0.280, 0.372] | 3 |
| N1 | random | 0.198 [0.172, 0.223] | 0.200 [0.174, 0.226] | 3 |
| N1 | oracle | 0.775 [0.733, 0.815] | 0.777 [0.735, 0.818] | 3 |
| N2-appearance | F-1-continuous | 0.167 [0.067, 0.283] | 0.167 [0.067, 0.283] | 0 |
| N2-appearance | F-1-micro | 0.158 [0.058, 0.267] | 0.158 [0.058, 0.267] | 0 |
| N2-appearance | F-1-macro | 0.175 [0.075, 0.300] | 0.175 [0.075, 0.300] | 0 |
| N2-appearance | F-5-continuous | 0.208 [0.100, 0.333] | 0.208 [0.100, 0.333] | 0 |
| N2-appearance | F-5-micro | 0.200 [0.092, 0.317] | 0.200 [0.092, 0.317] | 0 |
| N2-appearance | F-5-macro | 0.192 [0.083, 0.308] | 0.192 [0.083, 0.308] | 0 |
| N2-appearance | F-15-continuous | 0.233 [0.125, 0.358] | 0.233 [0.125, 0.358] | 0 |
| N2-appearance | F-15-micro | 0.217 [0.108, 0.333] | 0.217 [0.108, 0.333] | 0 |
| N2-appearance | F-15-macro | 0.217 [0.117, 0.325] | 0.217 [0.117, 0.325] | 0 |
| N2-appearance | C-F-15 | 0.225 [0.125, 0.333] | 0.225 [0.125, 0.333] | 0 |
| N2-appearance | CJ-1e-04 | 0.158 [0.083, 0.242] | 0.158 [0.083, 0.242] | 0 |
| N2-appearance | J-1e-04 | 0.183 [0.075, 0.308] | 0.183 [0.075, 0.308] | 0 |
| N2-appearance | J-dec-1e-01 | 0.217 [0.117, 0.333] | 0.217 [0.117, 0.333] | 0 |
| N2-appearance | prior | 0.175 [0.075, 0.300] | 0.175 [0.075, 0.300] | 0 |
| N2-appearance | random | 0.100 [0.042, 0.175] | 0.100 [0.042, 0.175] | 0 |
| N2-appearance | oracle | 0.500 [0.350, 0.650] | 0.500 [0.350, 0.650] | 0 |

## POST-HOC: decision-frame sensitivity (declared after G4 failed; changes no token)

Levels whose live decision frame parsed beyond the tolerance: issue-109-sc-evaluation-type010103-021, issue-109-sc-evaluation-type010105-140, issue-109-sc-evaluation-type010105-153, issue-110-evaluation-type010104-novel-001, issue-110-evaluation-type010105-novel-008. Re-ranking from each distinct live decision frame with every model arm and seed keeps the first pick in 351/351 and the first three picks in 351/351 comparisons.

## Guards

- G1_scoring_policy: pass ({"records": 132})
- G2_decision_determinism: pass ({"rerank_checked": 45360, "retained_checked": 12129})
- G3_chronology: pass ({"first_decision": "2026-10-15T00:20:27Z", "first_dispatch": "2026-10-15T01:10:49Z", "frozen_at": "2026-10-15T00:19:38Z"})
- G4_decision_frame: FAIL ({"complete_executions": 5106, "max_abs_diff": 0.0031819287687540054, "scoring_policy": true, "tolerance": 0.001})
- G5_execution_integrity: pass ({"chosen_executions": 5123, "executions": 5123, "incomplete_episodes": 0})

## Compute

GPU 3379 s; stage wall 72891 s; engine (per-execution worker wall) 953176 s; engine campaign wall 60001 s on 16 workers; live executions 5123. Smoke (pre-freeze): GPU 10 s, engine 4150 s, campaign wall 617 s.

