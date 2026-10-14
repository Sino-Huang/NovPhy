# Issue #112 v3 retrain on the #104 fit split (Gate-B checkpoints)

Frozen 2026-10-14T05:29:10Z before any data build or training. Validation: `python -u -m scripts.run_decision_dynamics_retrain --validate`.
Nothing is selected or scored here; no evaluation, policy or #110 capture is read.

## Training data

| quantity | retrain (#109 fit split) | v3 (12 #77 N1 lineages) |
| --- | --- | --- |
| levels | 480 | 12 |
| shots (complete branches) | 13919 (1 typed failure excluded) | 745 |
| windows | 220286 | 11780 |
| long windows | 69595 | 3725 |
| ranking cells | 680 (angle 314, grid 366) over 425 levels | 27 |
| carrier positions | 3904309 | 208886 |

## Members (arm LC, fold 'all', 9000 updates)

| unit | seed | complete | skipped | max pre-clip norm (update, request) | final local / long / rank | end state | usable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid F-15-macro | 20260908 | True | 0 | 4.24e+05 (7508, F-1-micro) | 0.1286 / 0.0090 / 0.3751 | no excursion (max ratio 1.17) | True |
| hybrid F-15-macro | 20260909 | True | 0 | 4.01e+03 (109, F-1-continuous) | 0.1550 / 0.0076 / 0.3503 | no excursion (max ratio 1.98) | True |
| hybrid F-15-macro | 20260910 | True | 0 | 4.32e+03 (84, F-1-macro) | 0.1226 / 0.0090 / 0.6301 | no excursion (max ratio 1.81) | True |
| continuous C-F-15 | 20260908 | True | 0 | 4.38e+09 (8042, C-F-5) | 0.0055 / 0.0055 / 0.3108 | no excursion (max ratio 0.79) | True |
| continuous C-F-15 | 20260909 | True | 0 | 4.58e+03 (64, C-F-1) | 0.0067 / 0.0127 / 0.3103 | no excursion (max ratio 1.18) | True |
| continuous C-F-15 | 20260910 | True | 0 | 5.73e+06 (4972, C-F-5) | 0.0045 / 0.0062 / 0.3078 | no excursion (max ratio 1.47) | True |

Updates with a pre-clip gradient norm above 1000 (clipped to the recipe's grad_clip; per request with any):

- hybrid 20260908: 104 of 9000 (F-1-continuous 39, F-1-macro 32, F-1-micro 33); after update 500: 1, updates 7508–7508
- hybrid 20260909: 108 of 9000 (F-1-continuous 40, F-1-macro 34, F-1-micro 34); none after update 500
- hybrid 20260910: 108 of 9000 (F-1-continuous 39, F-1-macro 35, F-1-micro 34); none after update 500
- continuous 20260908: 190 of 9000 (C-F-1 135, C-F-15 2, C-F-5 53); after update 500: 109, updates 2748–8552
- continuous 20260909: 92 of 9000 (C-F-1 92); none after update 500
- continuous 20260910: 88 of 9000 (C-F-1 86, C-F-5 2); after update 500: 6, updates 1936–4973

Median #77 local loss over each request's last 100 updates:

| hybrid seed | F-1-continuous | F-1-macro | F-1-micro | F-15-continuous | F-15-macro | F-15-micro | F-5-continuous | F-5-macro | F-5-micro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260908 | 0.0052 | 0.1403 | 0.0106 | 0.0049 | 0.1061 | 0.0112 | 0.0049 | 0.1299 | 0.0112 |
| 20260909 | 0.0057 | 0.1428 | 0.0112 | 0.0054 | 0.1066 | 0.0110 | 0.0055 | 0.1329 | 0.0110 |
| 20260910 | 0.0047 | 0.1458 | 0.0102 | 0.0050 | 0.1061 | 0.0106 | 0.0049 | 0.1325 | 0.0107 |

| continuous seed | C-F-1 | C-F-15 | C-F-5 |
| --- | --- | --- | --- |
| 20260908 | 0.0050 | 0.0047 | 0.0053 |
| 20260909 | 0.0051 | 0.0046 | 0.0051 |
| 20260910 | 0.0053 | 0.0047 | 0.0045 |

## Guards and handoff

- G1 isolation (fit-stream branches of fit-roster levels only): True
- G2 stability (no retirement, at most 45 skipped updates): True
- G3 end state (#117 excursion rule): {'hybrid/20260908': True, 'hybrid/20260909': True, 'hybrid/20260910': True, 'continuous/20260908': True, 'continuous/20260909': True, 'continuous/20260910': True}
- Usable seeds: Gate-B unit [20260908, 20260909, 20260910], C* [20260908, 20260909, 20260910]
- Gate-B handoff: **supported** (`gate_b_handoff.json`, supersedes the v3 handoff)

Compute: 32423 GPU s, 32483 wall s over all phases and workers; 0 engine seconds.
