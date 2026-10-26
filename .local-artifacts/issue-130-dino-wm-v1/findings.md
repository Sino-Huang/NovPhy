# #130 C1 DINO-WM: findings

Global guards pass: True; C1-car configuration car-release; G-fid (DINO-WM Tier E) pass True.

C1-dino: dropped before training (readiness_ceiling_below_threshold; R1 policy ceiling 0.5948 < 0.65); vetoes the program claim.

| arm | role | params | guards | grid AUC [CI] | X − U [CI] | token X−U | token U−X | disposition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1-car | secondary | 1,243,648 | fail (G_action) | 0.5251 [0.5093, 0.5411] | -0.2193 [-0.2412, -0.1985] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | 4 |
| C1-dino | primary (program set) | n/a | fail (G_readout) | n/a | n/a | readiness_or_precision_insufficient | readiness_or_precision_insufficient | 4 (dropped: readiness_ceiling_below_threshold) |

## Within family

- C1-dino - C1-car: n/a (readiness_or_precision_insufficient / readiness_or_precision_insufficient); not computable: C1-dino was not run (typed stop readiness_ceiling_below_threshold); every token is the stop token

## G-action in carrier space (DESCRIPTIVE; #127 decision: the frozen guard stands)

- C1-car: G-action per seed 20260908 0.0004 (0.01 x engine truth), 20260909 0.0004 (0.01 x engine truth), 20260910 0.0016 (0.04 x engine truth); engine truth 0.0395; the frozen G-action threshold 0.05 (set on #123 latent states) is not attainable in carrier space: the engine truth itself reaches 0.0395.

## Wrong anchor (right - wrong, grid)

- C1-car: -0.0101 [-0.0259, +0.0053] (readiness_or_precision_insufficient)

## P-novelty (#110 novelty levels; DESCRIPTIVE: no tokens, never part of a disposition)

Scored: none; not scored: C1-car (guards fail: ['G_action']); C1-dino (typed stop readiness_ceiling_below_threshold (never trained)); G-chronology (novelty anchors read after the primary records) True.

| ranker | novelty level | novel AUC [CI] | paired normal AUC [CI] | novel − normal [CI] | paired level pairs |
| --- | --- | --- | --- | --- | --- |
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

Intervals are descriptive (member-clustered bootstrap); tokens follow the #127 delta rule.
