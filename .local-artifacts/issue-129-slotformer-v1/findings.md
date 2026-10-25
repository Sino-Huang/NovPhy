# #129 B1 SlotFormer-E99: findings

Global guards pass: True; selected configuration C3-s15-long.

| arm | role | params | guards | grid AUC [CI] | X − U [CI] | token X−U | token U−X | disposition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | primary (program set) | 6,325,005 | fail (G_action) | 0.5476 [0.5264, 0.5692] | -0.1968 [-0.2228, -0.1705] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | 4 |
| B1-m | secondary (matched budget and parameters) | 1,793,445 | fail (G_action) | 0.5348 [0.5176, 0.5523] | -0.2096 [-0.2303, -0.1899] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | 4 |

## Within family

- B1 - B1-m: +0.0128 [-0.0196, +0.0447] (readiness_or_precision_insufficient / readiness_or_precision_insufficient)
- B1-head - B1: +0.0593 [+0.0227, +0.0965] (readiness_or_precision_insufficient / readiness_or_precision_insufficient)
- U+head - U: +0.0679 [+0.0583, +0.0773] (supported / not_supported_by_this_experiment)
- B1-head - U+head: -0.2054 [-0.2303, -0.1797] (readiness_or_precision_insufficient / readiness_or_precision_insufficient)

## Wrong anchor (right - wrong, grid)

- B1: -0.0065 [-0.0239, +0.0113] (readiness_or_precision_insufficient)
- B1-m: +0.0097 [+0.0030, +0.0160] (readiness_or_precision_insufficient)

## CLEVRER / Physion-Dominoes (EXPLORATORY; no ranking, no tokens)

Action-free B1 (selected configuration, 9 000 x 64, null launch) on the #117 item-5 annotation carriers; #79 position_mse from contexts 0-3, mean over scenes x contexts x seeds.

**clevrer**

| row | t = 15 | t = 30 | t = 60 | t = 120 |
| --- | --- | --- | --- | --- |
| B1-bench | 0.005404 | 0.004412 | 0.004158 | 0.002526 |
| J (#117 item 5, LC, full) F-15-continuous | 0.001795 | 0.002058 | 0.001214 | 0.006567 |
| J (#117 item 5, LC, full) F-5-continuous | 0.002574 | 0.004234 | 0.004886 | 0.01537 |

**physion**

| row | t = 15 | t = 30 | t = 60 | t = 120 |
| --- | --- | --- | --- | --- |
| B1-bench | 7.7e-05 | 0.0001457 | 0.0002473 | 0.0004823 |
| J (#117 item 5, LC, full) F-15-continuous | 0.0001865 | 0.0002947 | 0.0005389 | 0.001101 |
| J (#117 item 5, LC, full) F-5-continuous | 0.0004014 | 0.0008574 | 0.001726 | 0.002862 |


Intervals are descriptive (member-clustered bootstrap); tokens follow the #127 delta rule.
