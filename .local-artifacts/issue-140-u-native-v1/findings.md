# #140 U-native (budget-safeguard control): findings

Global guards pass: True. U-native = U's #112 v3 retrain recipe at 16,929 updates (1.881 x U's 9,000); decision https://github.com/Sino-Huang/NovPhy/issues/140#issuecomment-6479886189.

| arm | role | params | guards | grid AUC [CI] | X − U [CI] | token X−U | token U−X | disposition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U-native | control | 1,837,690 | fail (G_action) | 0.8132 [0.7977, 0.8287] | +0.0688 [+0.0579, +0.0795] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | 4 |

## A4 - U-native (grid; the control's purpose)

- A4 - U-native: +0.1813 [+0.1658, +0.1967]; token readiness_or_precision_insufficient (A4 - U-native), reverse readiness_or_precision_insufficient (U-native - A4); guards U-native False, A4 True
- Reading: A4 - U-native not supported: A4's wording stays budget-qualified.
- U-native - U (descriptive): +0.0688 [+0.0579, +0.0795]

## Training budget (measured)

| seed | U-native GPU-s | / U (1 294.19) | / A4 |
| --- | --- | --- | --- |
| 20260908 | 2189.4 | 1.692 | 0.899 |
| 20260909 | 2203.1 | 1.702 | 0.905 |
| 20260910 | 2028.4 | 1.567 | 0.833 |
| mean | 2140.3 | 1.654 | 0.879 |

## Wrong anchor (right - wrong, grid)

- U-native: +0.2020 [+0.1886, +0.2152] (readiness_or_precision_insufficient)

## Position error (descriptive; pooled mean, member-clustered interval)

| ranker | t = 15 | t = 30 | t = 60 | t = 150 | t = 225 |
| --- | --- | --- | --- | --- | --- |
| U-native | 0.0510 [0.0467, 0.0553] | 0.0519 [0.0475, 0.0564] | 0.0523 [0.0477, 0.0570] | 0.0514 [0.0471, 0.0558] | 0.0528 [0.0487, 0.0569] |
| U | 0.0505 [0.0460, 0.0550] | 0.0501 [0.0457, 0.0545] | 0.0484 [0.0443, 0.0526] | 0.0459 [0.0423, 0.0496] | 0.0474 [0.0440, 0.0508] |

Intervals are descriptive (member-clustered bootstrap); tokens follow the #127 delta rule.
