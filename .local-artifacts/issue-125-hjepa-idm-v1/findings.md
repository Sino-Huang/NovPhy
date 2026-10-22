# Issue #125 findings: H-JEPA with its inverse-dynamics remedy vs the Gate-B unit (sealed #109 split)

Plan frozen 2026-10-21T07:26:34Z; trigger evaluated 2026-10-20T18:59:24Z (failing HJ-px/20260908, HJ-px/20260909, HJ-px/20260910, HJ-car-m/20260908, HJ-car-m/20260909, HJ-car-m/20260910); selected gamma_1 {'carriers': 225.0, 'pixels': 225.0}; lambda 0.18 (#123). Intervals are member (level)-clustered percentile bootstraps (10 000 draws, PCG64 7201), DESCRIPTIVE. delta = 0.02.

**Disposition 3: per-contrast tokens only** (outcome branch: no scoped ranking claim).

Disposition 4 (the remedy does not prevent collapse): True ({'HJ-px-IDM': {'G_collapse': False, 'G_action': False}, 'HJ-car-m-IDM': {'G_collapse': False, 'G_action': False}}).

## Guards

| guard | pass |
| --- | --- |
| G_fid | True |
| G_port | True |
| G_port_IDM | True |
| G_isolation | True |
| G_chronology | True |
| G_scoring_policy | True |
| G_records | True |

| arm | role | params | G-collapse | G-action | G-readout (policy ceiling) | G-nonfinite (R1 / R2) | trained | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HJ-car-m-IDM | primary, controlled + remedy | 1,836,500 | False | False | False (0.5623) | True (0.000 / 0.000) | True | False |
| HJ-px-IDM | primary, faithful + remedy | 35,860,884 | False | False | True (0.8562) | True (0.000 / 0.000) | True | False |
| LeWM-car-m-IDM | temporal lever under the remedy (carriers) | 1,835,196 | False | False | False (0.6193) | True (0.000 / 0.000) | True | False |
| LeWM-px-IDM | temporal lever under the remedy (pixels) | 11,563,900 | True | True | False (0.5698) | True (0.000 / 0.000) | True | False |
| HJ-px (#123, SI1 / SI2) | | | | | | | | False |
| HJ-car-m (#123, SI1 / SI2) | | | | | | | | False |

| arm | seed | erank / D | chi | sensitivity | policy R1 AUC | policy ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| HJ-car-m-IDM | 20260908 | 13.9 / 112 | 0.016 | 1.92e-05 | 0.4406 | 0.5749 |
| HJ-car-m-IDM | 20260909 | 13.6 / 112 | 0.014 | 0.000503 | 0.4652 | 0.5813 |
| HJ-car-m-IDM | 20260910 | 13.2 / 112 | 0.016 | 4.54e-05 | 0.5448 | 0.5306 |
| HJ-px-IDM | 20260908 | 110.8 / 192 | 0.001 | 1.67e-06 | 0.4871 | 0.8711 |
| HJ-px-IDM | 20260909 | 109.0 / 192 | 0.002 | 1.74e-05 | 0.4891 | 0.8613 |
| HJ-px-IDM | 20260910 | 110.9 / 192 | 0.002 | 2.65e-06 | 0.5246 | 0.8362 |
| LeWM-car-m-IDM | 20260908 | 25.7 / 112 | 0.044 | 0.00875 | 0.4995 | 0.6174 |
| LeWM-car-m-IDM | 20260909 | 25.5 / 112 | 0.045 | 0.0129 | 0.4606 | 0.6511 |
| LeWM-car-m-IDM | 20260910 | 25.4 / 112 | 0.045 | 0.00988 | 0.5221 | 0.5893 |
| LeWM-px-IDM | 20260908 | 162.7 / 192 | 0.058 | 0.995 | 0.4720 | 0.5885 |
| LeWM-px-IDM | 20260909 | 158.0 / 192 | 0.064 | 0.95 | 0.5015 | 0.5519 |
| LeWM-px-IDM | 20260910 | 159.4 / 192 | 0.055 | 1.07 | 0.4563 | 0.5690 |

## Grid rankers (311 mixed cells x 3 seeds)

| ranker | pooled AUC | seed 20260908 | seed 20260909 | seed 20260910 | top-1 | chance top-1 |
| --- | --- | --- | --- | --- | --- | --- |
| U | 0.7444 [0.7269, 0.7620] | 0.7141 [0.6917, 0.7362] | 0.8011 [0.7826, 0.8199] | 0.7181 [0.6961, 0.7404] | 0.195 | 0.089 |
| C* | 0.7834 [0.7676, 0.7992] | 0.8262 [0.8091, 0.8431] | 0.7334 [0.7127, 0.7536] | 0.7905 [0.7688, 0.8120] | 0.192 | 0.089 |
| P | 0.7203 [0.6965, 0.7442] | seed-free | |  | 0.180 | 0.089 |
| J | 0.7057 [0.6846, 0.7267] | 0.6690 [0.6449, 0.6931] | 0.7012 [0.6787, 0.7234] | 0.7470 [0.7261, 0.7678] | 0.160 | 0.089 |
| J_dec | 0.7460 [0.7277, 0.7637] | 0.7205 [0.6989, 0.7417] | 0.7933 [0.7739, 0.8124] | 0.7242 [0.7035, 0.7451] | 0.180 | 0.089 |
| HJ-car-m-IDM | 0.4900 [0.4685, 0.5112] | 0.4447 [0.4144, 0.4744] | 0.4788 [0.4504, 0.5071] | 0.5465 [0.5171, 0.5755] | 0.040 | 0.089 |
| HJ-px-IDM | 0.4948 [0.4779, 0.5122] | 0.4828 [0.4531, 0.5123] | 0.5069 [0.4793, 0.5357] | 0.4945 [0.4645, 0.5250] | 0.069 | 0.089 |
| LeWM-car-m-IDM | 0.4790 [0.4630, 0.4949] | 0.4832 [0.4565, 0.5108] | 0.4743 [0.4452, 0.5035] | 0.4794 [0.4523, 0.5068] | 0.054 | 0.089 |
| LeWM-px-IDM | 0.4971 [0.4801, 0.5144] | 0.5037 [0.4736, 0.5344] | 0.5052 [0.4757, 0.5353] | 0.4824 [0.4523, 0.5124] | 0.062 | 0.089 |
| HJ-px | 0.4997 [0.4827, 0.5165] | 0.4874 [0.4575, 0.5169] | 0.4995 [0.4681, 0.5308] | 0.5121 [0.4820, 0.5422] | 0.093 | 0.089 |
| HJ-car-m | 0.4896 [0.4694, 0.5099] | 0.4982 [0.4660, 0.5298] | 0.4856 [0.4552, 0.5157] | 0.4852 [0.4536, 0.5172] | 0.079 | 0.089 |
| HJ-car-m-IDM:R2 | 0.4932 [0.4787, 0.5081] | 0.4026 [0.3730, 0.4316] | 0.5663 [0.5392, 0.5936] | 0.5106 [0.4828, 0.5387] | 0.065 | 0.089 |
| HJ-car-m-IDM:R3 | 0.4850 [0.4710, 0.4991] | 0.5516 [0.5216, 0.5810] | 0.5402 [0.5127, 0.5679] | 0.3632 [0.3381, 0.3889] | 0.070 | 0.089 |
| HJ-car-m-IDM:R4 | 0.4940 [0.4792, 0.5091] | 0.3997 [0.3709, 0.4289] | 0.5670 [0.5400, 0.5939] | 0.5152 [0.4867, 0.5438] | 0.063 | 0.089 |
| HJ-px-IDM:R2 | 0.4898 [0.4724, 0.5074] | 0.4819 [0.4501, 0.5128] | 0.4866 [0.4575, 0.5153] | 0.5010 [0.4721, 0.5300] | 0.073 | 0.089 |
| HJ-px-IDM:R3 | 0.5120 [0.4962, 0.5281] | 0.4981 [0.4688, 0.5273] | 0.5455 [0.5163, 0.5745] | 0.4925 [0.4618, 0.5229] | 0.077 | 0.089 |
| HJ-px-IDM:R4 | 0.4909 [0.4729, 0.5086] | 0.4767 [0.4465, 0.5059] | 0.4933 [0.4638, 0.5224] | 0.5028 [0.4734, 0.5321] | 0.065 | 0.089 |
| LeWM-car-m-IDM:R2 | 0.4783 [0.4645, 0.4919] | 0.4961 [0.4725, 0.5190] | 0.4862 [0.4629, 0.5092] | 0.4526 [0.4307, 0.4746] | 0.021 | 0.089 |
| LeWM-car-m-IDM:R3 | 0.4863 [0.4697, 0.5025] | 0.5255 [0.4984, 0.5524] | 0.4952 [0.4677, 0.5214] | 0.4381 [0.4116, 0.4640] | 0.033 | 0.089 |
| LeWM-car-m-IDM:R4 | 0.4811 [0.4672, 0.4948] | 0.4972 [0.4734, 0.5198] | 0.4880 [0.4628, 0.5128] | 0.4581 [0.4356, 0.4810] | 0.034 | 0.089 |
| LeWM-px-IDM:R2 | 0.5035 [0.4849, 0.5222] | 0.5096 [0.4780, 0.5399] | 0.4902 [0.4597, 0.5208] | 0.5106 [0.4800, 0.5419] | 0.059 | 0.089 |
| LeWM-px-IDM:R3 | 0.5096 [0.4921, 0.5271] | 0.5215 [0.4911, 0.5517] | 0.5049 [0.4724, 0.5367] | 0.5023 [0.4735, 0.5312] | 0.069 | 0.089 |
| LeWM-px-IDM:R4 | 0.5095 [0.4910, 0.5278] | 0.5109 [0.4796, 0.5407] | 0.5062 [0.4750, 0.5378] | 0.5114 [0.4823, 0.5413] | 0.061 | 0.089 |

## Contrasts (grid, paired per (level, seed))

| contrast | first - second | estimate | token (first - second) | token (second - first) | guards |
| --- | --- | --- | --- | --- | --- |
| PI1 | HJ-px-IDM - U | -0.2496 [-0.2739, -0.2246] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| PI2 | HJ-car-m-IDM - U | -0.2544 [-0.2863, -0.2243] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI1 | HJ-px-IDM - HJ-px | -0.0049 [-0.0291, +0.0197] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI2 | HJ-car-m-IDM - HJ-car-m | +0.0003 [-0.0282, +0.0284] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI3 | HJ-px-IDM - LeWM-px-IDM | -0.0023 [-0.0268, +0.0214] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI4 | HJ-car-m-IDM - LeWM-car-m-IDM | +0.0110 [-0.0171, +0.0385] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI5_pixels | HJ-px-IDM - C* | -0.2886 [-0.3116, -0.2653] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI5_carriers | HJ-car-m-IDM - C* | -0.2934 [-0.3233, -0.2649] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI6_pixels | HJ-px-IDM - P | -0.2255 [-0.2539, -0.1966] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI6_carriers | HJ-car-m-IDM - P | -0.2303 [-0.2651, -0.1962] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI7_pixels_J | HJ-px-IDM - J | -0.2110 [-0.2380, -0.1834] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI7_pixels_J_dec | HJ-px-IDM - J_dec | -0.2512 [-0.2758, -0.2260] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI7_carriers_J | HJ-car-m-IDM - J | -0.2157 [-0.2499, -0.1832] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |
| SI7_carriers_J_dec | HJ-car-m-IDM - J_dec | -0.2560 [-0.2882, -0.2258] | readiness_or_precision_insufficient | readiness_or_precision_insufficient | False |

SI8 wrong-anchor control (right - wrong, 480 fit anchors):

| arm | right | wrong | right - wrong | token |
| --- | --- | --- | --- | --- |
| HJ-px-IDM | 0.4948 [0.4779, 0.5122] | 0.5037 [0.5026, 0.5048] | -0.0089 [-0.0257, +0.0086] | readiness_or_precision_insufficient |
| HJ-car-m-IDM | 0.4900 [0.4685, 0.5112] | 0.5019 [0.5002, 0.5036] | -0.0119 [-0.0333, +0.0093] | readiness_or_precision_insufficient |

## Descriptive inventories (no tokens)

| ranker | offset | angle | power |
| --- | --- | --- | --- |
| U | 0.6670 [0.6503, 0.6827] | 0.6791 [0.6663, 0.6912] | 0.8058 [0.7729, 0.8396] |
| C* | 0.7748 [0.7524, 0.7962] | 0.7590 [0.7422, 0.7752] | 0.8068 [0.7711, 0.8440] |
| P | 0.5000 [0.5000, 0.5000] | 0.8119 [0.7778, 0.8450] | 0.5000 [0.5000, 0.5000] |
| J | 0.7522 [0.7039, 0.7978] | 0.7765 [0.7402, 0.8115] | 0.8076 [0.7695, 0.8463] |
| J_dec | 0.6904 [0.6737, 0.7062] | 0.6988 [0.6827, 0.7147] | 0.7943 [0.7555, 0.8331] |
| HJ-car-m-IDM | 0.4807 [0.4504, 0.5098] | 0.4872 [0.4670, 0.5064] | 0.4775 [0.4328, 0.5239] |
| HJ-px-IDM | 0.4810 [0.4419, 0.5196] | 0.4936 [0.4589, 0.5290] | 0.4986 [0.4538, 0.5440] |
| LeWM-car-m-IDM | 0.5090 [0.4680, 0.5494] | 0.4799 [0.4476, 0.5119] | 0.4713 [0.4203, 0.5256] |
| LeWM-px-IDM | 0.4901 [0.4547, 0.5258] | 0.5129 [0.4791, 0.5463] | 0.4807 [0.4327, 0.5285] |
| HJ-px | 0.5004 [0.4625, 0.5386] | 0.4367 [0.3995, 0.4741] | 0.4947 [0.4541, 0.5370] |
| HJ-car-m | 0.4792 [0.4338, 0.5235] | 0.5019 [0.4638, 0.5397] | 0.4982 [0.4547, 0.5428] |
| HJ-car-m-IDM:R2 | 0.5392 [0.5107, 0.5666] | 0.5318 [0.5107, 0.5532] | 0.5129 [0.4729, 0.5532] |
| HJ-car-m-IDM:R3 | 0.4761 [0.4471, 0.5033] | 0.4957 [0.4720, 0.5198] | 0.5332 [0.4826, 0.5787] |
| HJ-car-m-IDM:R4 | 0.5462 [0.5179, 0.5734] | 0.5293 [0.5064, 0.5519] | 0.5274 [0.4882, 0.5684] |
| HJ-px-IDM:R2 | 0.5058 [0.4700, 0.5427] | 0.5208 [0.4848, 0.5583] | 0.4652 [0.4125, 0.5164] |
| HJ-px-IDM:R3 | 0.4800 [0.4414, 0.5200] | 0.4964 [0.4649, 0.5272] | 0.5347 [0.4873, 0.5818] |
| HJ-px-IDM:R4 | 0.4783 [0.4404, 0.5165] | 0.5295 [0.4897, 0.5687] | 0.4675 [0.4128, 0.5212] |
| LeWM-car-m-IDM:R2 | 0.5049 [0.4775, 0.5327] | 0.5026 [0.4813, 0.5244] | 0.4974 [0.4559, 0.5381] |
| LeWM-car-m-IDM:R3 | 0.4909 [0.4580, 0.5248] | 0.5040 [0.4763, 0.5312] | 0.5378 [0.4848, 0.5893] |
| LeWM-car-m-IDM:R4 | 0.5013 [0.4746, 0.5295] | 0.4984 [0.4754, 0.5222] | 0.5119 [0.4686, 0.5545] |
| LeWM-px-IDM:R2 | 0.4959 [0.4595, 0.5329] | 0.4981 [0.4545, 0.5410] | 0.4429 [0.3869, 0.4987] |
| LeWM-px-IDM:R3 | 0.4843 [0.4422, 0.5276] | 0.4903 [0.4550, 0.5247] | 0.4942 [0.4531, 0.5355] |
| LeWM-px-IDM:R4 | 0.5284 [0.4904, 0.5678] | 0.5155 [0.4755, 0.5548] | 0.4673 [0.4145, 0.5174] |

## Engine position MSE (#100 metric; P_c for the IDM arms, predicted carriers for U / C*)

| model | t = 15 | t = 30 | t = 60 | t = 150 | t = 225 | floor t = 225 |
| --- | --- | --- | --- | --- | --- | --- |
| HJ-car-m-IDM | 0.0130 [0.0123, 0.0136] | 0.0152 [0.0144, 0.0159] | 0.0160 [0.0151, 0.0170] | 0.0282 [0.0261, 0.0303] | 0.0390 [0.0359, 0.0422] | 0.0113 [0.0109, 0.0118] |
| HJ-px-IDM | 0.0138 [0.0132, 0.0144] | 0.0139 [0.0133, 0.0145] | 0.0126 [0.0120, 0.0132] | 0.0124 [0.0118, 0.0130] | 0.0155 [0.0149, 0.0161] | 0.0147 [0.0140, 0.0153] |
| LeWM-car-m-IDM | 0.0135 [0.0130, 0.0140] | 0.0138 [0.0133, 0.0144] | 0.0138 [0.0132, 0.0144] | 0.0148 [0.0142, 0.0154] | 0.0187 [0.0181, 0.0193] | 0.0120 [0.0115, 0.0126] |
| LeWM-px-IDM | 0.0115 [0.0110, 0.0120] | 0.0128 [0.0123, 0.0134] | 0.0130 [0.0125, 0.0135] | 0.0198 [0.0190, 0.0205] | 0.0247 [0.0238, 0.0256] | 0.0105 [0.0100, 0.0110] |
| U | 0.0505 [0.0460, 0.0550] | 0.0501 [0.0457, 0.0545] | 0.0484 [0.0443, 0.0526] | 0.0459 [0.0423, 0.0496] | 0.0474 [0.0440, 0.0508] | n/a |
| C* | 0.0547 [0.0500, 0.0595] | 0.0545 [0.0499, 0.0592] | 0.0543 [0.0497, 0.0589] | 0.0594 [0.0544, 0.0645] | 0.0670 [0.0617, 0.0723] | n/a |

## Compute

- issue-125 GPU seconds 226279 (62.9 GPU-h of active compute; cap 120); engine seconds 0.
