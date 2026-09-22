# Issue-77 N2 appearance evaluation - findings

Evaluation complete: True. Claim boundary: two appearance cells only; appearance is the perception control; zero-shot and few-shot reported separately and never mixed.

## Zero-shot (frozen N1 checkpoints; no novelty fitting)

| Pair | Side | System | Seed-mean regret (per seed) |
| --- | --- | --- | --- |
| type010101 | normal | continuous_h1 | 20260908=0.7558; 20260909=0.5894; 20260910=0.5894 |
| type010101 | normal | continuous_h5 | 20260908=0.6636; 20260909=0.5436; 20260910=0.6086 |
| type010101 | normal | continuous_h15 | 20260908=0.6644; 20260909=0.6703; 20260910=0.6717 |
| type010101 | normal | hybrid_continuous_h1 | 20260908=0.5894; 20260909=0.6340; 20260910=0.5902 |
| type010101 | normal | hybrid_micro_h1 | 20260908=0.5573; 20260909=0.6240; 20260910=0.6437 |
| type010101 | normal | hybrid_macro_h1 | 20260908=0.5855; 20260909=0.6240; 20260910=0.5894 |
| type010101 | normal | hybrid_continuous_h5 | 20260908=0.7599; 20260909=0.7544; 20260910=0.6364 |
| type010101 | normal | hybrid_micro_h5 | 20260908=0.5263; 20260909=0.5855; 20260910=0.4245 |
| type010101 | normal | hybrid_macro_h5 | 20260908=0.6903; 20260909=0.6093; 20260910=0.5894 |
| type010101 | normal | hybrid_continuous_h15 | 20260908=0.6724; 20260909=0.7144; 20260910=0.5884 |
| type010101 | normal | hybrid_micro_h15 | 20260908=0.6031; 20260909=0.5161; 20260910=0.6628 |
| type010101 | normal | hybrid_macro_h15 | 20260908=0.3271; 20260909=0.6063; 20260910=0.6628 |
| type010101 | novel | continuous_h1 | 20260908=0.0504; 20260909=0.4147; 20260910=0.0504 |
| type010101 | novel | continuous_h5 | 20260908=0.0504; 20260909=0.0504; 20260910=0.0504 |
| type010101 | novel | continuous_h15 | 20260908=0.4464; 20260909=0.5135; 20260910=0.0555 |
| type010101 | novel | hybrid_continuous_h1 | 20260908=0.0504; 20260909=0.0504; 20260910=0.0951 |
| type010101 | novel | hybrid_micro_h1 | 20260908=0.0504; 20260909=0.3371; 20260910=0.0504 |
| type010101 | novel | hybrid_macro_h1 | 20260908=0.0504; 20260909=0.0504; 20260910=0.0504 |
| type010101 | novel | hybrid_continuous_h5 | 20260908=0.5041; 20260909=0.0504; 20260910=0.0504 |
| type010101 | novel | hybrid_micro_h5 | 20260908=0.0504; 20260909=0.0821; 20260910=0.0504 |
| type010101 | novel | hybrid_macro_h5 | 20260908=0.0504; 20260909=0.0000; 20260910=0.4147 |
| type010101 | novel | hybrid_continuous_h15 | 20260908=0.5135; 20260909=0.5135; 20260910=0.5358 |
| type010101 | novel | hybrid_micro_h15 | 20260908=0.5135; 20260909=0.5135; 20260910=0.5358 |
| type010101 | novel | hybrid_macro_h15 | 20260908=0.5135; 20260909=0.5135; 20260910=0.5135 |
| type010102 | normal | continuous_h1 | 20260908=0.3904; 20260909=0.3904; 20260910=0.3904 |
| type010102 | normal | continuous_h5 | 20260908=0.7062; 20260909=0.3904; 20260910=0.7569 |
| type010102 | normal | continuous_h15 | 20260908=0.8609; 20260909=0.4645; 20260910=0.4645 |
| type010102 | normal | hybrid_continuous_h1 | 20260908=0.3904; 20260909=0.3904; 20260910=0.5584 |
| type010102 | normal | hybrid_micro_h1 | 20260908=0.3575; 20260909=0.3904; 20260910=0.4936 |
| type010102 | normal | hybrid_macro_h1 | 20260908=0.3904; 20260909=0.2385; 20260910=0.3904 |
| type010102 | normal | hybrid_continuous_h5 | 20260908=0.5584; 20260909=0.4322; 20260910=0.3904 |
| type010102 | normal | hybrid_micro_h5 | 20260908=0.3904; 20260909=0.4677; 20260910=0.3904 |
| type010102 | normal | hybrid_macro_h5 | 20260908=0.8609; 20260909=0.6289; 20260910=0.3904 |
| type010102 | normal | hybrid_continuous_h15 | 20260908=0.3105; 20260909=0.3105; 20260910=0.3904 |
| type010102 | normal | hybrid_micro_h15 | 20260908=0.3105; 20260909=0.3557; 20260910=0.3904 |
| type010102 | normal | hybrid_macro_h15 | 20260908=0.8931; 20260909=0.2015; 20260910=0.3904 |
| type010102 | novel | continuous_h1 | 20260908=0.7958; 20260909=0.2047; 20260910=0.2047 |
| type010102 | novel | continuous_h5 | 20260908=0.2047; 20260909=0.2047; 20260910=0.2047 |
| type010102 | novel | continuous_h15 | 20260908=0.8221; 20260909=0.7573; 20260910=0.7452 |
| type010102 | novel | hybrid_continuous_h1 | 20260908=0.2047; 20260909=0.6807; 20260910=0.2047 |
| type010102 | novel | hybrid_micro_h1 | 20260908=0.2047; 20260909=0.4582; 20260910=0.2047 |
| type010102 | novel | hybrid_macro_h1 | 20260908=0.2047; 20260909=0.2082; 20260910=0.2047 |
| type010102 | novel | hybrid_continuous_h5 | 20260908=0.2082; 20260909=0.2047; 20260910=0.6498 |
| type010102 | novel | hybrid_micro_h5 | 20260908=0.2047; 20260909=0.2047; 20260910=0.2047 |
| type010102 | novel | hybrid_macro_h5 | 20260908=0.3628; 20260909=0.0850; 20260910=0.2047 |
| type010102 | novel | hybrid_continuous_h15 | 20260908=0.7660; 20260909=0.7660; 20260910=0.8221 |
| type010102 | novel | hybrid_micro_h15 | 20260908=0.7617; 20260909=0.3664; 20260910=0.8221 |
| type010102 | novel | hybrid_macro_h15 | 20260908=0.8221; 20260909=0.3664; 20260910=0.3664 |

## Few-shot (adapted on novel predictor lineages; budgets in summary.json)

| Pair | Side | System | Seed-mean regret (per seed) |
| --- | --- | --- | --- |
| type010101 | normal | continuous_h1 | 20260908=0.6597; 20260909=0.5894; 20260910=0.5894 |
| type010101 | normal | continuous_h5 | 20260908=0.5894; 20260909=0.5894; 20260910=0.6231 |
| type010101 | normal | continuous_h15 | 20260908=0.6644; 20260909=0.7024; 20260910=0.6644 |
| type010101 | normal | hybrid_continuous_h1 | 20260908=0.6240; 20260909=0.4631; 20260910=0.5041 |
| type010101 | normal | hybrid_micro_h1 | 20260908=0.5894; 20260909=0.5239; 20260910=0.6291 |
| type010101 | normal | hybrid_macro_h1 | 20260908=0.7628; 20260909=0.6162; 20260910=0.6448 |
| type010101 | normal | hybrid_continuous_h5 | 20260908=0.8199; 20260909=0.6544; 20260910=0.6239 |
| type010101 | normal | hybrid_micro_h5 | 20260908=0.7527; 20260909=0.5711; 20260910=0.4783 |
| type010101 | normal | hybrid_macro_h5 | 20260908=0.6613; 20260909=0.7074; 20260910=0.6602 |
| type010101 | normal | hybrid_continuous_h15 | 20260908=0.6644; 20260909=0.7944; 20260910=0.5350 |
| type010101 | normal | hybrid_micro_h15 | 20260908=0.6644; 20260909=0.6679; 20260910=0.4404 |
| type010101 | normal | hybrid_macro_h15 | 20260908=0.6644; 20260909=0.7925; 20260910=0.6682 |
| type010101 | novel | continuous_h1 | 20260908=0.1549; 20260909=0.0504; 20260910=0.0504 |
| type010101 | novel | continuous_h5 | 20260908=0.0504; 20260909=0.0504; 20260910=0.5135 |
| type010101 | novel | continuous_h15 | 20260908=0.4464; 20260909=0.4464; 20260910=0.4464 |
| type010101 | novel | hybrid_continuous_h1 | 20260908=0.0504; 20260909=0.0504; 20260910=0.0504 |
| type010101 | novel | hybrid_micro_h1 | 20260908=0.0504; 20260909=0.0504; 20260910=0.0504 |
| type010101 | novel | hybrid_macro_h1 | 20260908=0.1539; 20260909=0.0504; 20260910=0.0504 |
| type010101 | novel | hybrid_continuous_h5 | 20260908=0.0841; 20260909=0.4464; 20260910=0.0555 |
| type010101 | novel | hybrid_micro_h5 | 20260908=0.4147; 20260909=0.4464; 20260910=0.0504 |
| type010101 | novel | hybrid_macro_h5 | 20260908=0.1288; 20260909=0.0504; 20260910=0.0597 |
| type010101 | novel | hybrid_continuous_h15 | 20260908=0.5135; 20260909=0.4464; 20260910=0.1277 |
| type010101 | novel | hybrid_micro_h15 | 20260908=0.5358; 20260909=0.4464; 20260910=0.1159 |
| type010101 | novel | hybrid_macro_h15 | 20260908=0.5135; 20260909=0.4464; 20260910=0.5135 |
| type010102 | normal | continuous_h1 | 20260908=0.3904; 20260909=0.3904; 20260910=0.3904 |
| type010102 | normal | continuous_h5 | 20260908=0.5571; 20260909=0.3904; 20260910=0.3904 |
| type010102 | normal | continuous_h15 | 20260908=0.8609; 20260909=0.3904; 20260910=0.8609 |
| type010102 | normal | hybrid_continuous_h1 | 20260908=0.3904; 20260909=0.7904; 20260910=0.4385 |
| type010102 | normal | hybrid_micro_h1 | 20260908=0.3904; 20260909=0.2085; 20260910=0.3904 |
| type010102 | normal | hybrid_macro_h1 | 20260908=0.3904; 20260909=0.3904; 20260910=0.3904 |
| type010102 | normal | hybrid_continuous_h5 | 20260908=0.3904; 20260909=0.4945; 20260910=0.2272 |
| type010102 | normal | hybrid_micro_h5 | 20260908=0.5937; 20260909=0.7584; 20260910=0.7571 |
| type010102 | normal | hybrid_macro_h5 | 20260908=0.3904; 20260909=0.5184; 20260910=0.3904 |
| type010102 | normal | hybrid_continuous_h15 | 20260908=0.8609; 20260909=0.3904; 20260910=0.9108 |
| type010102 | normal | hybrid_micro_h15 | 20260908=0.8609; 20260909=0.3569; 20260910=0.2015 |
| type010102 | normal | hybrid_macro_h15 | 20260908=0.8609; 20260909=0.3904; 20260910=0.2668 |
| type010102 | novel | continuous_h1 | 20260908=0.2047; 20260909=0.2047; 20260910=0.2047 |
| type010102 | novel | continuous_h5 | 20260908=0.2047; 20260909=0.2047; 20260910=0.7597 |
| type010102 | novel | continuous_h15 | 20260908=0.8158; 20260909=0.7511; 20260910=0.9593 |
| type010102 | novel | hybrid_continuous_h1 | 20260908=0.6078; 20260909=0.3506; 20260910=0.2047 |
| type010102 | novel | hybrid_micro_h1 | 20260908=0.2047; 20260909=0.3564; 20260910=0.3926 |
| type010102 | novel | hybrid_macro_h1 | 20260908=0.7047; 20260909=0.2047; 20260910=0.2047 |
| type010102 | novel | hybrid_continuous_h5 | 20260908=0.8325; 20260909=0.2047; 20260910=0.4582 |
| type010102 | novel | hybrid_micro_h5 | 20260908=0.5992; 20260909=0.6640; 20260910=0.0850 |
| type010102 | novel | hybrid_macro_h5 | 20260908=0.3601; 20260909=0.2082; 20260910=0.3601 |
| type010102 | novel | hybrid_continuous_h15 | 20260908=0.8221; 20260909=0.3601; 20260910=0.8080 |
| type010102 | novel | hybrid_micro_h15 | 20260908=0.8158; 20260909=0.3601; 20260910=0.7660 |
| type010102 | novel | hybrid_macro_h15 | 20260908=0.8221; 20260909=0.3601; 20260910=0.7317 |

## Cross-side change in the hybrid-minus-continuous effect

Positive effect = reference regret minus tested regret (better hybrid). Change = novel-side effect minus normal-side effect; negative means the hybrid advantage shrinks on the novelty. Two-sample bootstrap over each side's independent states; descriptive only.

| Pair | Condition | Tested | Reference | Novel effect | Normal effect | Change [95%] |
| --- | --- | --- | --- | --- | --- | --- |
| type010101 | zero-shot | hybrid_continuous_h1 | continuous_h1 | +0.1065 | +0.0404 | +0.0662 [-0.1008, +0.2288] |
| type010101 | zero-shot | hybrid_continuous_h15 | continuous_h15 | -0.1824 | +0.0104 | -0.1928 [-0.4525, +0.0845] |
| type010101 | zero-shot | hybrid_macro_h15 | hybrid_continuous_h15 | +0.0075 | +0.1263 | -0.1189 [-0.2569, +0.0199] |
| type010101 | few-shot | hybrid_continuous_h1 | continuous_h1 | +0.0348 | +0.0824 | -0.0476 [-0.1489, +0.0384] |
| type010101 | few-shot | hybrid_continuous_h15 | continuous_h15 | +0.0839 | +0.0125 | +0.0714 [-0.1083, +0.2579] |
| type010101 | few-shot | hybrid_macro_h15 | hybrid_continuous_h15 | -0.1286 | -0.0438 | -0.0848 [-0.2938, +0.1262] |
| type010102 | zero-shot | hybrid_continuous_h1 | continuous_h1 | +0.0384 | -0.0560 | +0.0943 [-0.0486, +0.2373] |
| type010102 | zero-shot | hybrid_continuous_h15 | continuous_h15 | -0.0098 | +0.2595 | -0.2693 [-0.5412, -0.0050] |
| type010102 | zero-shot | hybrid_macro_h15 | hybrid_continuous_h15 | +0.2664 | -0.1578 | +0.4242 [+0.0915, +0.7657] |
| type010102 | few-shot | hybrid_continuous_h1 | continuous_h1 | -0.1830 | -0.1494 | -0.0337 [-0.2021, +0.1503] |
| type010102 | few-shot | hybrid_continuous_h15 | continuous_h15 | +0.1787 | -0.0166 | +0.1953 [+0.0583, +0.3363] |
| type010102 | few-shot | hybrid_macro_h15 | hybrid_continuous_h15 | +0.0254 | +0.2147 | -0.1892 [-0.2720, -0.0945] |

## Adaptation change on the novel side (few-shot minus zero-shot)

| Pair | System | Regret change [95%] (negative favors adaptation) |
| --- | --- | --- |
| type010101 | continuous_h1 | -0.0866 [-0.2422, +0.0690] |
| type010101 | continuous_h5 | +0.1544 [+0.0062, +0.3025] |
| type010101 | continuous_h15 | +0.1080 [+0.0339, +0.1820] |
| type010101 | hybrid_continuous_h1 | -0.0149 [-0.0303, +0.0005] |
| type010101 | hybrid_micro_h1 | -0.0956 [-0.1912, +0.0000] |
| type010101 | hybrid_macro_h1 | +0.0345 [+0.0000, +0.0690] |
| type010101 | hybrid_continuous_h5 | -0.0063 [-0.0570, +0.0444] |
| type010101 | hybrid_micro_h5 | +0.2429 [+0.0000, +0.4858] |
| type010101 | hybrid_macro_h5 | -0.0754 [-0.1739, +0.0231] |
| type010101 | hybrid_continuous_h15 | -0.1584 [-0.3246, +0.0078] |
| type010101 | hybrid_micro_h15 | -0.1549 [-0.3246, +0.0149] |
| type010101 | hybrid_macro_h15 | -0.0223 [-0.0596, +0.0149] |
| type010102 | continuous_h1 | -0.1970 [-0.2688, -0.1253] |
| type010102 | continuous_h5 | +0.1850 [+0.1012, +0.2688] |
| type010102 | continuous_h15 | +0.0672 [+0.0432, +0.0912] |
| type010102 | hybrid_continuous_h1 | +0.0244 [-0.0486, +0.0973] |
| type010102 | hybrid_micro_h1 | +0.0287 [-0.1690, +0.2265] |
| type010102 | hybrid_macro_h1 | +0.1655 [+0.0000, +0.3310] |
| type010102 | hybrid_continuous_h5 | +0.1442 [+0.1012, +0.1872] |
| type010102 | hybrid_micro_h5 | +0.2447 [-0.0798, +0.5692] |
| type010102 | hybrid_macro_h5 | +0.0920 [+0.0071, +0.1768] |
| type010102 | hybrid_continuous_h15 | -0.1213 [-0.2384, -0.0042] |
| type010102 | hybrid_micro_h15 | -0.0028 [-0.0084, +0.0028] |
| type010102 | hybrid_macro_h15 | +0.1197 [-0.0042, +0.2436] |

## Headroom (zero-shot condition targets)

| Pair | Side | State | Candidates | Informative | Prior(ordinal09) regret | Uniform regret | Dropped |
| --- | --- | --- | --- | --- | --- | --- | --- |
| type010101 | normal | issue-76-bounded-transfer-032 | 13 | True | 0.9081 | 0.7310 | 0 |
| type010101 | normal | issue-76-bounded-transfer-033 | 13 | True | 0.2595 | 0.4690 | 0 |
| type010101 | normal | issue-76-bounded-transfer-034 | 13 | True | 0.4108 | 0.5077 | 0 |
| type010101 | normal | issue-76-bounded-transfer-035 | 13 | True | 0.9604 | 0.8484 | 0 |
| type010101 | normal | issue-76-bounded-transfer-036 | 13 | True | 0.5888 | 0.6662 | 0 |
| type010101 | normal | issue-76-bounded-transfer-037 | 13 | True | 0.8694 | 0.5905 | 0 |
| type010101 | normal | issue-76-bounded-transfer-038 | 13 | True | 0.9151 | 0.7284 | 0 |
| type010101 | normal | issue-76-bounded-transfer-039 | 13 | True | 0.7292 | 0.6169 | 0 |
| type010101 | novel | issue-77-n2-007 | 13 | True | 0.0946 | 0.2812 | 0 |
| type010101 | novel | issue-77-n2-008 | 13 | True | 0.2152 | 0.1279 | 0 |
| type010102 | normal | issue-77-n2n-001 | 6 | True | 0.8357 | 0.7247 | 7 |
| type010102 | normal | issue-77-n2n-002 | 4 | True | 0.0358 | 0.2659 | 9 |
| type010102 | normal | issue-77-n2n-005 | 11 | True | 0.2080 | 0.4765 | 2 |
| type010102 | normal | issue-77-n2n-007 | 4 | True | 0.0000 | 0.6508 | 9 |
| type010102 | normal | issue-77-n2n-008 | 9 | True | 0.4732 | 0.4996 | 4 |
| type010102 | novel | issue-77-n2-014 | 12 | True | 0.7128 | 0.6362 | 1 |
| type010102 | novel | issue-77-n2-016 | 13 | True | 0.8063 | 0.7069 | 0 |

## Limitations

- shared semantically supervised CNN, not end-to-end symbol-free comparison
- t=600 is end-of-window cost on mostly right-censored branches, not settled cost
- two independent held-out novel lineages per family; intervals descriptive only
- R3 normal-side lineages retain #75 development model-selection scoring optimism
- typed branch failures excluded from candidates and reported, never worst-cased
- controllers not adapted; fixed systems only
- linear MACs not full FLOPs or matched total work
