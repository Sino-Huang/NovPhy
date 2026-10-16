# Issue #107: granularity under novelty (request selection on the #110 contract-v2 cells)

Frozen 2026-10-16T02:47:55Z. One score. Intervals: member (evaluation level)-clustered bootstrap, 10 000 draws, DESCRIPTIVE. Margin delta = 0.02. Zero-shot and few-shot are separate sections; no estimate pools their rows. Zero engine seconds.

## Disposition: 'request selection is a cheap adaptation surface under novelty'

| policy | role | policy-only retune - zero-shot | token | policy-only - predictor-only (non-inferiority) | token | policy retune cheaper | disposition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| J_dec | primary | -0.007 [-0.021, 0.006] | not_supported_by_this_experiment | -0.158 [-0.214, -0.104] | not_supported_by_this_experiment | yes | **not_supported_by_this_experiment** |
| J | secondary | -0.057 [-0.079, -0.034] | not_supported_by_this_experiment | -0.170 [-0.233, -0.110] | not_supported_by_this_experiment | yes | **not_supported_by_this_experiment** |

Rule: 'request selection is a cheap adaptation surface under novelty', J_dec primary (J secondary, same rule), on the pooled novel population (every mixed evaluation level of the in-scope cells of novelty levels [1, 2, 3, 4, 5, 6, 7], x 3 seeds, member = evaluation level): (a) gain = AUC(J_dec policy-only retune) - AUC(J_dec zero-shot), delta rule (supported iff mean >= 0.02 and lower > 0; not supported iff upper < 0.02); (b) parity = AUC(J_dec policy-only) - AUC(J_dec predictor-only), non-inferiority (supported iff lower > -0.02; not supported iff upper < -0.02); (c) cheap = the summed GPU seconds of the policy-only retunes (levels x seeds) are below the summed #110 few-shot hybrid predictor member training GPU seconds of the same levels and seeds. supported iff (a) and (b) are supported and (c) holds; not_supported_by_this_experiment iff (a) or (b) is not supported or (c) fails; otherwise readiness_or_precision_insufficient. A failed guard turns every token into readiness_or_precision_insufficient

Guards: G1_isolation pass, G2_scoring_policy pass, G3_chronology pass, G4_checkpoint_binding pass, G5_prefix_parse_replication pass, G6_handoff_inventories pass.

## Scope (45 #110-supported cells)

In scope: 29 cells with mixed-verdict evaluation levels. Out of scope (16 cells, token `readiness_or_precision_insufficient`):

| cell | reason |
| --- | --- |
| 0/multiple_forces | rollout_error_only: degenerate single-shot pig-removal verdicts (multi-hit pigs) |
| 1/multiple_forces | rollout_error_only: degenerate single-shot pig-removal verdicts (multi-hit pigs) |
| 2/multiple_forces | rollout_error_only: degenerate single-shot pig-removal verdicts (multi-hit pigs) |
| 3/single_force | no_mixed_verdict_evaluation_level: no decision headroom on the evaluation grid |
| 3/multiple_forces | rollout_error_only: degenerate single-shot pig-removal verdicts (multi-hit pigs) |
| 3/sliding | no_mixed_verdict_evaluation_level: no decision headroom on the evaluation grid |
| 4/multiple_forces | rollout_error_only: degenerate single-shot pig-removal verdicts (multi-hit pigs) |
| 4/sliding | no_mixed_verdict_evaluation_level: no decision headroom on the evaluation grid |
| 5/multiple_forces | rollout_error_only: degenerate single-shot pig-removal verdicts (multi-hit pigs) |
| 6/multiple_forces | rollout_error_only: degenerate single-shot pig-removal verdicts (multi-hit pigs) |
| 7/multiple_forces | rollout_error_only: degenerate single-shot pig-removal verdicts (multi-hit pigs) |
| 8/single_force | no_mixed_verdict_evaluation_level: no decision headroom on the evaluation grid |
| 8/multiple_forces | rollout_error_only: degenerate single-shot pig-removal verdicts (multi-hit pigs) |
| 8/rolling | no_mixed_verdict_evaluation_level: no decision headroom on the evaluation grid |
| 8/falling | no_mixed_verdict_evaluation_level: no decision headroom on the evaluation grid |
| 8/sliding | no_mixed_verdict_evaluation_level: no decision headroom on the evaluation grid |

## Zero-shot (normal-trained world and policies, frozen): ranking AUC per arm

| novelty type | levels (mixed) | J | J_dec | F-15-macro | F_hindsight | no-model prior |
| --- | --- | --- | --- | --- | --- | --- |
| N1 (#109 evaluation) | 311 | 0.719 [0.699, 0.739] | 0.724 [0.705, 0.742] | 0.715 [0.696, 0.734] | F-15-micro: 0.725 [0.706, 0.743] | 0.705 [0.682, 0.727] |
| 0: normal (paired normal sides) | 135 | 0.632 [0.594, 0.669] | 0.648 [0.610, 0.684] | 0.650 [0.612, 0.687] | F-15-micro: 0.664 [0.628, 0.699] | 0.761 [0.726, 0.795] |
| 1: appearance (pink pig/blocks) | 20 | 0.656 [0.545, 0.757] | 0.642 [0.533, 0.740] | 0.627 [0.516, 0.728] | F-15-micro: 0.646 [0.535, 0.746] | 0.654 [0.565, 0.742] |
| 2: fan (bird push) | 17 | 0.514 [0.420, 0.601] | 0.529 [0.421, 0.630] | 0.533 [0.430, 0.628] | F-15-continuous: 0.542 [0.435, 0.641] | 0.741 [0.610, 0.849] |
| 3: stronger air turbulence | 6 | 0.657 [0.508, 0.798] | 0.620 [0.517, 0.730] | 0.605 [0.491, 0.726] | F-1-continuous: 0.667 [0.452, 0.877] | 0.718 [0.659, 0.778] |
| 4: magnet | 18 | 0.607 [0.534, 0.673] | 0.595 [0.518, 0.670] | 0.606 [0.532, 0.675] | F-1-macro: 0.617 [0.548, 0.686] | 0.553 [0.487, 0.617] |
| 5: right slingshot (mirrored layout) | 17 | 0.602 [0.483, 0.711] | 0.597 [0.471, 0.714] | 0.602 [0.491, 0.706] | F-1-macro: 0.662 [0.567, 0.747] | 0.857 [0.758, 0.941] |
| 6: inverse gravity | 22 | 0.378 [0.278, 0.480] | 0.333 [0.242, 0.430] | 0.356 [0.263, 0.454] | F-15-macro: 0.356 [0.263, 0.454] | 0.210 [0.125, 0.300] |
| 7: inverse air turbulence | 22 | 0.545 [0.412, 0.670] | 0.558 [0.406, 0.701] | 0.596 [0.438, 0.748] | F-15-micro: 0.618 [0.460, 0.768] | 0.635 [0.508, 0.759] |
| novel 1-7 pooled | 122 | 0.551 [0.504, 0.596] | 0.541 [0.491, 0.590] | 0.552 [0.502, 0.599] | F-15-micro: 0.552 [0.500, 0.602] | 0.599 [0.546, 0.650] |

Nine fixed requests, zero-shot world (AUC):

| novelty type | F-1-continuous | F-1-micro | F-1-macro | F-5-continuous | F-5-micro | F-5-macro | F-15-continuous | F-15-micro | F-15-macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N1 (#109 evaluation) | 0.691 [0.669, 0.713] | 0.684 [0.663, 0.706] | 0.701 [0.679, 0.723] | 0.720 [0.701, 0.739] | 0.721 [0.702, 0.740] | 0.715 [0.696, 0.735] | 0.724 [0.705, 0.743] | 0.725 [0.706, 0.743] | 0.715 [0.696, 0.734] |
| 0: normal (paired normal sides) | 0.624 [0.583, 0.665] | 0.625 [0.586, 0.663] | 0.635 [0.595, 0.675] | 0.656 [0.618, 0.694] | 0.662 [0.625, 0.698] | 0.650 [0.612, 0.688] | 0.660 [0.623, 0.696] | 0.664 [0.628, 0.699] | 0.650 [0.612, 0.687] |
| 1: appearance (pink pig/blocks) | 0.610 [0.491, 0.722] | 0.610 [0.492, 0.720] | 0.615 [0.493, 0.728] | 0.636 [0.526, 0.735] | 0.637 [0.527, 0.737] | 0.625 [0.514, 0.726] | 0.644 [0.534, 0.743] | 0.646 [0.535, 0.746] | 0.627 [0.516, 0.728] |
| 2: fan (bird push) | 0.452 [0.363, 0.541] | 0.458 [0.369, 0.545] | 0.469 [0.378, 0.558] | 0.521 [0.420, 0.618] | 0.527 [0.425, 0.624] | 0.523 [0.424, 0.617] | 0.542 [0.435, 0.641] | 0.539 [0.429, 0.642] | 0.533 [0.430, 0.628] |
| 3: stronger air turbulence | 0.667 [0.452, 0.877] | 0.653 [0.446, 0.860] | 0.652 [0.448, 0.852] | 0.619 [0.482, 0.756] | 0.628 [0.506, 0.750] | 0.626 [0.485, 0.767] | 0.594 [0.487, 0.707] | 0.611 [0.503, 0.729] | 0.605 [0.491, 0.726] |
| 4: magnet | 0.616 [0.542, 0.691] | 0.606 [0.536, 0.675] | 0.617 [0.548, 0.686] | 0.614 [0.538, 0.683] | 0.611 [0.535, 0.682] | 0.600 [0.530, 0.663] | 0.607 [0.531, 0.676] | 0.611 [0.534, 0.684] | 0.606 [0.532, 0.675] |
| 5: right slingshot (mirrored layout) | 0.643 [0.542, 0.735] | 0.628 [0.533, 0.714] | 0.662 [0.567, 0.747] | 0.615 [0.491, 0.728] | 0.607 [0.484, 0.719] | 0.622 [0.510, 0.725] | 0.608 [0.483, 0.724] | 0.606 [0.481, 0.722] | 0.602 [0.491, 0.706] |
| 6: inverse gravity | 0.323 [0.233, 0.420] | 0.332 [0.254, 0.416] | 0.313 [0.220, 0.411] | 0.297 [0.202, 0.399] | 0.305 [0.210, 0.406] | 0.349 [0.251, 0.452] | 0.296 [0.203, 0.397] | 0.304 [0.208, 0.406] | 0.356 [0.263, 0.454] |
| 7: inverse air turbulence | 0.558 [0.425, 0.683] | 0.572 [0.449, 0.689] | 0.569 [0.441, 0.692] | 0.588 [0.426, 0.744] | 0.604 [0.448, 0.754] | 0.582 [0.427, 0.730] | 0.595 [0.434, 0.750] | 0.618 [0.460, 0.768] | 0.596 [0.438, 0.748] |
| novel 1-7 pooled | 0.535 [0.488, 0.583] | 0.536 [0.491, 0.581] | 0.540 [0.494, 0.588] | 0.543 [0.491, 0.594] | 0.548 [0.496, 0.597] | 0.549 [0.500, 0.597] | 0.545 [0.493, 0.596] | 0.552 [0.500, 0.602] | 0.552 [0.502, 0.599] |

Zero-shot contrasts (paired by level and seed):

| novelty type | J_dec - F_hindsight | J_dec - F-15-macro | J_dec - prior | J - F_hindsight | J - F-15-macro | J - prior | J_dec - J |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N1 (#109 evaluation) | -0.001 [-0.003, 0.001] N | 0.009 [0.006, 0.012] N | 0.019 [-0.007, 0.045] I | -0.005 [-0.013, 0.002] N | 0.004 [-0.002, 0.011] N | 0.014 [-0.014, 0.043] I | 0.004 [-0.003, 0.012] N |
| 0: normal (paired normal sides) | -0.017 [-0.026, -0.008] N | -0.002 [-0.009, 0.005] N | -0.113 [-0.159, -0.067] N | -0.033 [-0.049, -0.017] N | -0.018 [-0.034, -0.003] N | -0.129 [-0.176, -0.083] N | 0.016 [0.000, 0.032] I |
| 1: appearance (pink pig/blocks) | -0.004 [-0.008, -0.001] N | 0.016 [0.002, 0.032] I | -0.012 [-0.159, 0.123] I | 0.009 [-0.023, 0.043] I | 0.029 [0.006, 0.054] S | 0.002 [-0.148, 0.140] I | -0.013 [-0.045, 0.017] N |
| 2: fan (bird push) | -0.013 [-0.028, -0.001] N | -0.004 [-0.018, 0.008] N | -0.212 [-0.314, -0.100] N | -0.028 [-0.061, 0.008] N | -0.019 [-0.044, 0.008] N | -0.227 [-0.330, -0.117] N | 0.014 [-0.019, 0.043] I |
| 3: stronger air turbulence | -0.046 [-0.176, 0.100] I | 0.015 [0.000, 0.044] I | -0.098 [-0.241, 0.030] I | -0.010 [-0.100, 0.081] I | 0.051 [0.007, 0.088] S | -0.062 [-0.244, 0.116] I | -0.036 [-0.088, 0.037] I |
| 4: magnet | -0.021 [-0.072, 0.032] I | -0.011 [-0.034, 0.009] N | 0.042 [-0.088, 0.171] I | -0.009 [-0.049, 0.034] I | 0.001 [-0.027, 0.030] I | 0.054 [-0.063, 0.164] I | -0.012 [-0.053, 0.023] I |
| 5: right slingshot (mirrored layout) | -0.065 [-0.114, -0.021] N | -0.005 [-0.028, 0.020] N | -0.260 [-0.398, -0.115] N | -0.060 [-0.103, -0.023] N | -0.000 [-0.020, 0.019] N | -0.255 [-0.387, -0.111] N | -0.005 [-0.022, 0.012] N |
| 6: inverse gravity | -0.023 [-0.040, -0.005] N | -0.023 [-0.040, -0.005] N | 0.124 [0.071, 0.176] S | 0.022 [-0.008, 0.056] I | 0.022 [-0.008, 0.056] I | 0.168 [0.101, 0.234] S | -0.044 [-0.078, -0.014] N |
| 7: inverse air turbulence | -0.060 [-0.099, -0.027] N | -0.039 [-0.070, -0.013] N | -0.078 [-0.215, 0.074] I | -0.072 [-0.112, -0.033] N | -0.051 [-0.100, -0.003] N | -0.090 [-0.210, 0.045] I | 0.012 [-0.029, 0.054] I |
| novel 1-7 pooled | -0.011 [-0.022, -0.001] N | -0.011 [-0.020, -0.002] N | -0.058 [-0.111, -0.007] N | -0.001 [-0.016, 0.015] N | -0.000 [-0.014, 0.014] N | -0.048 [-0.101, 0.003] N | -0.010 [-0.024, 0.003] N |

## Few-shot (level-k novel fit sides): ranking AUC per arm

| novelty type | levels (mixed) | J policy-only | J_dec policy-only | J predictor-only | J_dec predictor-only | J both | J_dec both | F_hindsight (few-shot world) | few-shot prior |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1: appearance (pink pig/blocks) | 20 | 0.585 [0.477, 0.683] | 0.651 [0.540, 0.749] | 0.617 [0.540, 0.693] | 0.656 [0.547, 0.755] | 0.668 [0.564, 0.764] | 0.623 [0.538, 0.701] | F-1-continuous: 0.670 [0.565, 0.765] | 0.672 [0.581, 0.762] |
| 2: fan (bird push) | 17 | 0.514 [0.416, 0.608] | 0.547 [0.443, 0.644] | 0.547 [0.475, 0.621] | 0.559 [0.454, 0.656] | 0.576 [0.470, 0.674] | 0.569 [0.470, 0.660] | F-1-continuous: 0.581 [0.470, 0.684] | 0.798 [0.656, 0.923] |
| 3: stronger air turbulence | 6 | 0.640 [0.544, 0.740] | 0.629 [0.511, 0.748] | 0.459 [0.171, 0.748] | 0.472 [0.183, 0.763] | 0.562 [0.299, 0.833] | 0.559 [0.296, 0.826] | F-1-continuous: 0.512 [0.256, 0.778] | 0.819 [0.758, 0.894] |
| 4: magnet | 18 | 0.582 [0.506, 0.659] | 0.558 [0.479, 0.639] | 0.596 [0.517, 0.669] | 0.636 [0.532, 0.733] | 0.623 [0.517, 0.726] | 0.618 [0.519, 0.709] | F-15-macro: 0.640 [0.542, 0.731] | 0.588 [0.471, 0.706] |
| 5: right slingshot (mirrored layout) | 17 | 0.558 [0.459, 0.655] | 0.615 [0.499, 0.723] | 0.634 [0.578, 0.698] | 0.682 [0.609, 0.752] | 0.725 [0.665, 0.785] | 0.689 [0.614, 0.761] | F-1-macro: 0.755 [0.693, 0.815] | 0.589 [0.479, 0.689] |
| 6: inverse gravity | 22 | 0.258 [0.181, 0.341] | 0.317 [0.231, 0.409] | 0.820 [0.760, 0.874] | 0.790 [0.730, 0.846] | 0.743 [0.680, 0.802] | 0.793 [0.731, 0.849] | F-1-macro: 0.823 [0.764, 0.878] | 0.790 [0.737, 0.840] |
| 7: inverse air turbulence | 22 | 0.473 [0.350, 0.590] | 0.525 [0.391, 0.653] | 0.782 [0.716, 0.843] | 0.842 [0.759, 0.914] | 0.787 [0.714, 0.854] | 0.813 [0.729, 0.887] | F-1-micro: 0.858 [0.790, 0.918] | 0.880 [0.831, 0.925] |
| novel 1-7 pooled | 122 | 0.495 [0.449, 0.539] | 0.534 [0.486, 0.580] | 0.665 [0.628, 0.700] | 0.692 [0.650, 0.732] | 0.686 [0.648, 0.724] | 0.686 [0.647, 0.723] | F-1-macro: 0.714 [0.675, 0.752] | 0.732 [0.691, 0.772] |

Nine fixed requests, few-shot world (AUC):

| novelty type | F-1-continuous | F-1-micro | F-1-macro | F-5-continuous | F-5-micro | F-5-macro | F-15-continuous | F-15-micro | F-15-macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1: appearance (pink pig/blocks) | 0.670 [0.565, 0.765] | 0.664 [0.569, 0.751] | 0.664 [0.553, 0.764] | 0.663 [0.558, 0.757] | 0.657 [0.561, 0.745] | 0.650 [0.540, 0.751] | 0.665 [0.562, 0.759] | 0.661 [0.567, 0.749] | 0.649 [0.541, 0.749] |
| 2: fan (bird push) | 0.581 [0.470, 0.684] | 0.580 [0.463, 0.687] | 0.579 [0.474, 0.680] | 0.568 [0.457, 0.670] | 0.560 [0.446, 0.664] | 0.575 [0.478, 0.667] | 0.579 [0.475, 0.674] | 0.566 [0.459, 0.664] | 0.569 [0.480, 0.657] |
| 3: stronger air turbulence | 0.512 [0.256, 0.778] | 0.495 [0.242, 0.752] | 0.506 [0.228, 0.793] | 0.497 [0.194, 0.801] | 0.469 [0.169, 0.769] | 0.490 [0.190, 0.807] | 0.484 [0.184, 0.784] | 0.465 [0.165, 0.765] | 0.490 [0.190, 0.807] |
| 4: magnet | 0.607 [0.536, 0.682] | 0.610 [0.542, 0.677] | 0.623 [0.566, 0.683] | 0.636 [0.535, 0.732] | 0.630 [0.532, 0.719] | 0.635 [0.543, 0.719] | 0.640 [0.539, 0.735] | 0.616 [0.521, 0.704] | 0.640 [0.542, 0.731] |
| 5: right slingshot (mirrored layout) | 0.741 [0.684, 0.797] | 0.731 [0.674, 0.790] | 0.755 [0.693, 0.815] | 0.720 [0.650, 0.789] | 0.717 [0.642, 0.789] | 0.751 [0.695, 0.806] | 0.680 [0.605, 0.752] | 0.682 [0.604, 0.759] | 0.718 [0.649, 0.783] |
| 6: inverse gravity | 0.805 [0.746, 0.857] | 0.822 [0.763, 0.876] | 0.823 [0.764, 0.878] | 0.802 [0.741, 0.858] | 0.814 [0.756, 0.866] | 0.820 [0.760, 0.875] | 0.795 [0.729, 0.854] | 0.810 [0.747, 0.867] | 0.805 [0.746, 0.860] |
| 7: inverse air turbulence | 0.855 [0.786, 0.917] | 0.858 [0.790, 0.918] | 0.855 [0.782, 0.920] | 0.821 [0.741, 0.892] | 0.825 [0.736, 0.901] | 0.838 [0.760, 0.907] | 0.838 [0.755, 0.911] | 0.849 [0.769, 0.920] | 0.851 [0.772, 0.920] |
| novel 1-7 pooled | 0.708 [0.670, 0.745] | 0.709 [0.670, 0.746] | 0.714 [0.675, 0.752] | 0.699 [0.658, 0.739] | 0.697 [0.655, 0.737] | 0.708 [0.667, 0.748] | 0.697 [0.656, 0.737] | 0.695 [0.654, 0.735] | 0.703 [0.662, 0.743] |

Few-shot contrasts, J_dec (paired by level and seed; `policy - predictor` uses the non-inferiority rule):

J_dec:

| novelty type | policy - zero-shot | policy - predictor | predictor - zero-shot | both - predictor | both - zero-shot | policy - F_hindsight (zero-shot world) | predictor - F_hindsight (few-shot world) | both - F_hindsight (few-shot world) | policy - few-shot prior | both - few-shot prior |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1: appearance (pink pig/blocks) | 0.008 [0.001, 0.019] N | -0.005 [-0.051, 0.043] I | 0.014 [-0.038, 0.066] I | -0.033 [-0.076, 0.007] N | -0.019 [-0.073, 0.037] I | 0.004 [-0.003, 0.013] N | -0.014 [-0.029, -0.001] N | -0.047 [-0.093, -0.004] N | -0.022 [-0.197, 0.136] I | -0.050 [-0.200, 0.093] I |
| 2: fan (bird push) | 0.018 [0.003, 0.038] I | -0.012 [-0.058, 0.034] I | 0.030 [-0.016, 0.074] I | 0.011 [-0.025, 0.047] I | 0.041 [-0.007, 0.086] I | 0.005 [-0.009, 0.020] I | -0.022 [-0.055, 0.012] N | -0.011 [-0.045, 0.024] I | -0.251 [-0.393, -0.086] N | -0.228 [-0.370, -0.075] N |
| 3: stronger air turbulence | 0.009 [-0.058, 0.081] I | 0.156 [-0.093, 0.407] I | -0.148 [-0.377, 0.089] I | 0.087 [-0.054, 0.227] I | -0.061 [-0.284, 0.156] I | -0.038 [-0.172, 0.096] I | -0.040 [-0.165, 0.052] I | 0.047 [-0.053, 0.152] I | -0.191 [-0.342, -0.029] N | -0.260 [-0.573, 0.060] I |
| 4: magnet | -0.037 [-0.082, 0.011] N | -0.078 [-0.159, 0.016] I | 0.041 [-0.046, 0.117] I | -0.019 [-0.047, 0.003] N | 0.022 [-0.068, 0.104] I | -0.058 [-0.128, 0.011] N | -0.004 [-0.027, 0.016] N | -0.023 [-0.052, 0.002] N | -0.030 [-0.150, 0.083] I | 0.029 [-0.086, 0.144] I |
| 5: right slingshot (mirrored layout) | 0.018 [0.004, 0.041] I | -0.067 [-0.161, 0.018] I | 0.085 [-0.011, 0.198] I | 0.008 [-0.034, 0.050] I | 0.093 [-0.023, 0.232] I | -0.047 [-0.091, -0.008] N | -0.073 [-0.129, -0.022] N | -0.066 [-0.112, -0.019] N | 0.026 [-0.150, 0.207] I | 0.101 [-0.018, 0.226] I |
| 6: inverse gravity | -0.016 [-0.033, 0.002] N | -0.473 [-0.594, -0.344] N | 0.457 [0.325, 0.581] S | 0.003 [-0.015, 0.019] N | 0.460 [0.329, 0.583] S | -0.038 [-0.063, -0.014] N | -0.033 [-0.054, -0.015] N | -0.030 [-0.057, -0.006] N | -0.473 [-0.577, -0.357] N | 0.003 [-0.043, 0.047] I |
| 7: inverse air turbulence | -0.033 [-0.086, 0.022] I | -0.317 [-0.466, -0.178] N | 0.284 [0.141, 0.445] S | -0.029 [-0.052, -0.009] N | 0.255 [0.109, 0.417] S | -0.093 [-0.143, -0.045] N | -0.017 [-0.036, 0.000] N | -0.045 [-0.075, -0.018] N | -0.355 [-0.484, -0.235] N | -0.067 [-0.117, -0.021] N |
| novel 1-7 pooled | -0.007 [-0.021, 0.006] N | -0.158 [-0.214, -0.104] N | 0.151 [0.097, 0.206] S | -0.006 [-0.020, 0.009] N | 0.145 [0.091, 0.202] S | -0.018 [-0.033, -0.004] N | -0.022 [-0.038, -0.008] N | -0.028 [-0.046, -0.011] N | -0.198 [-0.263, -0.131] N | -0.046 [-0.093, 0.003] N |

J:

| novelty type | policy - zero-shot | policy - predictor | predictor - zero-shot | both - predictor | both - zero-shot | policy - F_hindsight (zero-shot world) | predictor - F_hindsight (few-shot world) | both - F_hindsight (few-shot world) | policy - few-shot prior | both - few-shot prior |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1: appearance (pink pig/blocks) | -0.071 [-0.113, -0.032] N | -0.032 [-0.143, 0.076] I | -0.039 [-0.137, 0.067] I | 0.051 [-0.008, 0.109] I | 0.012 [-0.055, 0.087] I | -0.062 [-0.110, -0.018] N | -0.053 [-0.113, 0.008] N | -0.002 [-0.016, 0.011] N | -0.088 [-0.266, 0.081] I | -0.004 [-0.169, 0.146] I |
| 2: fan (bird push) | 0.000 [-0.039, 0.039] I | -0.033 [-0.113, 0.045] I | 0.033 [-0.042, 0.111] I | 0.029 [-0.038, 0.094] I | 0.062 [0.008, 0.113] S | -0.028 [-0.049, -0.004] N | -0.033 [-0.105, 0.045] I | -0.004 [-0.025, 0.016] N | -0.283 [-0.414, -0.132] N | -0.221 [-0.358, -0.073] N |
| 3: stronger air turbulence | -0.016 [-0.142, 0.115] I | 0.181 [-0.126, 0.480] I | -0.197 [-0.424, 0.033] I | 0.103 [0.003, 0.222] S | -0.094 [-0.342, 0.144] I | -0.026 [-0.200, 0.159] I | -0.053 [-0.190, 0.058] I | 0.050 [0.034, 0.069] S | -0.179 [-0.291, -0.091] N | -0.257 [-0.583, 0.067] I |
| 4: magnet | -0.025 [-0.077, 0.027] I | -0.013 [-0.103, 0.084] I | -0.012 [-0.105, 0.067] I | 0.028 [-0.011, 0.071] I | 0.016 [-0.093, 0.107] I | -0.034 [-0.065, -0.005] N | -0.045 [-0.089, -0.000] N | -0.017 [-0.058, 0.016] N | -0.006 [-0.141, 0.121] I | 0.035 [-0.083, 0.152] I |
| 5: right slingshot (mirrored layout) | -0.043 [-0.099, 0.013] N | -0.076 [-0.176, 0.028] I | 0.032 [-0.079, 0.148] I | 0.091 [0.030, 0.154] S | 0.124 [0.037, 0.227] S | -0.104 [-0.160, -0.053] N | -0.121 [-0.196, -0.052] N | -0.030 [-0.075, 0.015] N | -0.031 [-0.189, 0.138] I | 0.136 [0.024, 0.262] S |
| 6: inverse gravity | -0.119 [-0.187, -0.055] N | -0.561 [-0.688, -0.425] N | 0.442 [0.300, 0.577] S | -0.076 [-0.114, -0.039] N | 0.365 [0.214, 0.505] S | -0.098 [-0.152, -0.045] N | -0.004 [-0.010, 0.003] N | -0.080 [-0.119, -0.042] N | -0.532 [-0.639, -0.413] N | -0.047 [-0.107, 0.009] N |
| 7: inverse air turbulence | -0.072 [-0.116, -0.028] N | -0.309 [-0.452, -0.178] N | 0.237 [0.102, 0.388] S | 0.005 [-0.041, 0.048] I | 0.242 [0.104, 0.396] S | -0.145 [-0.196, -0.093] N | -0.076 [-0.109, -0.046] N | -0.071 [-0.104, -0.039] N | -0.407 [-0.524, -0.300] N | -0.093 [-0.140, -0.047] N |
| novel 1-7 pooled | -0.057 [-0.079, -0.034] N | -0.170 [-0.233, -0.110] N | 0.114 [0.056, 0.174] S | 0.022 [-0.002, 0.044] I | 0.135 [0.084, 0.189] S | -0.057 [-0.079, -0.035] N | -0.049 [-0.070, -0.027] N | -0.028 [-0.045, -0.011] N | -0.237 [-0.303, -0.170] N | -0.045 [-0.095, 0.005] N |

Token marks: S supported, N not_supported_by_this_experiment, I readiness_or_precision_insufficient (DESCRIPTIVE per novelty type; the disposition uses the pooled novel row only).

## Request shares (% of rollout steps) and the shift toward/away from the paired normal operating point

Reference: the same kind's normal policy on the paired normal side (same pair and ordinal), zero-shot world. Shares use every evaluation level of the in-scope cells (outcome-free).

| population | arm | 1-continuous | 1-micro | 1-macro | 5-continuous | 5-micro | 5-macro | 15-continuous | 15-micro | 15-macro | TV shift vs paired normal | moves toward | moves away |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| #101 N1 (E99 Gate-B predictor) | J | 12 | 2 | 79 | 0 | 0 | 0 | 0 | 5 | 1 | | | |
| N1 (#109 evaluation) | zs:J | 64 | 2 | 2 | 1 | 2 | 0 | 3 | 26 | 0 | | | |
| 0: normal (paired normal sides) | zs:J | 47 | 6 | 7 | 1 | 1 | 0 | 4 | 34 | 0 | | | |
| #101 N1 (E99 Gate-B predictor) | J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 52 | 10 | 38 | | | |
| N1 (#109 evaluation) | zs:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 76 | 1 | 22 | | | |
| 0: normal (paired normal sides) | zs:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 69 | 2 | 29 | | | |
| 1: appearance (pink pig/blocks) | zs:J | 67 | 2 | 2 | 1 | 2 | 0 | 3 | 23 | 0 | 0.09 [0.05, 0.15] | 1-continuous, 5-continuous, 5-micro | 1-micro, 15-micro |
| 1: appearance (pink pig/blocks) | zs:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 76 | 2 | 22 | 0.08 [0.04, 0.12] | - | - |
| 1: appearance (pink pig/blocks) | policy:J | 35 | 3 | 31 | 1 | 10 | 7 | 5 | 5 | 4 | 0.62 [0.59, 0.64] | 1-macro, 5-micro, 5-macro, 15-continuous, 15-macro | 1-continuous, 15-micro |
| 1: appearance (pink pig/blocks) | policy:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 82 | 2 | 16 | 0.13 [0.09, 0.18] | 15-continuous | 15-macro |
| 1: appearance (pink pig/blocks) | predictor:J | 28 | 1 | 43 | 0 | 0 | 2 | 19 | 7 | 0 | 0.66 [0.64, 0.68] | 1-macro, 5-macro, 15-continuous | 1-continuous, 1-micro, 5-continuous, 5-micro, 15-micro, 15-macro |
| 1: appearance (pink pig/blocks) | predictor:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 82 | 1 | 18 | 0.29 [0.26, 0.33] | 15-continuous | 15-micro, 15-macro |
| 1: appearance (pink pig/blocks) | both:J | 20 | 9 | 34 | 0 | 1 | 0 | 15 | 20 | 1 | 0.68 [0.65, 0.70] | 1-micro, 1-macro, 5-macro, 15-continuous, 15-macro | 1-continuous, 5-continuous, 5-micro, 15-micro |
| 1: appearance (pink pig/blocks) | both:J_dec | 0 | 0 | 0 | 8 | 0 | 0 | 69 | 0 | 24 | 0.35 [0.31, 0.39] | 5-continuous | 15-micro |
| 2: fan (bird push) | zs:J | 50 | 7 | 5 | 0 | 2 | 0 | 2 | 33 | 0 | 0.14 [0.11, 0.18] | - | - |
| 2: fan (bird push) | zs:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 64 | 2 | 35 | 0.15 [0.11, 0.20] | - | - |
| 2: fan (bird push) | policy:J | 48 | 9 | 21 | 1 | 1 | 0 | 6 | 12 | 1 | 0.42 [0.38, 0.47] | 1-macro, 5-continuous, 5-macro, 15-continuous, 15-macro | 5-micro, 15-micro |
| 2: fan (bird push) | policy:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 68 | 5 | 27 | 0.20 [0.15, 0.24] | 15-continuous, 15-micro | 15-macro |
| 2: fan (bird push) | predictor:J | 7 | 1 | 51 | 0 | 0 | 10 | 8 | 20 | 2 | 0.73 [0.70, 0.76] | 1-macro, 5-macro, 15-continuous, 15-macro | 1-continuous, 1-micro, 5-continuous, 5-micro, 15-micro |
| 2: fan (bird push) | predictor:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 77 | 0 | 23 | 0.41 [0.36, 0.45] | 15-continuous | 15-micro, 15-macro |
| 2: fan (bird push) | both:J | 43 | 4 | 5 | 0 | 1 | 0 | 8 | 34 | 4 | 0.47 [0.43, 0.52] | 15-continuous, 15-macro | 1-continuous |
| 2: fan (bird push) | both:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 65 | 2 | 33 | 0.43 [0.38, 0.47] | - | - |
| 3: stronger air turbulence | zs:J | 18 | 15 | 7 | 0 | 0 | 0 | 6 | 55 | 0 | 0.21 [0.14, 0.28] | 1-micro, 15-micro | 1-macro |
| 3: stronger air turbulence | zs:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 76 | 0 | 23 | 0.12 [0.08, 0.16] | 15-continuous | 15-macro |
| 3: stronger air turbulence | policy:J | 33 | 2 | 22 | 0 | 2 | 0 | 6 | 25 | 10 | 0.42 [0.38, 0.47] | 1-continuous, 1-macro, 5-micro, 15-macro | 1-micro, 15-micro |
| 3: stronger air turbulence | policy:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 56 | 10 | 34 | 0.24 [0.18, 0.30] | 15-micro | 15-continuous |
| 3: stronger air turbulence | predictor:J | 3 | 1 | 59 | 0 | 0 | 2 | 2 | 33 | 0 | 0.54 [0.50, 0.58] | 1-macro, 5-macro | 1-continuous, 1-micro, 15-continuous, 15-micro |
| 3: stronger air turbulence | predictor:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 43 | 2 | 55 | 0.37 [0.32, 0.42] | 15-micro, 15-macro | 15-continuous |
| 3: stronger air turbulence | both:J | 26 | 15 | 14 | 0 | 1 | 2 | 9 | 24 | 9 | 0.59 [0.56, 0.63] | 1-continuous, 1-micro, 5-continuous, 5-macro, 15-macro | 15-micro |
| 3: stronger air turbulence | both:J_dec | 0 | 0 | 8 | 0 | 0 | 1 | 53 | 0 | 37 | 0.31 [0.21, 0.42] | 1-macro, 5-macro | 15-continuous |
| 4: magnet | zs:J | 45 | 10 | 5 | 0 | 1 | 0 | 2 | 36 | 0 | 0.10 [0.06, 0.15] | - | - |
| 4: magnet | zs:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 65 | 2 | 32 | 0.11 [0.06, 0.17] | - | - |
| 4: magnet | policy:J | 37 | 24 | 17 | 2 | 1 | 0 | 0 | 14 | 4 | 0.46 [0.42, 0.50] | 1-micro, 1-macro, 5-continuous, 5-macro, 15-macro | 1-continuous, 15-continuous, 15-micro |
| 4: magnet | policy:J_dec | 2 | 0 | 0 | 2 | 0 | 1 | 71 | 5 | 19 | 0.27 [0.21, 0.32] | 1-continuous, 5-continuous, 5-macro, 15-micro | 15-macro |
| 4: magnet | predictor:J | 4 | 1 | 55 | 0 | 0 | 0 | 5 | 34 | 0 | 0.70 [0.67, 0.73] | 1-macro, 5-macro, 15-continuous | 1-continuous, 1-micro, 5-continuous, 5-micro, 15-macro |
| 4: magnet | predictor:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 75 | 0 | 25 | 0.37 [0.31, 0.42] | 15-continuous | 15-micro |
| 4: magnet | both:J | 29 | 2 | 18 | 0 | 0 | 5 | 27 | 14 | 5 | 0.76 [0.69, 0.83] | 1-macro, 5-macro, 15-continuous, 15-macro | 1-continuous, 1-micro, 5-continuous, 15-micro |
| 4: magnet | both:J_dec | 0 | 0 | 0 | 0 | 0 | 1 | 60 | 3 | 37 | 0.50 [0.45, 0.56] | 5-macro | - |
| 5: right slingshot (mirrored layout) | zs:J | 42 | 10 | 1 | 0 | 1 | 0 | 4 | 42 | 0 | 0.27 [0.24, 0.32] | 1-micro, 15-micro | 1-continuous, 1-macro, 5-continuous, 5-micro |
| 5: right slingshot (mirrored layout) | zs:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 89 | 2 | 9 | 0.18 [0.15, 0.21] | 15-continuous, 15-micro | 15-macro |
| 5: right slingshot (mirrored layout) | policy:J | 32 | 10 | 27 | 5 | 10 | 0 | 8 | 8 | 0 | 0.52 [0.47, 0.57] | 1-micro, 1-macro, 5-continuous, 5-micro, 5-macro, 15-continuous | 1-continuous, 15-micro |
| 5: right slingshot (mirrored layout) | policy:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 72 | 2 | 25 | 0.20 [0.17, 0.23] | 1-macro, 15-micro | - |
| 5: right slingshot (mirrored layout) | predictor:J | 33 | 1 | 50 | 0 | 0 | 5 | 6 | 2 | 3 | 0.70 [0.69, 0.71] | 1-macro, 5-macro, 15-continuous, 15-macro | 1-continuous, 1-micro, 5-continuous, 5-micro, 15-micro |
| 5: right slingshot (mirrored layout) | predictor:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 53 | 2 | 45 | 0.40 [0.38, 0.43] | 15-micro, 15-macro | 15-continuous |
| 5: right slingshot (mirrored layout) | both:J | 32 | 18 | 13 | 0 | 0 | 0 | 11 | 18 | 8 | 0.57 [0.54, 0.60] | 1-micro, 1-macro, 5-macro, 15-continuous, 15-macro | 1-continuous, 5-continuous, 5-micro, 15-micro |
| 5: right slingshot (mirrored layout) | both:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 67 | 5 | 27 | 0.39 [0.36, 0.42] | 1-continuous, 1-macro, 5-continuous, 5-macro, 15-micro | 15-continuous |
| 6: inverse gravity | zs:J | 63 | 3 | 1 | 3 | 1 | 0 | 2 | 27 | 0 | 0.25 [0.20, 0.30] | 1-continuous, 5-continuous | 1-micro, 1-macro, 5-micro |
| 6: inverse gravity | zs:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 88 | 2 | 9 | 0.15 [0.09, 0.23] | 15-continuous | 15-macro |
| 6: inverse gravity | policy:J | 34 | 1 | 16 | 3 | 0 | 20 | 5 | 9 | 11 | 0.66 [0.63, 0.70] | 1-macro, 5-continuous, 5-macro, 15-continuous, 15-macro | 1-continuous, 1-micro, 5-micro, 15-micro |
| 6: inverse gravity | policy:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 85 | 2 | 12 | 0.16 [0.10, 0.23] | 5-continuous, 5-macro, 15-continuous | 15-macro |
| 6: inverse gravity | predictor:J | 10 | 2 | 57 | 0 | 1 | 0 | 2 | 28 | 0 | 0.67 [0.64, 0.71] | 1-macro | 1-continuous, 1-micro, 5-continuous, 5-micro |
| 6: inverse gravity | predictor:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 77 | 1 | 23 | 0.27 [0.22, 0.31] | - | 15-micro |
| 6: inverse gravity | both:J | 14 | 3 | 38 | 0 | 1 | 1 | 4 | 19 | 21 | 0.71 [0.68, 0.73] | 1-macro, 5-macro, 15-macro | 1-continuous, 1-micro, 5-continuous, 5-micro, 15-micro |
| 6: inverse gravity | both:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 66 | 1 | 33 | 0.42 [0.36, 0.50] | 15-macro | 15-continuous |
| 7: inverse air turbulence | zs:J | 23 | 7 | 15 | 0 | 0 | 0 | 7 | 46 | 1 | 0.15 [0.09, 0.22] | 15-macro | - |
| 7: inverse air turbulence | zs:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 67 | 2 | 31 | 0.08 [0.04, 0.13] | 15-macro | 15-continuous |
| 7: inverse air turbulence | policy:J | 27 | 2 | 32 | 2 | 0 | 0 | 6 | 25 | 6 | 0.49 [0.46, 0.53] | 1-macro, 5-continuous, 5-micro, 5-macro, 15-macro | 1-micro, 15-micro |
| 7: inverse air turbulence | policy:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 59 | 5 | 36 | 0.28 [0.23, 0.32] | 15-micro, 15-macro | 15-continuous |
| 7: inverse air turbulence | predictor:J | 0 | 1 | 82 | 0 | 0 | 0 | 1 | 13 | 2 | 0.73 [0.70, 0.76] | 1-macro, 5-macro, 15-macro | 1-continuous, 1-micro, 5-micro, 15-continuous, 15-micro |
| 7: inverse air turbulence | predictor:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 70 | 2 | 28 | 0.38 [0.34, 0.42] | - | - |
| 7: inverse air turbulence | both:J | 41 | 21 | 5 | 0 | 3 | 0 | 18 | 10 | 2 | 0.67 [0.63, 0.70] | 1-continuous, 1-micro, 5-continuous, 5-micro, 5-macro, 15-continuous, 15-macro | 1-macro, 15-micro |
| 7: inverse air turbulence | both:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 65 | 3 | 32 | 0.30 [0.25, 0.34] | 15-micro, 15-macro | 15-continuous |
| novel 1-7 pooled | zs:J | 46 | 7 | 5 | 1 | 1 | 0 | 3 | 36 | 0 | 0.17 [0.15, 0.20] | 1-micro, 15-micro, 15-macro | 1-macro |
| novel 1-7 pooled | zs:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 75 | 2 | 23 | 0.13 [0.11, 0.15] | 15-continuous | 15-macro |
| novel 1-7 pooled | policy:J | 35 | 7 | 24 | 2 | 4 | 4 | 5 | 13 | 5 | 0.52 [0.50, 0.54] | 1-micro, 1-macro, 5-continuous, 5-micro, 5-macro, 15-continuous, 15-macro | 1-continuous, 15-micro |
| novel 1-7 pooled | policy:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 72 | 4 | 24 | 0.20 [0.19, 0.22] | 1-continuous, 1-macro, 5-continuous, 5-micro, 5-macro, 15-micro | 15-macro |
| novel 1-7 pooled | predictor:J | 13 | 1 | 57 | 0 | 0 | 3 | 6 | 18 | 1 | 0.69 [0.67, 0.70] | 1-macro, 5-macro, 15-continuous, 15-macro | 1-continuous, 1-micro, 5-continuous, 5-micro, 15-micro |
| novel 1-7 pooled | predictor:J_dec | 0 | 0 | 0 | 0 | 0 | 0 | 70 | 1 | 29 | 0.35 [0.34, 0.37] | - | 15-micro |
| novel 1-7 pooled | both:J | 29 | 10 | 19 | 0 | 1 | 1 | 13 | 20 | 7 | 0.63 [0.61, 0.65] | 1-micro, 1-macro, 5-macro, 15-continuous, 15-macro | 1-continuous, 5-continuous, 15-micro |
| novel 1-7 pooled | both:J_dec | 0 | 0 | 1 | 1 | 0 | 0 | 65 | 2 | 31 | 0.39 [0.37, 0.41] | 1-continuous, 1-macro, 5-continuous, 5-macro, 15-macro | 15-continuous |

Shift of the horizon marginal (percentage points, Delta = 1 / 5 / 15) vs the paired normal side:

| population | arm | Delta 1 | Delta 5 | Delta 15 | continuous | micro | macro |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1: appearance (pink pig/blocks) | zs:J | +2 [+0, +4] | +1 [+0, +3] | -3 [-7, -1] | +5 [+1, +10] | -4 [-8, -1] | -1 [-3, +1] |
| 1: appearance (pink pig/blocks) | zs:J_dec | +0 [+0, +0] | +0 [+0, +0] | -0 [-0, +0] | +3 [-1, +8] | +0 [-0, +0] | -3 [-8, +1] |
| 1: appearance (pink pig/blocks) | policy:J | +0 [-3, +4] | +16 [+13, +18] | -16 [-19, -13] | -24 [-31, -18] | -14 [-18, -10] | +38 [+35, +41] |
| 1: appearance (pink pig/blocks) | policy:J_dec | +0 [+0, +0] | +0 [+0, +0] | -0 [-0, +0] | +10 [+5, +15] | +0 [-0, +0] | -10 [-15, -5] |
| 1: appearance (pink pig/blocks) | predictor:J | +4 [+1, +6] | +0 [-1, +1] | -4 [-7, -1] | -19 [-25, -14] | -23 [-27, -20] | +43 [+40, +46] |
| 1: appearance (pink pig/blocks) | predictor:J_dec | +0 [+0, +0] | +0 [+0, +0] | -0 [-0, -0] | +9 [+2, +15] | -1 [-1, -0] | -8 [-14, -2] |
| 1: appearance (pink pig/blocks) | both:J | -5 [-9, -1] | -2 [-3, -1] | +7 [+3, +11] | -31 [-38, -23] | -2 [-8, +3] | +33 [+27, +38] |
| 1: appearance (pink pig/blocks) | both:J_dec | +0 [+0, +0] | +8 [+5, +11] | -8 [-11, -5] | +3 [-2, +9] | -1 [-2, -1] | -2 [-8, +4] |
| 2: fan (bird push) | zs:J | -1 [-3, +1] | -0 [-1, +1] | +1 [-1, +3] | -0 [-3, +3] | +2 [-2, +6] | -2 [-4, +2] |
| 2: fan (bird push) | zs:J_dec | +0 [+0, +0] | +0 [+0, +0] | -0 [-0, -0] | +3 [-3, +9] | -0 [-1, +0] | -3 [-8, +3] |
| 2: fan (bird push) | policy:J | +14 [+12, +16] | +0 [-1, +1] | -14 [-16, -12] | +2 [-6, +10] | -18 [-22, -14] | +16 [+7, +25] |
| 2: fan (bird push) | policy:J_dec | +0 [+0, +0] | +0 [+0, +0] | -0 [-0, +0] | +8 [+2, +14] | +3 [+2, +3] | -11 [-16, -5] |
| 2: fan (bird push) | predictor:J | -4 [-9, +1] | +8 [+5, +11] | -4 [-8, -0] | -38 [-41, -34] | -18 [-23, -13] | +56 [+52, +59] |
| 2: fan (bird push) | predictor:J_dec | +0 [+0, +0] | +0 [+0, +0] | -0 [-0, -0] | +16 [+9, +23] | -1 [-2, -1] | -15 [-22, -8] |
| 2: fan (bird push) | both:J | -12 [-15, -7] | -1 [-2, +0] | +12 [+9, +16] | -1 [-9, +6] | -1 [-8, +6] | +2 [-3, +7] |
| 2: fan (bird push) | both:J_dec | +0 [+0, +0] | +0 [+0, +0] | -0 [-0, -0] | +5 [-2, +12] | +0 [-1, +2] | -5 [-13, +3] |
| 3: stronger air turbulence | zs:J | -2 [-5, +2] | -0 [-1, +0] | +2 [-1, +5] | -4 [-9, +1] | +11 [+5, +17] | -7 [-10, -3] |
| 3: stronger air turbulence | zs:J_dec | +0 [+0, +0] | +0 [+0, +1] | -0 [-1, -0] | +6 [+1, +11] | -0 [-1, +1] | -6 [-11, -1] |
| 3: stronger air turbulence | policy:J | +16 [+12, +20] | +1 [+1, +2] | -17 [-21, -14] | +11 [+6, +16] | -30 [-36, -24] | +18 [+13, +24] |
| 3: stronger air turbulence | policy:J_dec | +0 [+0, +0] | +0 [+0, +0] | -0 [-0, +0] | -14 [-22, -7] | +10 [+8, +12] | +4 [-2, +10] |
| 3: stronger air turbulence | predictor:J | +22 [+16, +26] | +2 [+1, +3] | -24 [-27, -19] | -22 [-28, -15] | -25 [-30, -21] | +48 [+40, +54] |
| 3: stronger air turbulence | predictor:J_dec | +0 [+0, +0] | +0 [+0, +0] | -0 [-0, +0] | -27 [-34, -21] | +2 [+1, +3] | +25 [+19, +31] |
| 3: stronger air turbulence | both:J | +14 [+10, +18] | +3 [+0, +6] | -17 [-21, -12] | +8 [+2, +13] | -18 [-28, -10] | +11 [+3, +18] |
| 3: stronger air turbulence | both:J_dec | +8 [+5, +12] | +1 [+0, +1] | -9 [-13, -5] | -17 [-28, -7] | +0 [-0, +1] | +17 [+7, +28] |
| 4: magnet | zs:J | -1 [-3, +1] | +0 [-1, +1] | +1 [-1, +3] | -1 [-6, +2] | +1 [-3, +4] | +1 [-2, +4] |
| 4: magnet | zs:J_dec | +0 [+0, +0] | +0 [+0, +0] | -0 [-0, +0] | -1 [-8, +5] | +0 [-0, +0] | +1 [-5, +8] |
| 4: magnet | policy:J | +18 [+15, +20] | +2 [+1, +3] | -20 [-22, -17] | -9 [-14, -3] | -8 [-12, -3] | +17 [+11, +23] |
| 4: magnet | policy:J_dec | +2 [+1, +4] | +3 [+1, +6] | -6 [-10, -2] | +8 [+2, +16] | +2 [+1, +4] | -11 [-18, -4] |
| 4: magnet | predictor:J | -1 [-7, +5] | -1 [-2, -0] | +2 [-4, +8] | -38 [-41, -36] | -12 [-19, -5] | +51 [+45, +56] |
| 4: magnet | predictor:J_dec | +0 [+0, +0] | +0 [+0, +0] | -0 [-0, +0] | +8 [+1, +16] | -2 [-2, -1] | -6 [-14, +1] |
| 4: magnet | both:J | -12 [-17, -8] | +4 [+1, +8] | +8 [+3, +12] | +8 [-4, +20] | -31 [-43, -17] | +23 [+16, +30] |
| 4: magnet | both:J_dec | +0 [+0, +0] | +1 [+0, +2] | -1 [-2, -0] | -7 [-14, +0] | +0 [-1, +2] | +6 [-1, +14] |
| 5: right slingshot (mirrored layout) | zs:J | -14 [-17, -11] | -2 [-3, -1] | +16 [+13, +20] | -20 [-25, -16] | +22 [+18, +26] | -2 [-3, -1] |
| 5: right slingshot (mirrored layout) | zs:J_dec | +0 [+0, +0] | +0 [+0, +0] | +0 [-0, +0] | +16 [+13, +19] | +0 [+0, +1] | -17 [-20, -14] |
| 5: right slingshot (mirrored layout) | policy:J | +2 [-2, +5] | +12 [+9, +15] | -13 [-17, -10] | -21 [-28, -14] | -3 [-7, +1] | +24 [+18, +29] |
| 5: right slingshot (mirrored layout) | policy:J_dec | +0 [+0, +1] | +0 [+0, +0] | -0 [-1, -0] | -1 [-8, +5] | +1 [+1, +2] | -0 [-6, +6] |
| 5: right slingshot (mirrored layout) | predictor:J | +16 [+14, +19] | +2 [+1, +4] | -19 [-22, -16] | -28 [-32, -23] | -28 [-31, -24] | +56 [+53, +58] |
| 5: right slingshot (mirrored layout) | predictor:J_dec | +0 [+0, +0] | +0 [+0, +0] | +0 [-0, +0] | -19 [-24, -15] | +1 [+0, +1] | +19 [+14, +23] |
| 5: right slingshot (mirrored layout) | both:J | -4 [-6, -1] | -3 [-4, -2] | +7 [+4, +9] | -23 [-27, -19] | +6 [+3, +9] | +17 [+15, +20] |
| 5: right slingshot (mirrored layout) | both:J_dec | +0 [+0, +1] | +0 [+0, +1] | -1 [-1, -0] | -5 [-10, -0] | +4 [+2, +6] | +2 [-4, +8] |
| 6: inverse gravity | zs:J | +3 [-2, +7] | +0 [-0, +1] | -3 [-8, +2] | +8 [+1, +15] | -5 [-11, +2] | -3 [-6, -1] |
| 6: inverse gravity | zs:J_dec | +0 [+0, +0] | +0 [+0, +0] | -0 [-0, +0] | +11 [+4, +19] | +0 [-0, +1] | -11 [-20, -5] |
| 6: inverse gravity | policy:J | -13 [-16, -10] | +19 [+19, +20] | -6 [-10, -3] | -17 [-23, -11] | -26 [-31, -21] | +43 [+40, +46] |
| 6: inverse gravity | policy:J_dec | +0 [+0, +0] | +0 [+0, +1] | -0 [-1, -0] | +8 [+2, +15] | +0 [-1, +2] | -8 [-16, -2] |
| 6: inverse gravity | predictor:J | +5 [+1, +8] | -2 [-3, -2] | -2 [-6, +2] | -47 [-52, -42] | -6 [-11, -1] | +53 [+49, +56] |
| 6: inverse gravity | predictor:J_dec | +0 [+0, +0] | +0 [+0, +0] | -0 [-0, +0] | -0 [-8, +8] | -1 [-2, -1] | +2 [-7, +9] |
| 6: inverse gravity | both:J | -10 [-15, -6] | -2 [-2, -1] | +12 [+7, +17] | -42 [-50, -34] | -15 [-22, -7] | +56 [+52, +60] |
| 6: inverse gravity | both:J_dec | +0 [+0, +0] | +0 [+0, +1] | -0 [-1, -0] | -11 [-18, -4] | -1 [-2, +0] | +12 [+4, +19] |
| 7: inverse air turbulence | zs:J | -2 [-5, +2] | +0 [-0, +0] | +2 [-2, +5] | -3 [-8, +4] | +0 [-3, +4] | +2 [-1, +5] |
| 7: inverse air turbulence | zs:J_dec | +0 [+0, +0] | +0 [+0, +0] | +0 [-0, +0] | -5 [-9, -1] | +0 [+0, +0] | +5 [+1, +9] |
| 7: inverse air turbulence | policy:J | +14 [+11, +17] | +3 [+1, +4] | -17 [-20, -13] | +2 [-5, +8] | -27 [-31, -22] | +25 [+19, +31] |
| 7: inverse air turbulence | policy:J_dec | +0 [+0, +0] | -0 [-0, +0] | +0 [-0, +0] | -13 [-19, -7] | +3 [+1, +5] | +10 [+4, +16] |
| 7: inverse air turbulence | predictor:J | +36 [+31, +40] | +0 [+0, +1] | -36 [-40, -31] | -32 [-37, -28] | -39 [-45, -34] | +71 [+68, +75] |
| 7: inverse air turbulence | predictor:J_dec | +0 [+0, +0] | +0 [-0, +1] | -0 [-1, +0] | -2 [-7, +4] | +0 [-0, +0] | +2 [-4, +7] |
| 7: inverse air turbulence | both:J | +19 [+16, +23] | +3 [+2, +4] | -23 [-26, -19] | +26 [+16, +35] | -19 [-26, -11] | -7 [-10, -3] |
| 7: inverse air turbulence | both:J_dec | +0 [+0, +0] | -0 [-0, +0] | +0 [-0, +0] | -8 [-12, -3] | +1 [+0, +2] | +6 [+2, +11] |
| novel 1-7 pooled | zs:J | -2 [-4, -1] | -0 [-0, +0] | +2 [+1, +4] | -2 [-4, +0] | +3 [+1, +5] | -1 [-2, -0] |
| novel 1-7 pooled | zs:J_dec | +0 [+0, +0] | +0 [+0, +0] | -0 [-0, -0] | +5 [+3, +7] | +0 [-0, +0] | -5 [-7, -3] |
| novel 1-7 pooled | policy:J | +6 [+4, +8] | +8 [+7, +10] | -14 [-16, -13] | -10 [-13, -7] | -17 [-19, -15] | +27 [+24, +29] |
| novel 1-7 pooled | policy:J_dec | +0 [+0, +1] | +0 [+0, +1] | -1 [-1, -0] | +2 [-1, +4] | +2 [+2, +3] | -4 [-7, -1] |
| novel 1-7 pooled | predictor:J | +11 [+8, +13] | +1 [+1, +2] | -12 [-14, -10] | -33 [-35, -30] | -22 [-24, -19] | +54 [+53, +56] |
| novel 1-7 pooled | predictor:J_dec | +0 [+0, +0] | +0 [-0, +0] | -0 [-0, +0] | -1 [-4, +2] | -0 [-1, -0] | +1 [-2, +4] |
| novel 1-7 pooled | both:J | -2 [-4, +0] | +0 [-0, +1] | +2 [-0, +4] | -10 [-14, -5] | -10 [-13, -7] | +20 [+17, +23] |
| novel 1-7 pooled | both:J_dec | +1 [+0, +1] | +1 [+1, +2] | -2 [-3, -1] | -5 [-7, -2] | +1 [-0, +1] | +4 [+2, +7] |

## Request shares on every supported cell (post-freeze DESCRIPTIVE supplement)

post-freeze DESCRIPTIVE supplement (phase-A review F2), outcome-free: request shares of the frozen policies on every #110-supported cell. The frozen scope's records give the in-scope cells; the supported cells the frozen scope excluded (no mixed-verdict level, or rollout-error only; all of level 8) were rolled out after the publication with the same frozen policies, worlds and anchor rule, zero engine seconds, under deterministic_scoring(), for request shares only (supplement/). No verdict is read, no AUC, contrast or token is formed, and the frozen tables, scope and dispositions are unchanged. Policy-only and both-retuned policies exist only for the levels the frozen scope trained them on (1-7). 186 supplement records, 141 GPU s, 0 engine s; hashes in `supplement/manifest.json`. Horizon (Delta 1 / 5 / 15) and description (continuous / micro / macro) marginals, % of rollout steps; shift vs the same kind's normal policy on the paired normal side.

| novelty type | arm | units | Delta 1 | Delta 5 | Delta 15 | continuous | micro | macro | TV shift vs paired normal | moves toward | moves away |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0: normal (paired normal sides) | zs:J | 960 | 61 | 2 | 37 | 55 | 39 | 6 | n/a | - | - |
| 0: normal (paired normal sides) | zs:J_dec | 960 | 0 | 0 | 100 | 72 | 2 | 27 | n/a | - | - |
| 1: appearance (pink pig/blocks) | zs:J | 120 | 72 | 4 | 24 | 72 | 26 | 2 | 0.12 [0.07, 0.17] | 1-continuous, 5-continuous, 5-micro | 1-micro, 15-micro, 15-macro |
| 1: appearance (pink pig/blocks) | zs:J_dec | 120 | 0 | 0 | 100 | 77 | 2 | 21 | 0.09 [0.06, 0.13] | 15-continuous | 15-macro |
| 1: appearance (pink pig/blocks) | policy:J | 120 | 68 | 19 | 12 | 40 | 17 | 43 | 0.62 [0.60, 0.64] | 1-macro, 5-micro, 5-macro, 15-continuous, 15-macro | 1-continuous, 15-micro |
| 1: appearance (pink pig/blocks) | policy:J_dec | 120 | 0 | 0 | 100 | 82 | 2 | 16 | 0.13 [0.10, 0.17] | 15-continuous | 15-macro |
| 1: appearance (pink pig/blocks) | predictor:J | 120 | 73 | 3 | 25 | 47 | 7 | 46 | 0.66 [0.65, 0.68] | 1-macro, 5-macro, 15-continuous | 1-continuous, 1-micro, 5-continuous, 5-micro, 15-micro, 15-macro |
| 1: appearance (pink pig/blocks) | predictor:J_dec | 120 | 0 | 0 | 100 | 80 | 1 | 19 | 0.30 [0.27, 0.33] | 15-continuous | 15-micro, 15-macro |
| 1: appearance (pink pig/blocks) | both:J | 120 | 63 | 1 | 36 | 35 | 28 | 37 | 0.68 [0.66, 0.71] | 1-micro, 1-macro, 5-macro, 15-continuous, 15-macro | 1-continuous, 5-continuous, 5-micro, 15-micro |
| 1: appearance (pink pig/blocks) | both:J_dec | 120 | 0 | 8 | 92 | 78 | 0 | 22 | 0.33 [0.30, 0.37] | 5-continuous | 15-micro |
| 2: fan (bird push) | zs:J | 117 | 64 | 2 | 34 | 57 | 38 | 5 | 0.13 [0.10, 0.16] | - | - |
| 2: fan (bird push) | zs:J_dec | 117 | 0 | 0 | 100 | 65 | 2 | 33 | 0.14 [0.10, 0.18] | - | - |
| 2: fan (bird push) | policy:J | 117 | 79 | 3 | 18 | 53 | 19 | 27 | 0.45 [0.40, 0.49] | 1-macro, 5-continuous, 5-macro, 15-continuous, 15-macro | 1-continuous, 5-micro, 15-micro |
| 2: fan (bird push) | policy:J_dec | 117 | 0 | 0 | 100 | 70 | 5 | 26 | 0.18 [0.15, 0.22] | 15-continuous, 15-micro | 15-macro |
| 2: fan (bird push) | predictor:J | 117 | 59 | 9 | 32 | 18 | 22 | 61 | 0.75 [0.72, 0.78] | 1-macro, 5-macro, 15-continuous, 15-macro | 1-continuous, 1-micro, 5-continuous, 5-micro, 15-micro |
| 2: fan (bird push) | predictor:J_dec | 117 | 0 | 0 | 100 | 76 | 1 | 24 | 0.40 [0.36, 0.44] | 15-continuous | 15-micro, 15-macro |
| 2: fan (bird push) | both:J | 117 | 52 | 1 | 46 | 51 | 38 | 12 | 0.47 [0.44, 0.51] | 15-continuous, 15-macro | 1-continuous, 5-continuous |
| 2: fan (bird push) | both:J_dec | 117 | 0 | 0 | 100 | 64 | 2 | 34 | 0.42 [0.39, 0.46] | - | - |
| 3: stronger air turbulence | zs:J | 120 | 39 | 0 | 61 | 27 | 62 | 11 | 0.24 [0.20, 0.29] | 1-micro, 15-micro | 1-continuous, 1-macro |
| 3: stronger air turbulence | zs:J_dec | 120 | 0 | 0 | 100 | 73 | 1 | 26 | 0.14 [0.11, 0.17] | - | - |
| 3: stronger air turbulence | policy:J | 120 | 57 | 2 | 41 | 39 | 32 | 29 | 0.46 [0.43, 0.49] | 1-continuous, 1-macro, 5-micro, 15-macro | 1-micro, 15-continuous, 15-micro |
| 3: stronger air turbulence | policy:J_dec | 120 | 0 | 0 | 100 | 56 | 10 | 34 | 0.28 [0.25, 0.32] | 15-micro | 15-continuous |
| 3: stronger air turbulence | predictor:J | 120 | 57 | 3 | 40 | 19 | 33 | 49 | 0.54 [0.51, 0.57] | 1-macro, 5-macro | 1-continuous, 1-micro, 15-micro, 15-macro |
| 3: stronger air turbulence | predictor:J_dec | 120 | 0 | 0 | 100 | 53 | 1 | 46 | 0.36 [0.32, 0.40] | 15-macro | 15-continuous |
| 3: stronger air turbulence | both:J | 120 | 54 | 8 | 38 | 49 | 31 | 20 | 0.63 [0.59, 0.66] | 1-continuous, 5-continuous, 5-micro, 5-macro, 15-macro | 15-micro |
| 3: stronger air turbulence | both:J_dec | 120 | 9 | 1 | 90 | 56 | 0 | 44 | 0.37 [0.31, 0.42] | 1-continuous, 1-macro, 5-continuous, 5-macro | 15-continuous, 15-micro |
| 4: magnet | zs:J | 120 | 61 | 2 | 37 | 49 | 46 | 5 | 0.10 [0.07, 0.13] | - | - |
| 4: magnet | zs:J_dec | 120 | 0 | 0 | 100 | 68 | 2 | 30 | 0.13 [0.09, 0.17] | - | - |
| 4: magnet | policy:J | 120 | 79 | 4 | 18 | 40 | 37 | 23 | 0.48 [0.45, 0.51] | 1-micro, 1-macro, 5-continuous, 5-macro, 15-macro | 1-continuous, 15-continuous, 15-micro |
| 4: magnet | policy:J_dec | 120 | 3 | 4 | 93 | 77 | 5 | 19 | 0.26 [0.23, 0.31] | 1-continuous, 1-macro, 5-continuous, 5-macro, 15-micro | 15-macro |
| 4: magnet | predictor:J | 120 | 60 | 0 | 40 | 12 | 34 | 54 | 0.69 [0.67, 0.72] | 1-macro, 5-macro, 15-continuous | 1-continuous, 1-micro, 5-continuous, 5-micro, 15-macro |
| 4: magnet | predictor:J_dec | 120 | 0 | 0 | 100 | 77 | 0 | 23 | 0.35 [0.31, 0.39] | 15-continuous | 15-micro, 15-macro |
| 4: magnet | both:J | 120 | 48 | 7 | 45 | 53 | 17 | 29 | 0.77 [0.71, 0.83] | 1-macro, 5-macro, 15-continuous, 15-macro | 1-continuous, 1-micro, 5-continuous, 5-micro, 15-micro |
| 4: magnet | both:J_dec | 120 | 0 | 0 | 100 | 62 | 2 | 36 | 0.50 [0.45, 0.54] | 5-macro, 15-macro | 15-continuous |
| 5: right slingshot (mirrored layout) | zs:J | 120 | 53 | 1 | 46 | 45 | 53 | 2 | 0.28 [0.25, 0.31] | 1-micro, 15-micro | 1-continuous, 1-macro, 5-continuous, 5-micro |
| 5: right slingshot (mirrored layout) | zs:J_dec | 120 | 0 | 0 | 100 | 89 | 2 | 10 | 0.18 [0.15, 0.21] | 15-continuous, 15-micro | 15-macro |
| 5: right slingshot (mirrored layout) | policy:J | 120 | 68 | 16 | 16 | 44 | 30 | 26 | 0.52 [0.48, 0.57] | 1-micro, 1-macro, 5-continuous, 5-micro, 5-macro, 15-continuous | 1-continuous, 15-micro |
| 5: right slingshot (mirrored layout) | policy:J_dec | 120 | 0 | 0 | 100 | 70 | 3 | 27 | 0.18 [0.15, 0.21] | 1-macro, 15-micro | - |
| 5: right slingshot (mirrored layout) | predictor:J | 120 | 83 | 5 | 13 | 37 | 3 | 59 | 0.71 [0.70, 0.73] | 1-macro, 5-macro, 15-macro | 1-continuous, 1-micro, 5-continuous, 5-micro, 15-micro |
| 5: right slingshot (mirrored layout) | predictor:J_dec | 120 | 0 | 0 | 100 | 54 | 2 | 44 | 0.41 [0.39, 0.43] | 15-micro, 15-macro | 15-continuous |
| 5: right slingshot (mirrored layout) | both:J | 120 | 62 | 0 | 37 | 43 | 37 | 20 | 0.57 [0.54, 0.60] | 1-micro, 1-macro, 5-macro, 15-continuous, 15-macro | 1-continuous, 5-continuous, 5-micro, 15-micro |
| 5: right slingshot (mirrored layout) | both:J_dec | 120 | 1 | 0 | 99 | 67 | 5 | 28 | 0.40 [0.38, 0.43] | 1-continuous, 1-macro, 5-continuous, 5-macro, 15-micro | 15-continuous |
| 6: inverse gravity | zs:J | 120 | 69 | 4 | 28 | 69 | 30 | 1 | 0.23 [0.18, 0.28] | 1-continuous, 5-continuous | 1-micro, 1-macro, 5-micro |
| 6: inverse gravity | zs:J_dec | 120 | 0 | 0 | 100 | 88 | 2 | 10 | 0.13 [0.07, 0.19] | 15-continuous | 15-macro |
| 6: inverse gravity | policy:J | 120 | 51 | 23 | 26 | 43 | 10 | 47 | 0.65 [0.62, 0.69] | 1-macro, 5-continuous, 5-macro, 15-continuous, 15-macro | 1-continuous, 1-micro, 5-micro, 15-micro |
| 6: inverse gravity | policy:J_dec | 120 | 0 | 0 | 100 | 86 | 2 | 12 | 0.14 [0.09, 0.19] | 5-continuous, 5-macro, 15-continuous | 15-macro |
| 6: inverse gravity | predictor:J | 120 | 70 | 1 | 29 | 13 | 29 | 58 | 0.68 [0.65, 0.71] | 1-macro, 5-macro | 1-continuous, 1-micro, 5-continuous, 5-micro |
| 6: inverse gravity | predictor:J_dec | 120 | 0 | 0 | 100 | 78 | 1 | 21 | 0.25 [0.21, 0.29] | - | 15-micro |
| 6: inverse gravity | both:J | 120 | 54 | 2 | 44 | 17 | 24 | 59 | 0.71 [0.69, 0.73] | 1-macro, 5-macro, 15-macro | 1-continuous, 1-micro, 5-continuous, 5-micro, 15-micro |
| 6: inverse gravity | both:J_dec | 120 | 0 | 0 | 100 | 67 | 1 | 32 | 0.40 [0.34, 0.46] | 15-macro | 15-continuous, 15-micro |
| 7: inverse air turbulence | zs:J | 120 | 46 | 0 | 54 | 34 | 52 | 14 | 0.15 [0.10, 0.21] | 15-macro | - |
| 7: inverse air turbulence | zs:J_dec | 120 | 0 | 0 | 100 | 71 | 2 | 27 | 0.08 [0.04, 0.12] | 15-micro, 15-macro | 15-continuous |
| 7: inverse air turbulence | policy:J | 120 | 61 | 2 | 37 | 34 | 26 | 41 | 0.52 [0.48, 0.55] | 1-macro, 5-continuous, 5-macro, 15-macro | 1-micro, 15-micro |
| 7: inverse air turbulence | policy:J_dec | 120 | 0 | 0 | 100 | 59 | 5 | 36 | 0.29 [0.25, 0.33] | 15-micro, 15-macro | 15-continuous |
| 7: inverse air turbulence | predictor:J | 120 | 81 | 0 | 19 | 2 | 15 | 83 | 0.73 [0.70, 0.75] | 1-macro, 5-macro, 15-macro | 1-continuous, 1-micro, 5-micro, 15-continuous, 15-micro |
| 7: inverse air turbulence | predictor:J_dec | 120 | 0 | 0 | 100 | 70 | 2 | 28 | 0.37 [0.34, 0.40] | 15-macro | 15-continuous |
| 7: inverse air turbulence | both:J | 120 | 68 | 3 | 29 | 55 | 38 | 7 | 0.66 [0.63, 0.69] | 1-continuous, 1-micro, 5-continuous, 5-micro, 5-macro, 15-continuous, 15-macro | 1-macro, 15-micro |
| 7: inverse air turbulence | both:J_dec | 120 | 0 | 0 | 100 | 65 | 3 | 32 | 0.30 [0.26, 0.34] | 15-micro, 15-macro | 15-continuous |
| 8: storm | zs:J | 120 | 68 | 3 | 28 | 66 | 31 | 3 | 0.18 [0.14, 0.22] | - | 15-continuous |
| 8: storm | zs:J_dec | 120 | 0 | 0 | 100 | 72 | 2 | 26 | 0.13 [0.10, 0.16] | - | - |
| 8: storm | predictor:J | 120 | 69 | 0 | 31 | 0 | 31 | 69 | 0.87 [0.83, 0.90] | 1-macro, 15-micro | 1-continuous, 1-micro, 5-continuous, 5-micro, 15-continuous |
| 8: storm | predictor:J_dec | 120 | 0 | 0 | 100 | 61 | 1 | 37 | 0.27 [0.22, 0.33] | 15-macro | 15-continuous, 15-micro |
| novel 1-8 pooled (every supported cell) | zs:J | 957 | 59 | 2 | 39 | 53 | 42 | 5 | 0.18 [0.16, 0.20] | 1-micro, 15-micro | 1-continuous, 1-macro |
| novel 1-8 pooled (every supported cell) | zs:J_dec | 957 | 0 | 0 | 100 | 75 | 2 | 23 | 0.13 [0.11, 0.14] | 15-continuous | 15-macro |
| novel 1-8 pooled (every supported cell) | policy:J | 837 | 66 | 10 | 24 | 42 | 24 | 34 | 0.53 [0.51, 0.54] | 1-micro, 1-macro, 5-continuous, 5-micro, 5-macro, 15-continuous, 15-macro | 1-continuous, 15-micro |
| novel 1-8 pooled (every supported cell) | policy:J_dec | 837 | 0 | 1 | 99 | 71 | 4 | 24 | 0.21 [0.19, 0.23] | 1-continuous, 1-macro, 5-continuous, 5-micro, 5-macro, 15-micro | 15-macro |
| novel 1-8 pooled (every supported cell) | predictor:J | 957 | 69 | 3 | 28 | 18 | 22 | 60 | 0.70 [0.69, 0.72] | 1-macro, 5-macro, 15-continuous, 15-macro | 1-continuous, 1-micro, 5-continuous, 5-micro, 15-micro |
| novel 1-8 pooled (every supported cell) | predictor:J_dec | 957 | 0 | 0 | 100 | 69 | 1 | 30 | 0.34 [0.33, 0.36] | 15-macro | 15-continuous, 15-micro |
| novel 1-8 pooled (every supported cell) | both:J | 837 | 57 | 3 | 39 | 43 | 30 | 26 | 0.64 [0.63, 0.66] | 1-micro, 1-macro, 5-macro, 15-continuous, 15-macro | 1-continuous, 5-continuous, 15-micro |
| novel 1-8 pooled (every supported cell) | both:J_dec | 837 | 1 | 1 | 97 | 66 | 2 | 33 | 0.39 [0.37, 0.41] | 1-continuous, 1-macro, 5-continuous, 5-macro, 15-macro | 15-continuous |

Per cell, zero-shot J_dec and J (source: retained frozen-scope records or the supplement):

| cell | source | arm | Delta 1 | Delta 5 | Delta 15 | continuous | micro | macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0/single_force | retained | zs:J_dec | 0 | 0 | 100 | 82 | 2 | 16 |
| 0/single_force | retained | zs:J | 68 | 2 | 30 | 69 | 28 | 3 |
| 0/multiple_forces | supplement | zs:J_dec | 0 | 0 | 100 | 80 | 2 | 18 |
| 0/multiple_forces | supplement | zs:J | 65 | 2 | 33 | 67 | 31 | 2 |
| 0/rolling | retained | zs:J_dec | 0 | 0 | 100 | 67 | 1 | 32 |
| 0/rolling | retained | zs:J | 58 | 2 | 41 | 47 | 45 | 9 |
| 0/falling | retained | zs:J_dec | 0 | 0 | 100 | 67 | 1 | 32 |
| 0/falling | retained | zs:J | 56 | 2 | 42 | 44 | 48 | 8 |
| 0/sliding | retained | zs:J_dec | 0 | 0 | 100 | 63 | 2 | 35 |
| 0/sliding | retained | zs:J | 57 | 2 | 41 | 47 | 45 | 9 |
| 1/single_force | retained | zs:J_dec | 0 | 0 | 100 | 83 | 2 | 15 |
| 1/single_force | retained | zs:J | 79 | 5 | 16 | 82 | 17 | 0 |
| 1/multiple_forces | supplement | zs:J_dec | 0 | 0 | 100 | 82 | 2 | 16 |
| 1/multiple_forces | supplement | zs:J | 77 | 4 | 19 | 80 | 20 | 0 |
| 1/rolling | retained | zs:J_dec | 0 | 0 | 100 | 72 | 0 | 28 |
| 1/rolling | retained | zs:J | 66 | 2 | 32 | 63 | 34 | 3 |
| 1/falling | retained | zs:J_dec | 0 | 0 | 100 | 74 | 2 | 24 |
| 1/falling | retained | zs:J | 67 | 6 | 28 | 63 | 33 | 4 |
| 1/sliding | retained | zs:J_dec | 0 | 0 | 100 | 77 | 2 | 21 |
| 1/sliding | retained | zs:J | 71 | 2 | 27 | 74 | 25 | 1 |
| 2/single_force | retained | zs:J_dec | 0 | 0 | 100 | 78 | 2 | 20 |
| 2/single_force | retained | zs:J | 74 | 2 | 24 | 77 | 23 | 0 |
| 2/multiple_forces | supplement | zs:J_dec | 0 | 0 | 100 | 72 | 2 | 25 |
| 2/multiple_forces | supplement | zs:J | 70 | 2 | 28 | 74 | 25 | 1 |
| 2/rolling | retained | zs:J_dec | 0 | 0 | 100 | 62 | 2 | 35 |
| 2/rolling | retained | zs:J | 59 | 2 | 39 | 39 | 54 | 7 |
| 2/falling | retained | zs:J_dec | 0 | 0 | 100 | 62 | 0 | 38 |
| 2/falling | retained | zs:J | 63 | 3 | 35 | 54 | 41 | 6 |
| 2/sliding | retained | zs:J_dec | 0 | 0 | 100 | 52 | 1 | 47 |
| 2/sliding | retained | zs:J | 56 | 1 | 43 | 38 | 52 | 10 |
| 3/single_force | supplement | zs:J_dec | 0 | 0 | 100 | 80 | 1 | 19 |
| 3/single_force | supplement | zs:J | 41 | 1 | 58 | 40 | 54 | 6 |
| 3/multiple_forces | supplement | zs:J_dec | 0 | 0 | 100 | 76 | 2 | 22 |
| 3/multiple_forces | supplement | zs:J | 40 | 0 | 60 | 36 | 51 | 14 |
| 3/rolling | retained | zs:J_dec | 0 | 1 | 99 | 79 | 1 | 21 |
| 3/rolling | retained | zs:J | 44 | 0 | 55 | 28 | 63 | 9 |
| 3/falling | retained | zs:J_dec | 0 | 0 | 100 | 74 | 0 | 26 |
| 3/falling | retained | zs:J | 34 | 0 | 66 | 18 | 76 | 6 |
| 3/sliding | supplement | zs:J_dec | 0 | 0 | 100 | 56 | 1 | 42 |
| 3/sliding | supplement | zs:J | 36 | 0 | 64 | 16 | 64 | 20 |
| 4/single_force | retained | zs:J_dec | 0 | 0 | 100 | 69 | 2 | 29 |
| 4/single_force | retained | zs:J | 67 | 1 | 31 | 67 | 29 | 4 |
| 4/multiple_forces | supplement | zs:J_dec | 0 | 0 | 100 | 77 | 2 | 21 |
| 4/multiple_forces | supplement | zs:J | 70 | 2 | 27 | 73 | 27 | 1 |
| 4/rolling | retained | zs:J_dec | 0 | 0 | 100 | 61 | 2 | 37 |
| 4/rolling | retained | zs:J | 56 | 1 | 43 | 38 | 55 | 7 |
| 4/falling | retained | zs:J_dec | 0 | 0 | 100 | 66 | 2 | 32 |
| 4/falling | retained | zs:J | 57 | 2 | 40 | 36 | 59 | 5 |
| 4/sliding | supplement | zs:J_dec | 0 | 0 | 100 | 64 | 3 | 32 |
| 4/sliding | supplement | zs:J | 54 | 1 | 45 | 33 | 61 | 6 |
| 5/single_force | retained | zs:J_dec | 0 | 0 | 100 | 89 | 2 | 8 |
| 5/single_force | retained | zs:J | 52 | 0 | 48 | 44 | 54 | 2 |
| 5/multiple_forces | supplement | zs:J_dec | 0 | 0 | 100 | 87 | 3 | 10 |
| 5/multiple_forces | supplement | zs:J | 51 | 0 | 48 | 42 | 56 | 2 |
| 5/rolling | retained | zs:J_dec | 0 | 0 | 100 | 88 | 2 | 10 |
| 5/rolling | retained | zs:J | 55 | 2 | 44 | 47 | 52 | 1 |
| 5/falling | retained | zs:J_dec | 0 | 0 | 100 | 87 | 0 | 13 |
| 5/falling | retained | zs:J | 51 | 2 | 47 | 42 | 55 | 2 |
| 5/sliding | retained | zs:J_dec | 0 | 0 | 100 | 92 | 2 | 6 |
| 5/sliding | retained | zs:J | 54 | 0 | 46 | 50 | 49 | 1 |
| 6/single_force | retained | zs:J_dec | 0 | 0 | 100 | 87 | 2 | 11 |
| 6/single_force | retained | zs:J | 72 | 4 | 24 | 76 | 23 | 1 |
| 6/multiple_forces | supplement | zs:J_dec | 0 | 0 | 100 | 87 | 2 | 11 |
| 6/multiple_forces | supplement | zs:J | 74 | 3 | 22 | 77 | 22 | 1 |
| 6/rolling | retained | zs:J_dec | 0 | 0 | 100 | 90 | 2 | 8 |
| 6/rolling | retained | zs:J | 64 | 4 | 32 | 62 | 37 | 1 |
| 6/falling | retained | zs:J_dec | 0 | 0 | 100 | 87 | 2 | 10 |
| 6/falling | retained | zs:J | 71 | 3 | 26 | 73 | 26 | 2 |
| 6/sliding | retained | zs:J_dec | 0 | 0 | 100 | 88 | 3 | 9 |
| 6/sliding | retained | zs:J | 62 | 4 | 34 | 59 | 40 | 1 |
| 7/single_force | retained | zs:J_dec | 0 | 0 | 100 | 85 | 2 | 13 |
| 7/single_force | retained | zs:J | 51 | 0 | 49 | 49 | 43 | 8 |
| 7/multiple_forces | supplement | zs:J_dec | 0 | 0 | 100 | 85 | 2 | 13 |
| 7/multiple_forces | supplement | zs:J | 50 | 0 | 49 | 47 | 46 | 7 |
| 7/rolling | retained | zs:J_dec | 0 | 0 | 100 | 53 | 2 | 44 |
| 7/rolling | retained | zs:J | 39 | 0 | 60 | 24 | 53 | 23 |
| 7/falling | retained | zs:J_dec | 0 | 0 | 100 | 69 | 1 | 29 |
| 7/falling | retained | zs:J | 50 | 0 | 50 | 28 | 58 | 14 |
| 7/sliding | retained | zs:J_dec | 0 | 1 | 99 | 62 | 1 | 37 |
| 7/sliding | retained | zs:J | 41 | 0 | 59 | 22 | 61 | 17 |
| 8/single_force | supplement | zs:J_dec | 0 | 0 | 100 | 75 | 2 | 22 |
| 8/single_force | supplement | zs:J | 72 | 1 | 27 | 74 | 26 | 1 |
| 8/multiple_forces | supplement | zs:J_dec | 0 | 0 | 100 | 80 | 2 | 18 |
| 8/multiple_forces | supplement | zs:J | 71 | 3 | 26 | 72 | 27 | 2 |
| 8/rolling | supplement | zs:J_dec | 0 | 0 | 100 | 71 | 1 | 28 |
| 8/rolling | supplement | zs:J | 66 | 2 | 32 | 60 | 36 | 4 |
| 8/falling | supplement | zs:J_dec | 0 | 0 | 100 | 58 | 1 | 41 |
| 8/falling | supplement | zs:J | 62 | 1 | 36 | 56 | 38 | 6 |
| 8/sliding | supplement | zs:J_dec | 0 | 0 | 100 | 77 | 1 | 21 |
| 8/sliding | supplement | zs:J | 72 | 8 | 20 | 70 | 27 | 3 |

## Wrong-anchor control (right - wrong anchor AUC)

| population | zs:J | zs:J_dec | policy:J | policy:J_dec | predictor:J | predictor:J_dec | both:J | both:J_dec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N1 (#109 evaluation) | 0.133 [0.122, 0.145] | 0.146 [0.133, 0.159] | - | - | - | - | - | - |
| 0: normal (paired normal sides) | 0.056 [0.031, 0.079] | 0.060 [0.034, 0.085] | - | - | - | - | - | - |
| 1: appearance (pink pig/blocks) | 0.128 [0.078, 0.176] | 0.120 [0.053, 0.183] | 0.070 [0.020, 0.120] | 0.126 [0.056, 0.196] | 0.088 [0.017, 0.158] | 0.135 [0.045, 0.222] | 0.151 [0.060, 0.237] | 0.103 [0.029, 0.176] |
| 2: fan (bird push) | 0.047 [-0.008, 0.094] | 0.029 [-0.038, 0.088] | 0.022 [-0.043, 0.082] | 0.059 [-0.012, 0.123] | 0.036 [-0.032, 0.105] | 0.040 [-0.034, 0.109] | 0.025 [-0.041, 0.080] | 0.045 [-0.024, 0.106] |
| 3: stronger air turbulence | 0.038 [-0.020, 0.098] | -0.013 [-0.061, 0.031] | 0.054 [-0.029, 0.148] | 0.022 [-0.055, 0.105] | -0.013 [-0.402, 0.377] | 0.010 [-0.385, 0.415] | 0.120 [-0.228, 0.469] | 0.109 [-0.226, 0.445] |
| 4: magnet | 0.046 [-0.033, 0.120] | 0.033 [-0.065, 0.128] | 0.013 [-0.055, 0.071] | 0.007 [-0.089, 0.100] | 0.094 [0.013, 0.175] | 0.115 [0.007, 0.222] | 0.103 [-0.008, 0.213] | 0.098 [-0.002, 0.195] |
| 5: right slingshot (mirrored layout) | 0.035 [-0.038, 0.110] | 0.030 [-0.054, 0.119] | 0.035 [-0.051, 0.120] | 0.049 [-0.031, 0.133] | -0.016 [-0.076, 0.046] | 0.041 [-0.028, 0.107] | 0.084 [0.026, 0.143] | 0.071 [-0.003, 0.147] |
| 6: inverse gravity | 0.064 [0.015, 0.112] | 0.047 [0.008, 0.084] | 0.017 [-0.015, 0.053] | 0.029 [-0.003, 0.060] | -0.001 [-0.007, 0.006] | -0.014 [-0.035, 0.004] | -0.025 [-0.058, 0.007] | -0.004 [-0.028, 0.019] |
| 7: inverse air turbulence | -0.035 [-0.157, 0.075] | -0.014 [-0.147, 0.111] | -0.005 [-0.105, 0.088] | -0.057 [-0.172, 0.049] | 0.017 [-0.022, 0.055] | 0.041 [0.006, 0.073] | 0.032 [-0.011, 0.074] | 0.049 [0.012, 0.084] |
| novel 1-7 pooled | 0.046 [0.013, 0.075] | 0.038 [0.002, 0.071] | 0.026 [-0.002, 0.053] | 0.033 [-0.002, 0.065] | 0.033 [0.003, 0.064] | 0.056 [0.021, 0.090] | 0.062 [0.028, 0.096] | 0.061 [0.031, 0.092] |

## Per cell (J_dec AUC)

| cell | levels | zero-shot | policy-only | predictor-only | both | F_hindsight zero-shot |
| --- | --- | --- | --- | --- | --- | --- |
| 0/single_force | 26 | 0.515 [0.435, 0.594] | - | - | - | F-1-micro: 0.583 [0.492, 0.674] |
| 0/rolling | 33 | 0.695 [0.628, 0.759] | - | - | - | F-5-micro: 0.701 [0.633, 0.763] |
| 0/falling | 37 | 0.736 [0.673, 0.797] | - | - | - | F-15-continuous: 0.761 [0.701, 0.816] |
| 0/sliding | 39 | 0.612 [0.541, 0.680] | - | - | - | F-15-micro: 0.643 [0.580, 0.706] |
| 1/single_force | 5 | 0.397 [0.164, 0.629] | 0.397 [0.164, 0.629] | 0.416 [0.213, 0.619] | 0.474 [0.302, 0.645] | F-1-micro: 0.423 [0.142, 0.703] |
| 1/rolling | 4 | 0.793 [0.702, 0.880] | 0.807 [0.739, 0.880] | 0.717 [0.589, 0.844] | 0.689 [0.589, 0.789] | F-15-micro: 0.805 [0.708, 0.896] |
| 1/falling | 3 | 0.652 [0.400, 0.911] | 0.689 [0.400, 0.933] | 0.763 [0.467, 0.911] | 0.674 [0.422, 0.844] | F-15-micro: 0.659 [0.400, 0.911] |
| 1/sliding | 8 | 0.717 [0.606, 0.803] | 0.717 [0.606, 0.803] | 0.735 [0.597, 0.852] | 0.664 [0.545, 0.762] | F-1-continuous: 0.724 [0.588, 0.842] |
| 2/single_force | 1 | 0.111 [0.111, 0.111] | 0.111 [0.111, 0.111] | 0.156 [0.156, 0.156] | 0.178 [0.178, 0.178] | F-5-continuous: 0.111 [0.111, 0.111] |
| 2/rolling | 4 | 0.672 [0.500, 0.850] | 0.711 [0.556, 0.867] | 0.689 [0.511, 0.867] | 0.633 [0.406, 0.856] | F-15-continuous: 0.706 [0.544, 0.867] |
| 2/falling | 8 | 0.621 [0.567, 0.665] | 0.618 [0.564, 0.662] | 0.641 [0.577, 0.699] | 0.673 [0.646, 0.701] | F-15-micro: 0.626 [0.570, 0.668] |
| 2/sliding | 4 | 0.305 [0.152, 0.458] | 0.350 [0.175, 0.500] | 0.364 [0.193, 0.536] | 0.397 [0.282, 0.512] | F-1-continuous: 0.411 [0.231, 0.542] |
| 3/rolling | 4 | 0.691 [0.594, 0.797] | 0.715 [0.624, 0.822] | 0.564 [0.129, 1.000] | 0.655 [0.333, 0.978] | F-1-continuous: 0.855 [0.777, 0.933] |
| 3/falling | 2 | 0.478 [0.422, 0.533] | 0.456 [0.422, 0.489] | 0.289 [0.267, 0.311] | 0.367 [0.089, 0.644] | F-15-micro: 0.467 [0.467, 0.467] |
| 4/single_force | 8 | 0.541 [0.416, 0.661] | 0.505 [0.382, 0.622] | 0.503 [0.351, 0.636] | 0.496 [0.351, 0.630] | F-15-micro: 0.580 [0.450, 0.697] |
| 4/rolling | 8 | 0.581 [0.524, 0.640] | 0.536 [0.465, 0.612] | 0.694 [0.588, 0.790] | 0.686 [0.578, 0.782] | F-1-continuous: 0.709 [0.620, 0.799] |
| 4/falling | 2 | 0.870 [0.800, 0.940] | 0.859 [0.778, 0.940] | 0.938 [0.911, 0.964] | 0.831 [0.750, 0.911] | F-15-micro: 0.881 [0.822, 0.940] |
| 5/single_force | 2 | 0.447 [0.405, 0.489] | 0.465 [0.440, 0.489] | 0.600 [0.533, 0.667] | 0.644 [0.622, 0.667] | F-1-macro: 0.652 [0.571, 0.733] |
| 5/rolling | 5 | 0.678 [0.560, 0.769] | 0.690 [0.590, 0.770] | 0.728 [0.618, 0.844] | 0.670 [0.565, 0.770] | F-1-macro: 0.737 [0.711, 0.762] |
| 5/falling | 6 | 0.828 [0.757, 0.904] | 0.835 [0.763, 0.907] | 0.774 [0.698, 0.848] | 0.765 [0.674, 0.861] | F-15-continuous: 0.833 [0.763, 0.907] |
| 5/sliding | 4 | 0.222 [0.083, 0.333] | 0.267 [0.200, 0.333] | 0.528 [0.394, 0.644] | 0.622 [0.394, 0.850] | F-1-macro: 0.350 [0.250, 0.400] |
| 6/single_force | 7 | 0.437 [0.316, 0.572] | 0.405 [0.278, 0.540] | 0.794 [0.699, 0.888] | 0.800 [0.705, 0.894] | F-15-macro: 0.465 [0.357, 0.585] |
| 6/rolling | 3 | 0.033 [0.014, 0.051] | 0.044 [0.021, 0.085] | 0.811 [0.750, 0.880] | 0.816 [0.764, 0.915] | F-1-micro: 0.129 [0.104, 0.162] |
| 6/falling | 6 | 0.428 [0.286, 0.598] | 0.417 [0.294, 0.549] | 0.796 [0.675, 0.903] | 0.785 [0.618, 0.905] | F-15-macro: 0.462 [0.309, 0.630] |
| 6/sliding | 6 | 0.267 [0.145, 0.457] | 0.252 [0.142, 0.441] | 0.770 [0.631, 0.891] | 0.782 [0.665, 0.885] | F-15-macro: 0.277 [0.137, 0.476] |
| 7/single_force | 6 | 0.115 [0.067, 0.204] | 0.133 [0.081, 0.215] | 0.963 [0.926, 0.996] | 0.930 [0.870, 0.985] | F-1-micro: 0.233 [0.181, 0.296] |
| 7/rolling | 5 | 0.554 [0.334, 0.774] | 0.557 [0.344, 0.774] | 0.613 [0.490, 0.753] | 0.570 [0.446, 0.715] | F-15-micro: 0.614 [0.423, 0.805] |
| 7/falling | 8 | 0.900 [0.858, 0.942] | 0.758 [0.694, 0.833] | 0.953 [0.922, 0.981] | 0.919 [0.886, 0.956] | F-5-continuous: 0.958 [0.942, 0.983] |
| 7/sliding | 3 | 0.536 [0.178, 0.786] | 0.632 [0.200, 0.933] | 0.684 [0.422, 0.867] | 0.699 [0.422, 0.867] | F-15-micro: 0.684 [0.311, 0.956] |

## Adaptation cost (GPU s, summed over levels 1-7 x 3 seeds; the shared encoder and scene-head adaptations once per level)

- predictor_member_gpu_seconds: 6339
- encoder_gpu_seconds: 829
- scene_head_gpu_seconds: 22
- policy_pe_gpu_seconds: 740
- policy_dec_gpu_seconds: 841
- both_policy_pe_gpu_seconds: 792
- both_policy_dec_gpu_seconds: 748
- parameters: policy 49417, LC hybrid predictor 2850731

## Compute: 7287 GPU s, 7287 process-wall s, 0 engine s (4 x RTX 5090 (one process per GPU))

| stage | GPU s | wall s |
| --- | --- | --- |
| assemble | 43 | 43 |
| data | 1049 | 1049 |
| score | 2442 | 2442 |
| smoke | 18 | 18 |
| train-fewshot | 3121 | 3121 |
| train-normal | 614 | 614 |

