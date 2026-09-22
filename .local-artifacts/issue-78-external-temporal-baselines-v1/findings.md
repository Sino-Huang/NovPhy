# Issue-78 external temporal-adaptation baselines - findings

Diagnostics complete: True. Claim boundary: mechanism-class comparison under one frozen contract; no claim of exact reproduction of TAWM/VLWM numbers on their original environments; descriptive intervals only; no inferential superiority claim; no new captures; not the #64/#65 sealed benchmark; prior dispositions (#15, #72, #74, #75, #77) unchanged.

Declared endpoint semantics: t=600 is the end-of-window replay cost on mostly right-censored branches; NOT a settled cost

Comparator disclosure: continuous_h5 retains issue-74 selection optimism (selected_continuous_policy=continuous_h5, selection_optimism=true, selected on development evidence); disclosed in every contrast that uses it. Inference is NOT equalized across arms; the full frontier is reported.

Ports: TAWM (arXiv 2506.08441; reference code consulted at pinned commit ffb61f8e2bcdb0030cb4a7175e0b782cdad9af4c, never run at its original scale) with declared deviations (h-conditioning on the frozen grid {1,5,15}, uniform mixed-h sampler at per-update granularity, existing learned horizon controller for the adaptive evaluation); VLWM (arXiv 2606.21775) as a paper-fidelity port of Eqs. 3-6, no reference code consulted (7-way no-code search in plan.json), with the single-shot mapping deviation (launch action + k-1 null actions). Structural equivalence disclosure: on this carrier the TAWM cell shares the #77 continuous reference cell's recipe (same seed-initialized carrier, minibatch stream, loss, optimizer, horizon counts) and differs only in within-cycle horizon ordering (round-robin 1,5,15 versus the reference's blocked 1,1,1,5,5,5,15,15,15); the tawm vs continuous contrasts therefore estimate horizon-schedule-ordering effects under an identical budget, and the vlwm arm carries the independent mechanism contrast.

## Headroom (frozen states)

| State | Corpus | Family | Candidates | Informative | Prior(ordinal09) regret | Uniform regret | Dropped |
| --- | --- | --- | --- | --- | --- | --- | --- |
| issue-77-n1-007 | issue-77-n1-diagnostic-v1 | type010103 | 13 | True | 0.4823 | 0.4758 | 0 |
| issue-77-n1-008 | issue-77-n1-diagnostic-v1 | type010103 | 13 | True | 0.8437 | 0.5893 | 0 |
| issue-77-n1-014 | issue-77-n1-diagnostic-v1 | type010105 | 13 | True | 0.1819 | 0.2665 | 0 |
| issue-77-n1-016 | issue-77-n1-diagnostic-v1 | type010105 | 13 | True | 0.4587 | 0.5123 | 0 |
| issue-76-bounded-transfer-032 | issue-77-n2-eval-v1 | type010101 | 13 | True | 0.9081 | 0.7310 | 0 |
| issue-76-bounded-transfer-033 | issue-77-n2-eval-v1 | type010101 | 13 | True | 0.2595 | 0.4690 | 0 |
| issue-76-bounded-transfer-034 | issue-77-n2-eval-v1 | type010101 | 13 | True | 0.4108 | 0.5077 | 0 |
| issue-76-bounded-transfer-035 | issue-77-n2-eval-v1 | type010101 | 13 | True | 0.9604 | 0.8484 | 0 |
| issue-76-bounded-transfer-036 | issue-77-n2-eval-v1 | type010101 | 13 | True | 0.5888 | 0.6662 | 0 |
| issue-76-bounded-transfer-037 | issue-77-n2-eval-v1 | type010101 | 13 | True | 0.8694 | 0.5905 | 0 |
| issue-76-bounded-transfer-038 | issue-77-n2-eval-v1 | type010101 | 13 | True | 0.9151 | 0.7284 | 0 |
| issue-76-bounded-transfer-039 | issue-77-n2-eval-v1 | type010101 | 13 | True | 0.7292 | 0.6169 | 0 |
| issue-77-n2-007 | issue-77-n2-eval-v1 | type010101 | 13 | True | 0.0946 | 0.2812 | 0 |
| issue-77-n2-008 | issue-77-n2-eval-v1 | type010101 | 13 | True | 0.2152 | 0.1279 | 0 |
| issue-77-n2n-001 | issue-77-n2-eval-v1 | type010102 | 6 | True | 0.8357 | 0.7247 | 7 |
| issue-77-n2n-002 | issue-77-n2-eval-v1 | type010102 | 4 | True | 0.0358 | 0.2659 | 9 |
| issue-77-n2n-005 | issue-77-n2-eval-v1 | type010102 | 11 | True | 0.2080 | 0.4765 | 2 |
| issue-77-n2n-007 | issue-77-n2-eval-v1 | type010102 | 4 | True | 0.0000 | 0.6508 | 9 |
| issue-77-n2n-008 | issue-77-n2-eval-v1 | type010102 | 9 | True | 0.4732 | 0.4996 | 4 |
| issue-77-n2-014 | issue-77-n2-eval-v1 | type010102 | 12 | True | 0.7128 | 0.6362 | 1 |
| issue-77-n2-016 | issue-77-n2-eval-v1 | type010102 | 13 | True | 0.8063 | 0.7069 | 0 |

## Action ranking (mean over the frozen states; seeds listed individually)

| System | Seed | Mean regret | Top1 | Top3 | Prediction failure states |
| --- | --- | --- | --- | --- | --- |
| tawm_h1 | 20260908 | 0.3998 | 0.190 | 0.429 | 0 |
| tawm_h5 | 20260908 | 0.3998 | 0.190 | 0.429 | 0 |
| tawm_h15 | 20260908 | 0.6500 | 0.048 | 0.429 | 0 |
| vlwm_h1 | 20260908 | 0.4689 | 0.190 | 0.429 | 0 |
| vlwm_h5 | 20260908 | 0.4474 | 0.143 | 0.429 | 0 |
| vlwm_h15 | 20260908 | 0.5365 | 0.190 | 0.429 | 0 |
| tawm_adaptive | 20260908 | 0.3998 | 0.190 | 0.429 | 0 |
| tawm_h1 | 20260909 | 0.3998 | 0.190 | 0.429 | 0 |
| tawm_h5 | 20260909 | 0.3998 | 0.190 | 0.429 | 0 |
| tawm_h15 | 20260909 | 0.5895 | 0.095 | 0.238 | 0 |
| vlwm_h1 | 20260909 | 0.4112 | 0.190 | 0.381 | 0 |
| vlwm_h5 | 20260909 | 0.4021 | 0.190 | 0.381 | 0 |
| vlwm_h15 | 20260909 | 0.4854 | 0.190 | 0.381 | 0 |
| tawm_adaptive | 20260909 | 0.3998 | 0.190 | 0.429 | 0 |
| tawm_h1 | 20260910 | 0.3998 | 0.190 | 0.429 | 0 |
| tawm_h5 | 20260910 | 0.3998 | 0.190 | 0.429 | 0 |
| tawm_h15 | 20260910 | 0.6358 | 0.095 | 0.190 | 0 |
| vlwm_h1 | 20260910 | 0.3998 | 0.190 | 0.429 | 0 |
| vlwm_h5 | 20260910 | 0.3998 | 0.190 | 0.429 | 0 |
| vlwm_h15 | 20260910 | 0.4353 | 0.143 | 0.381 | 0 |
| tawm_adaptive | 20260910 | 0.3998 | 0.190 | 0.429 | 0 |

## Paired contrasts (positive favors the tested system; DESCRIPTIVE)

| Tested | Reference | Kind | Mean regret difference | Descriptive 95% interval |
| --- | --- | --- | --- | --- |
| tawm_h1 | continuous_h1 | external_vs_fixed | +0.0669 | [+0.0183, +0.1206] |
| tawm_h1 | continuous_h1 | external_vs_fixed | +0.0810 | [+0.0000, +0.2051] |
| tawm_h1 | continuous_h1 | external_vs_fixed | +0.0636 | [+0.0106, +0.1229] |
| tawm_h5 | continuous_h5 | external_vs_fixed | +0.0664 | [-0.0220, +0.1700] |
| tawm_h5 | continuous_h5 | external_vs_fixed | +0.0325 | [+0.0000, +0.0976] |
| tawm_h5 | continuous_h5 | external_vs_fixed | +0.0743 | [-0.0333, +0.1979] |
| tawm_h15 | continuous_h15 | external_vs_fixed | +0.0021 | [-0.0975, +0.1024] |
| tawm_h15 | continuous_h15 | external_vs_fixed | +0.0905 | [-0.0624, +0.2517] |
| tawm_h15 | continuous_h15 | external_vs_fixed | -0.0187 | [-0.1355, +0.1023] |
| vlwm_h1 | continuous_h1 | external_vs_fixed | +0.0400 | [-0.0283, +0.1079] |
| vlwm_h1 | continuous_h1 | external_vs_fixed | +0.0810 | [+0.0000, +0.2051] |
| vlwm_h1 | continuous_h1 | external_vs_fixed | +0.0304 | [-0.0498, +0.1073] |
| vlwm_h5 | continuous_h5 | external_vs_fixed | +0.0497 | [-0.0282, +0.1394] |
| vlwm_h5 | continuous_h5 | external_vs_fixed | +0.0325 | [+0.0000, +0.0976] |
| vlwm_h5 | continuous_h5 | external_vs_fixed | +0.0538 | [-0.0430, +0.1609] |
| vlwm_h15 | continuous_h15 | external_vs_fixed | +0.1415 | [-0.0144, +0.2855] |
| vlwm_h15 | continuous_h15 | external_vs_fixed | +0.2481 | [+0.0318, +0.4644] |
| vlwm_h15 | continuous_h15 | external_vs_fixed | +0.1164 | [-0.0568, +0.2831] |
| tawm_h1 | tawm_h15 | decay_signature | +0.2253 | [+0.0816, +0.3716] |
| tawm_h1 | tawm_h15 | decay_signature | +0.2576 | [-0.0703, +0.6385] |
| tawm_h1 | tawm_h15 | decay_signature | +0.2177 | [+0.0611, +0.3712] |
| vlwm_h1 | vlwm_h15 | decay_signature | +0.0591 | [-0.0251, +0.1482] |
| vlwm_h1 | vlwm_h15 | decay_signature | +0.1000 | [-0.0072, +0.2073] |
| vlwm_h1 | vlwm_h15 | decay_signature | +0.0494 | [-0.0524, +0.1537] |
| continuous_h1 | continuous_h15 | decay_signature_reference | +0.1605 | [-0.0125, +0.3256] |
| continuous_h1 | continuous_h15 | decay_signature_reference | +0.2671 | [+0.0593, +0.4270] |
| continuous_h1 | continuous_h15 | decay_signature_reference | +0.1354 | [-0.0657, +0.3336] |
| tawm_adaptive | continuous_h5 | work_reported_comparison | +0.0664 | [-0.0220, +0.1700] |
| tawm_adaptive | continuous_h5 | work_reported_comparison | +0.0325 | [+0.0000, +0.0976] |
| tawm_adaptive | continuous_h5 | work_reported_comparison | +0.0743 | [-0.0333, +0.1979] |
| tawm_h1 | continuous_h5 | work_reported_comparison | +0.0664 | [-0.0220, +0.1700] |
| tawm_h1 | continuous_h5 | work_reported_comparison | +0.0325 | [+0.0000, +0.0976] |
| tawm_h1 | continuous_h5 | work_reported_comparison | +0.0743 | [-0.0333, +0.1979] |
| tawm_h15 | continuous_h5 | work_reported_comparison | -0.1590 | [-0.2917, -0.0329] |
| tawm_h15 | continuous_h5 | work_reported_comparison | -0.2251 | [-0.6385, +0.0703] |
| tawm_h15 | continuous_h5 | work_reported_comparison | -0.1434 | [-0.2761, -0.0112] |
| tawm_h5 | continuous_h5 | work_reported_comparison | +0.0664 | [-0.0220, +0.1700] |
| tawm_h5 | continuous_h5 | work_reported_comparison | +0.0325 | [+0.0000, +0.0976] |
| tawm_h5 | continuous_h5 | work_reported_comparison | +0.0743 | [-0.0333, +0.1979] |
| vlwm_h1 | continuous_h5 | work_reported_comparison | +0.0395 | [-0.0532, +0.1492] |
| vlwm_h1 | continuous_h5 | work_reported_comparison | +0.0325 | [+0.0000, +0.0976] |
| vlwm_h1 | continuous_h5 | work_reported_comparison | +0.0412 | [-0.0716, +0.1736] |
| vlwm_h15 | continuous_h5 | work_reported_comparison | -0.0195 | [-0.1228, +0.0898] |
| vlwm_h15 | continuous_h5 | work_reported_comparison | -0.0675 | [-0.2073, +0.0751] |
| vlwm_h15 | continuous_h5 | work_reported_comparison | -0.0083 | [-0.1300, +0.1198] |
| vlwm_h5 | continuous_h5 | work_reported_comparison | +0.0497 | [-0.0282, +0.1394] |
| vlwm_h5 | continuous_h5 | work_reported_comparison | +0.0325 | [+0.0000, +0.0976] |
| vlwm_h5 | continuous_h5 | work_reported_comparison | +0.0538 | [-0.0430, +0.1609] |
| tawm_h1 | continuous_h1 | training_effect_analog | +0.0669 | [+0.0183, +0.1206] |
| tawm_h1 | continuous_h1 | training_effect_analog | +0.0810 | [+0.0000, +0.2051] |
| tawm_h1 | continuous_h1 | training_effect_analog | +0.0636 | [+0.0106, +0.1229] |
| vlwm_h1 | continuous_h1 | training_effect_analog | +0.0400 | [-0.0283, +0.1079] |
| vlwm_h1 | continuous_h1 | training_effect_analog | +0.0810 | [+0.0000, +0.2051] |
| vlwm_h1 | continuous_h1 | training_effect_analog | +0.0304 | [-0.0498, +0.1073] |

## Work-reported frontier (inference NOT equalized)

| System | Parameters | Controller params | Mean per-state wall s | Linear MACs (seed mean) | Transition calls (seed mean) | Candidates/seed |
| --- | --- | --- | --- | --- | --- | --- |
| tawm_h1 | 1862396 | 32229 | 4.434 | 267828120000 | 144600 | 241 |
| tawm_h5 | 1862396 | 32229 | 0.909 | 53565624000 | 28920 | 241 |
| tawm_h15 | 1862396 | 32229 | 0.326 | 17855208000 | 9640 | 241 |
| vlwm_h1 | 1862396 | 0 | 4.431 | 267828120000 | 144600 | 241 |
| vlwm_h5 | 1862396 | 0 | 0.911 | 53565624000 | 28920 | 241 |
| vlwm_h15 | 1862396 | 0 | 0.325 | 17855208000 | 9640 | 241 |
| tawm_adaptive | 1862396 | 32229 | 3.720 | 165230059960 | 87688 | 241 |
| continuous_h5 (frozen #77 records) | #77 checkpoints | #77 checkpoints | 0.971 | 53565624000 | 28920 | 241 |

Wall provenance: reference-arm walls come from frozen #77 records measured 2026-09-20; baseline walls were measured under this ticket's exclusive GPU lock

## Per-corpus contrasts (normal-mechanics N1 vs novelty N2; DESCRIPTIVE)

| Corpus | Tested | Reference | Kind | Mean difference | Descriptive 95% interval |
| --- | --- | --- | --- | --- | --- |
| N1 normal-mechanics | tawm_h1 | continuous_h1 | external_vs_fixed | +0.0810 | [+0.0000, +0.2051] |
| N2 novelty | tawm_h1 | continuous_h1 | external_vs_fixed | +0.0636 | [+0.0106, +0.1229] |
| N1 normal-mechanics | tawm_h5 | continuous_h5 | external_vs_fixed | +0.0325 | [+0.0000, +0.0976] |
| N2 novelty | tawm_h5 | continuous_h5 | external_vs_fixed | +0.0743 | [-0.0333, +0.1979] |
| N1 normal-mechanics | tawm_h15 | continuous_h15 | external_vs_fixed | +0.0905 | [-0.0624, +0.2517] |
| N2 novelty | tawm_h15 | continuous_h15 | external_vs_fixed | -0.0187 | [-0.1355, +0.1023] |
| N1 normal-mechanics | vlwm_h1 | continuous_h1 | external_vs_fixed | +0.0810 | [+0.0000, +0.2051] |
| N2 novelty | vlwm_h1 | continuous_h1 | external_vs_fixed | +0.0304 | [-0.0498, +0.1073] |
| N1 normal-mechanics | vlwm_h5 | continuous_h5 | external_vs_fixed | +0.0325 | [+0.0000, +0.0976] |
| N2 novelty | vlwm_h5 | continuous_h5 | external_vs_fixed | +0.0538 | [-0.0430, +0.1609] |
| N1 normal-mechanics | vlwm_h15 | continuous_h15 | external_vs_fixed | +0.2481 | [+0.0318, +0.4644] |
| N2 novelty | vlwm_h15 | continuous_h15 | external_vs_fixed | +0.1164 | [-0.0568, +0.2831] |
| N1 normal-mechanics | tawm_h1 | tawm_h15 | decay_signature | +0.2576 | [-0.0703, +0.6385] |
| N2 novelty | tawm_h1 | tawm_h15 | decay_signature | +0.2177 | [+0.0611, +0.3712] |
| N1 normal-mechanics | vlwm_h1 | vlwm_h15 | decay_signature | +0.1000 | [-0.0072, +0.2073] |
| N2 novelty | vlwm_h1 | vlwm_h15 | decay_signature | +0.0494 | [-0.0524, +0.1537] |
| N1 normal-mechanics | tawm_h1 | continuous_h1 | training_effect_analog | +0.0810 | [+0.0000, +0.2051] |
| N2 novelty | tawm_h1 | continuous_h1 | training_effect_analog | +0.0636 | [+0.0106, +0.1229] |
| N1 normal-mechanics | vlwm_h1 | continuous_h1 | training_effect_analog | +0.0810 | [+0.0000, +0.2051] |
| N2 novelty | vlwm_h1 | continuous_h1 | training_effect_analog | +0.0304 | [-0.0498, +0.1073] |

## Dispositions

- **q1_method_class: `readiness_or_precision_insufficient`**
- **q2_work_reported: `supported`**
- **q3_training_effect: `supported`**

Reference #77 hybrid contrasts (frozen, quoted for context): the h=1 training effect hybrid_continuous_h1 vs continuous_h1 = +0.0887 [+0.0155, +0.2118]; symbolic-execution effects at h=1 straddle zero (micro +0.0348 [-0.0102, +0.1147]; macro +0.0239 [-0.0199, +0.0938]). External temporal-only arms have no symbolic execution by construction.

## Limitations

- t=600 is the end-of-window replay cost on mostly right-censored branches, not a settled cost
- descriptive paired bootstrap intervals only; no inferential superiority claim
- inference is NOT equalized across arms; the full frontier is reported
- continuous_h5 comparator retains issue-74 selection optimism (selected on development evidence); disclosed in every contrast that uses it
- reference-arm walls come from frozen #77 records measured 2026-09-20; baseline walls were measured under this ticket's exclusive GPU lock
- linear MACs, not full FLOPs
- the adaptive arm's recursive curve is captured sparsely (transition boundaries only); its t=600 task endpoint is exact
