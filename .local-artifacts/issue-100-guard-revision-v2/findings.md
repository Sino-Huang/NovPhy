# Issue-100 v2: guard revision of the readout-fidelity / JEPA publication — findings

- identity `issue-100-guard-revision-v2`, revision of `issue-100-readout-fidelity-jepa-v1`; v2 frozen 2026-09-27T11:03:45Z, **after** the v1 publication
- validation command: `python -u -m scripts.run_readout_fidelity_guard_revision --validate` (runs the v1 `--validate` and re-derives every v2 table)
- disclosure: v2 was frozen after the v1 publication, with every v1 outcome known (all readout F1s, rollout errors, arm AUCs, contrasts and the v1 tokens). The v2 guard definitions come from the diagnoses of the two v1 guard failures (motion-value noise; cuDNN autotuning in the run process) and from the parse tolerance the v1 plan froze before any outcome; no F1, AUC or contrast entered their definition. v2 changes no estimand, readout, threshold, cohort, record or decision rule; the scored records stay the v1 records; v1 is retained byte-identical (sha256-bound below).

## Guards

| guard | v1 pass | v2 pass | revised | v2 observed |
|---|---|---|---|---|
| G1 | False | True | True | `{"max_abs_nonmotion_delta": 0.0007217824459075928, "max_motion_delta_times_elapsed_over_2": 1.5482306480407715e-05, "tolerance": 0.001, "v1_statistic_max_abs_delta": 0.03587867319583893, "windows_above_v1_tolerance": 18}` |
| G2 | True | True | False | `{"rows": [{"fresh": 773.8302905742938, "published": 773.74478325477, "relative_delta": 0.00011051101264180583, "seed": 20260908, "system": "hybrid_continuous_h1"}, {"fresh": 3893.468186598558, "published": 3893.706510103666, "relative_delta": 6.120736231390566e-05, "seed": 20260909, "system": "hybrid_continuous_h1"}, {"fresh": 4755.434934175932, "published": 4755.788769061749, "relative_delta": 7.` |
| G3 | False | True | True | `{"candidates_checked": 34242, "coordinates_equal": true, "max_abs_count_cost_delta_fresh_vs_96": 4.76837158203125e-07, "tolerance": 0.001, "v1_run_max_abs_count_cost_delta": 87.6999322772026}` |
| G4 | True | True | False | `{"controller_lineages": ["issue-77-n1-005", "issue-77-n1-010", "issue-77-n1-015"], "fit_members": ["issue-77-n1-001", "issue-77-n1-002", "issue-77-n1-003", "issue-77-n1-004", "issue-77-n1-005", "issue-77-n1-006", "issue-77-n1-009", "issue-77-n1-010", "issue-77-n1-011", "issue-77-n1-012", "issue-77-n1-013", "issue-77-n1-015"]}` |
| G5 | True | True | False | `{"controllers": 6, "jepa_final": [{"jepa_loss": 0.05497863945654697, "parser_loss": 0.13269282134456767, "step": 9000}, {"jepa_loss": 0.09990121208959156, "parser_loss": 0.3583387998243173, "step": 9000}, {"jepa_loss": 0.1037554090221723, "parser_loss": 0.07853459860715602, "step": 9000}], "jepa_steps": [9000, 9000, 9000]}` |
| G6 | True | True | False | `{"max_abs_delta": 2.384185791015625e-05, "tolerance": 0.0001}` |
| G7 | True | True | False | `{"held_out_engine_positives": {"contact": 25260.0, "steady-state": 25.0, "structure-unstable": 115.0, "supports": 25143.0}, "minimum": 20}` |
| caps | True | True | False | `{"gpu_seconds": 12203.429438068764, "status": "terminal", "wall_seconds": 13571.686308379984}` |

- **G1** v1: F77 audit carriers replicate every valid #77 shard carrier value within 0.001. v2: every non-motion carrier value within 0.001, and every motion value within 2 x 0.001 / elapsed seconds of its frame (a motion value is the difference of two parsed centers, each within the 0.001 parse tolerance, divided by the frame interval). Why: the guard protects 'the audit reads the #77 training carriers'; every v1 exceedance was a motion value, where center float noise from a different parse batch is divided by the frame interval (terminal frames of early-ending shots have intervals of a few milliseconds).
- **G3** v1: F77 arms replicate the #96 records (count cost of every candidate of every F/J/M/T/D record) within 0.001, and the F77 T/D coordinates equal #96's. v2: a fresh deterministic process (cudnn.benchmark off, cudnn.deterministic on, no prior GPU work) running the v1 arm code for F77 on every #96 cell reproduces every #96 count cost within 0.001, and the F77 T/D coordinates equal #96's (v1 observation, unchanged). Why: the guard protects 'these arms are the #96 arms'; the v1 run process had turned on cuDNN autotuning for JEPA training, so its anchor parses used differently rounded kernels and recursive Delta = 1 rollouts plus the count cost's 1000x pig weight amplified the noise. The v1 records stay the scored records; their divergence from #96 is reported descriptively.

## Dispositions

- **Q2 readout target (held-out, macro F1 >= 0.7 and micro edge F1 >= 0.5): v1 readiness_or_precision_insufficient → v2 not_supported_by_this_experiment**; met by none; better encoder E99

| encoder | macro F1 | micro edge F1 | meets |
|---|---|---|---|
| E99 | 0.0000 [0.0000, 0.0000] | 0.9886 [0.9718, 0.9995] | False |
| F77 | 0.0025 [0.0012, 0.0045] | 0.0000 [0.0000, 0.0000] | False |
| JEPA | 0.0000 [0.0000, 0.0000] | 0.8988 [0.8113, 0.9723] | False |

**Q4 (better encoder E99, cohort held_out, cost tie_free)**

| inventory | contrast | estimate | cells / members | v1 token | v2 token |
|---|---|---|---|---|---|
| angle | J-D | 0.0556 [-0.0278, 0.1389] | 6 / 2 | readiness_or_precision_insufficient | readiness_or_precision_insufficient |
| angle | J-T | 0.0694 [0.0278, 0.1111] | 6 / 2 | readiness_or_precision_insufficient | supported |
| grid | J-D | 0.2000 [0.1556, 0.2222] | 9 / 3 | readiness_or_precision_insufficient | supported |
| grid | J-T | 0.1630 [0.0889, 0.2444] | 9 / 3 | readiness_or_precision_insufficient | supported |
| offset | J-D | 0.1556 [0.1333, 0.1667] | 9 / 3 | readiness_or_precision_insufficient | supported |
| offset | J-T | 0.0222 [-0.0667, 0.1667] | 9 / 3 | readiness_or_precision_insufficient | readiness_or_precision_insufficient |
| power | J-D | 0.0117 [-0.1228, 0.0877] | 9 / 3 | readiness_or_precision_insufficient | readiness_or_precision_insufficient |
| power | J-T | 0.0526 [-0.1404, 0.2281] | 9 / 3 | readiness_or_precision_insufficient | readiness_or_precision_insufficient |

Limits: the held-out cohort has 2-3 mixed member clusters per inventory (a percentile bootstrap over 3 clusters has 10 distinct resamples); no multiplicity adjustment across the 8 contrasts.

## Noise impact of the v1 run records (F77 vs #96, count cost; descriptive)

| arm | mixed cells | cells whose AUC changed | max abs AUC change | mean abs AUC change | candidates above 0.001 / candidates |
|---|---|---|---|---|---|
| F-1-continuous | 114 | 2 | 0.4333 | 0.0040 | 85 / 1719 |
| F-1-micro | 114 | 1 | 0.0500 | 0.0004 | 73 / 1719 |
| F-1-macro | 114 | 5 | 0.1053 | 0.0023 | 209 / 1719 |
| F-5-continuous | 114 | 2 | 0.0833 | 0.0013 | 19 / 1719 |
| F-5-micro | 114 | 2 | 0.0833 | 0.0012 | 66 / 1719 |
| F-5-macro | 114 | 9 | 0.3000 | 0.0105 | 325 / 1719 |
| F-15-continuous | 114 | 0 | 0.0000 | 0.0000 | 499 / 1719 |
| F-15-micro | 114 | 0 | 0.0000 | 0.0000 | 485 / 1719 |
| F-15-macro | 114 | 0 | 0.0000 | 0.0000 | 875 / 1719 |
| J | 114 | 3 | 0.0833 | 0.0015 | 256 / 1719 |
| M | 114 | 5 | 0.1000 | 0.0030 | 217 / 1719 |
| T | 114 | 0 | 0.0000 | 0.0000 | 137 / 1719 |
| D | 114 | 6 | 0.2500 | 0.0053 | 274 / 1719 |

## Probe macro numbers (descriptive)

- claimed (2026-09-25 probe): steady 0.188, unstable 0.041; v1 (deployment carrier, prior frame 79): [0.6494, 0.1098]
- seed 20260908, single-frame carrier without prior: gated macro [0.1884, 0.0406]
- seed 20260909, single-frame carrier without prior: gated macro [0.1287, 0.0422]
- seed 20260910, single-frame carrier without prior: gated macro [0.1309, 0.0425]

Compute: replication 397 GPU s; engine seconds 0.

## Claim boundary

readout fidelity on retained #77 N1 captures of 16 normal-mechanics lineages; decision-only ranking of retained engine-verdicted executions; no engine access, rendering, capture, closed-loop or competence claim; every interval is DESCRIPTIVE; nothing here edits the ICLR 2026 submission.
