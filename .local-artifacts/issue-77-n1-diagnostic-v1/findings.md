# Issue-77 N1 dynamics diagnostic - findings

Diagnostics complete: True. Claim boundary: normal-mechanics breadth only; zero-shot/few-shot novelty evaluation is stage N2.

Declared endpoint semantics: observed-frame units (1 frame = 50 native steps); t=600 is end-of-window, right-censored branches included, NOT a settled cost

## Headroom

| State | Family | Candidates | Informative | Prior(ordinal09) regret | Uniform regret | Dropped |
| --- | --- | --- | --- | --- | --- | --- |
| issue-77-n1-007 | type010103 | 13 | True | 0.4823 | 0.4758 | 0 |
| issue-77-n1-008 | type010103 | 13 | True | 0.8437 | 0.5893 | 0 |
| issue-77-n1-014 | type010105 | 13 | True | 0.1819 | 0.2665 | 0 |
| issue-77-n1-016 | type010105 | 13 | True | 0.4587 | 0.5123 | 0 |

## Action ranking (mean over held-out states; seeds listed individually)

| System | Seed | Mean regret | Top1 | Top3 | Prediction failures |
| --- | --- | --- | --- | --- | --- |
| continuous_h1 | 20260908 | 0.3045 | 0.000 | 0.250 | 0 |
| continuous_h5 | 20260908 | 0.4021 | 0.000 | 0.250 | 0 |
| continuous_h15 | 20260908 | 0.6518 | 0.000 | 0.000 | 0 |
| hybrid_continuous_h1 | 20260908 | 0.3045 | 0.000 | 0.250 | 0 |
| hybrid_micro_h1 | 20260908 | 0.1865 | 0.250 | 0.500 | 0 |
| hybrid_macro_h1 | 20260908 | 0.3045 | 0.000 | 0.250 | 0 |
| hybrid_continuous_h5 | 20260908 | 0.4104 | 0.000 | 0.250 | 0 |
| hybrid_micro_h5 | 20260908 | 0.5060 | 0.000 | 0.250 | 0 |
| hybrid_macro_h5 | 20260908 | 0.4559 | 0.000 | 0.000 | 0 |
| hybrid_continuous_h15 | 20260908 | 0.7185 | 0.000 | 0.000 | 0 |
| hybrid_micro_h15 | 20260908 | 0.7379 | 0.000 | 0.250 | 0 |
| hybrid_macro_h15 | 20260908 | 0.4280 | 0.000 | 0.000 | 0 |
| continuous_h1 | 20260909 | 0.3045 | 0.000 | 0.250 | 0 |
| continuous_h5 | 20260909 | 0.3045 | 0.000 | 0.250 | 0 |
| continuous_h15 | 20260909 | 0.6860 | 0.000 | 0.000 | 0 |
| hybrid_continuous_h1 | 20260909 | 0.3045 | 0.000 | 0.250 | 0 |
| hybrid_micro_h1 | 20260909 | 0.2947 | 0.000 | 0.250 | 0 |
| hybrid_macro_h1 | 20260909 | 0.3045 | 0.000 | 0.250 | 0 |
| hybrid_continuous_h5 | 20260909 | 0.3045 | 0.000 | 0.250 | 0 |
| hybrid_micro_h5 | 20260909 | 0.3045 | 0.000 | 0.250 | 0 |
| hybrid_macro_h5 | 20260909 | 0.3933 | 0.000 | 0.250 | 0 |
| hybrid_continuous_h15 | 20260909 | 0.6201 | 0.000 | 0.250 | 0 |
| hybrid_micro_h15 | 20260909 | 0.6201 | 0.000 | 0.250 | 0 |
| hybrid_macro_h15 | 20260909 | 0.6416 | 0.000 | 0.250 | 0 |
| continuous_h1 | 20260910 | 0.5475 | 0.000 | 0.000 | 0 |
| continuous_h5 | 20260910 | 0.3045 | 0.000 | 0.250 | 0 |
| continuous_h15 | 20260910 | 0.6198 | 0.000 | 0.250 | 0 |
| hybrid_continuous_h1 | 20260910 | 0.2813 | 0.000 | 0.250 | 0 |
| hybrid_micro_h1 | 20260910 | 0.3045 | 0.000 | 0.250 | 0 |
| hybrid_macro_h1 | 20260910 | 0.2096 | 0.000 | 0.250 | 0 |
| hybrid_continuous_h5 | 20260910 | 0.3045 | 0.000 | 0.250 | 0 |
| hybrid_micro_h5 | 20260910 | 0.6417 | 0.000 | 0.000 | 0 |
| hybrid_macro_h5 | 20260910 | 0.3734 | 0.000 | 0.250 | 0 |
| hybrid_continuous_h15 | 20260910 | 0.6342 | 0.000 | 0.000 | 0 |
| hybrid_micro_h15 | 20260910 | 0.6262 | 0.000 | 0.000 | 0 |
| hybrid_macro_h15 | 20260910 | 0.6201 | 0.000 | 0.250 | 0 |

## Paired contrasts (positive favors the tested system)

| Tested | Reference | Kind | Mean regret difference | Descriptive 95% interval |
| --- | --- | --- | --- | --- |
| hybrid_continuous_h1 | continuous_h1 | training_effect | +0.0887 | [+0.0155, +0.2118] |
| hybrid_micro_h1 | hybrid_continuous_h1 | symbolic_execution | +0.0348 | [-0.0102, +0.1147] |
| hybrid_macro_h1 | hybrid_continuous_h1 | symbolic_execution | +0.0239 | [-0.0199, +0.0938] |
| hybrid_continuous_h5 | continuous_h5 | training_effect | -0.0028 | [-0.1333, +0.1708] |
| hybrid_micro_h5 | hybrid_continuous_h5 | symbolic_execution | -0.1443 | [-0.3414, +0.0024] |
| hybrid_macro_h5 | hybrid_continuous_h5 | symbolic_execution | -0.0677 | [-0.1480, +0.0074] |
| hybrid_continuous_h15 | continuous_h15 | training_effect | -0.0051 | [-0.1625, +0.1373] |
| hybrid_micro_h15 | hybrid_continuous_h15 | symbolic_execution | -0.0038 | [-0.1011, +0.0897] |
| hybrid_macro_h15 | hybrid_continuous_h15 | symbolic_execution | +0.0944 | [+0.0231, +0.1878] |

## Limitations

- shared semantically supervised CNN, not end-to-end symbol-free comparison
- t=600 is end-of-window cost on mostly right-censored branches, not settled cost
- at most four independent held-out states; bootstrap intervals are descriptive
- typed branch failures excluded from candidates and reported, never worst-cased
- linear MACs not full FLOPs or matched total work
