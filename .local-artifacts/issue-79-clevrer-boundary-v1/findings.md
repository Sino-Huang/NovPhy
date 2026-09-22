# Issue-79 CLEVRER boundary replication - findings

Diagnostics complete: True. Claim boundary: mechanism-shape replication only; not 'the negative result generalizes'; a non-replication does not falsify the NovPhy boundary and vice versa; the two families are reported separately; no CLEVRER leaderboard/VQA or perception claim; no Physion; no new NovPhy captures; #64/#65 sealed; prior dispositions unchanged; descriptive intervals only

Endpoint semantics: CLEVRER annotation frames; recursive fixed-pair rollouts from each eval context; no planning/ranking-regret estimand and no candidate inventory on CLEVRER.

## Membership

| Scene | Objects | Collisions | Windows |
| --- | --- | --- | --- |
| 10000 | 4 | 2 | 40 |
| 10001 | 5 | 3 | 40 |
| 10002 | 6 | 4 | 40 |
| 10003 | 4 | 2 | 40 |
| 10004 | 4 | 3 | 40 |
| 10005 | 5 | 2 | 40 |
| 10006 | 4 | 1 | 40 |
| 10007 | 6 | 2 | 40 |
| 10008 | 6 | 2 | 40 |
| 10009 | 4 | 3 | 40 |
| 10010 | 5 | 3 | 40 |
| 10011 | 5 | 3 | 40 |

## Recursive position MSE (mean over units; seeds listed individually)

| System | Seed | e=15 | e=30 | e=60 | e=120 | Typed failures |
| --- | --- | --- | --- | --- | --- | --- |
| continuous_h1 | 20260908 | 0.0133 | 0.0350 | 0.1420 | 11.6907 | 0 |
| continuous_h5 | 20260908 | 0.0016 | 0.0026 | 0.0050 | 0.0370 | 0 |
| continuous_h15 | 20260908 | 0.0013 | 0.0014 | 0.0009 | 0.0095 | 0 |
| hybrid_continuous_h1 | 20260908 | 0.0071 | 0.0494 | 0.1828 | 2.8006 | 0 |
| hybrid_continuous_h5 | 20260908 | 0.0013 | 0.0020 | 0.0046 | 0.0474 | 0 |
| hybrid_continuous_h15 | 20260908 | 0.0013 | 0.0016 | 0.0010 | 0.0191 | 0 |
| hybrid_micro_h1 | 20260908 | 0.0109 | 0.0625 | 0.2361 | 1.1852 | 0 |
| hybrid_micro_h5 | 20260908 | 0.0013 | 0.0020 | 0.0061 | 0.0476 | 0 |
| hybrid_micro_h15 | 20260908 | 0.0027 | 0.0017 | 0.0013 | 0.0192 | 0 |
| continuous_h1 | 20260909 | 0.0377 | 0.1747 | 0.6826 | 25.3873 | 0 |
| continuous_h5 | 20260909 | 0.0031 | 0.0069 | 0.0244 | 0.0634 | 0 |
| continuous_h15 | 20260909 | 0.0015 | 0.0019 | 0.0018 | 0.0120 | 0 |
| hybrid_continuous_h1 | 20260909 | 0.0183 | 0.0747 | 0.2465 | 2.5099 | 0 |
| hybrid_continuous_h5 | 20260909 | 0.0019 | 0.0077 | 0.0053 | 0.0537 | 0 |
| hybrid_continuous_h15 | 20260909 | 0.0016 | 0.0023 | 0.0012 | 0.0157 | 0 |
| hybrid_micro_h1 | 20260909 | 0.0179 | 0.0827 | 0.4319 | 6.3428 | 0 |
| hybrid_micro_h5 | 20260909 | 0.0020 | 0.0045 | 0.0091 | 0.0484 | 0 |
| hybrid_micro_h15 | 20260909 | 0.0026 | 0.0028 | 0.0017 | 0.0206 | 0 |
| continuous_h1 | 20260910 | 0.0095 | 0.0634 | 0.2877 | 3.5200 | 0 |
| continuous_h5 | 20260910 | 0.0009 | 0.0027 | 0.0049 | 0.0410 | 0 |
| continuous_h15 | 20260910 | 0.0011 | 0.0016 | 0.0012 | 0.0131 | 0 |
| hybrid_continuous_h1 | 20260910 | 0.0242 | 0.1027 | 0.4672 | 3.6363 | 0 |
| hybrid_continuous_h5 | 20260910 | 0.0026 | 0.0052 | 0.0093 | 0.0321 | 0 |
| hybrid_continuous_h15 | 20260910 | 0.0012 | 0.0016 | 0.0019 | 0.0163 | 0 |
| hybrid_micro_h1 | 20260910 | 0.0258 | 0.1045 | 0.4891 | 4.1230 | 0 |
| hybrid_micro_h5 | 20260910 | 0.0027 | 0.0069 | 0.0118 | 0.0469 | 0 |
| hybrid_micro_h15 | 20260910 | 0.0024 | 0.0019 | 0.0020 | 0.0173 | 0 |

## Paired contrasts on position MSE (positive favors the tested system; DESCRIPTIVE)

| Tested | Reference | Kind | h | Endpoint | Difference | Descriptive 95% interval |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid_continuous_h1 | continuous_h1 | training_effect | 1 | 15 | +0.0037 | [+0.0013, +0.0060] |
| hybrid_continuous_h1 | continuous_h1 | training_effect | 1 | 30 | +0.0155 | [+0.0031, +0.0256] |
| hybrid_continuous_h1 | continuous_h1 | training_effect | 1 | 60 | +0.0719 | [+0.0333, +0.1090] |
| hybrid_continuous_h1 | continuous_h1 | training_effect | 1 | 120 | +10.5504 | [+8.7690, +12.4330] |
| hybrid_continuous_h5 | continuous_h5 | training_effect | 5 | 15 | -0.0000 | [-0.0004, +0.0004] |
| hybrid_continuous_h5 | continuous_h5 | training_effect | 5 | 30 | -0.0009 | [-0.0028, +0.0005] |
| hybrid_continuous_h5 | continuous_h5 | training_effect | 5 | 60 | +0.0050 | [+0.0027, +0.0078] |
| hybrid_continuous_h5 | continuous_h5 | training_effect | 5 | 120 | +0.0027 | [-0.0003, +0.0052] |
| hybrid_continuous_h15 | continuous_h15 | training_effect | 15 | 15 | -0.0001 | [-0.0004, +0.0003] |
| hybrid_continuous_h15 | continuous_h15 | training_effect | 15 | 30 | -0.0002 | [-0.0005, +0.0002] |
| hybrid_continuous_h15 | continuous_h15 | training_effect | 15 | 60 | -0.0000 | [-0.0002, +0.0001] |
| hybrid_continuous_h15 | continuous_h15 | training_effect | 15 | 120 | -0.0055 | [-0.0074, -0.0039] |
| hybrid_micro_h1 | hybrid_continuous_h1 | symbolic_execution | 1 | 15 | -0.0017 | [-0.0034, +0.0002] |
| hybrid_micro_h1 | hybrid_continuous_h1 | symbolic_execution | 1 | 30 | -0.0077 | [-0.0138, -0.0017] |
| hybrid_micro_h1 | hybrid_continuous_h1 | symbolic_execution | 1 | 60 | -0.0869 | [-0.1154, -0.0580] |
| hybrid_micro_h1 | hybrid_continuous_h1 | symbolic_execution | 1 | 120 | -0.9014 | [-1.3407, -0.4539] |
| hybrid_micro_h5 | hybrid_continuous_h5 | symbolic_execution | 5 | 15 | -0.0001 | [-0.0002, +0.0000] |
| hybrid_micro_h5 | hybrid_continuous_h5 | symbolic_execution | 5 | 30 | +0.0005 | [-0.0003, +0.0015] |
| hybrid_micro_h5 | hybrid_continuous_h5 | symbolic_execution | 5 | 60 | -0.0026 | [-0.0036, -0.0016] |
| hybrid_micro_h5 | hybrid_continuous_h5 | symbolic_execution | 5 | 120 | -0.0032 | [-0.0076, +0.0010] |
| hybrid_micro_h15 | hybrid_continuous_h15 | symbolic_execution | 15 | 15 | -0.0012 | [-0.0015, -0.0009] |
| hybrid_micro_h15 | hybrid_continuous_h15 | symbolic_execution | 15 | 30 | -0.0003 | [-0.0005, -0.0001] |
| hybrid_micro_h15 | hybrid_continuous_h15 | symbolic_execution | 15 | 60 | -0.0003 | [-0.0005, -0.0002] |
| hybrid_micro_h15 | hybrid_continuous_h15 | symbolic_execution | 15 | 120 | -0.0020 | [-0.0030, -0.0008] |

## Contact-activity regime of the micro symbolic-execution effect (DESCRIPTIVE)

| h | Endpoint | Active units | Active difference | Still units | Still difference | Active-minus-still |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 15 | 24 | -0.0000 | 24 | -0.0033 | +0.0033 |
| 1 | 30 | 24 | -0.0069 | 24 | -0.0084 | +0.0015 |
| 1 | 60 | 24 | -0.0921 | 24 | -0.0818 | -0.0103 |
| 1 | 120 | 24 | -0.9552 | 24 | -0.8475 | -0.1077 |
| 5 | 15 | 24 | +0.0001 | 24 | -0.0002 | +0.0003 |
| 5 | 30 | 24 | +0.0012 | 24 | -0.0002 | +0.0015 |
| 5 | 60 | 24 | -0.0025 | 24 | -0.0026 | +0.0000 |
| 5 | 120 | 24 | -0.0070 | 24 | +0.0006 | -0.0076 |
| 15 | 15 | 24 | -0.0015 | 24 | -0.0009 | -0.0005 |
| 15 | 30 | 24 | -0.0005 | 24 | -0.0001 | -0.0004 |
| 15 | 60 | 24 | -0.0004 | 24 | -0.0003 | -0.0001 |
| 15 | 120 | 24 | -0.0030 | 24 | -0.0010 | -0.0020 |

## Dispositions

- Q1 signature replication: **not_supported_by_this_experiment**
- Q2 contact-activity regime: **not_supported_by_this_experiment**

Both families are reported separately: a CLEVRER non-replication does not falsify the NovPhy boundary, and vice versa.

## Limitations

- out-of-family replication: mechanism-shape evidence only, no leaderboard/VQA or perception claim
- engine-annotation features in, no visual perception on either arm; shared frozen parser contract
- descriptive paired bootstrap over 48 (scene, context) units per seed; no inferential superiority claim
- inference and symbolic-head work are not equalized across arms; recorded, not matched
- macro arm unsupported on CLEVRER; its estimand dropped (Phase 0)
