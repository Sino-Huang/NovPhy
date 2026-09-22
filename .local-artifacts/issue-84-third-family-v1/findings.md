# Issue-84 third-family replication (Physion-Dominoes) - findings

Diagnostics complete: True. Claim boundary: mechanism-shape replication only, either direction is a valid publishable outcome; a third non-replication does not falsify the NovPhy boundary and a replication does not confirm it; the three families (NovPhy #77, CLEVRER #79, Physion-Dominoes #84) are reported separately and no prior published verdict is recomputed or amended; no leaderboard/VQA/perception claim; no new NovPhy captures; no sealed/final lineage access; #64/#65 remain unauthorized; descriptive intervals only

Endpoint semantics: released dominoes-trial frames at dt=0.01 s; recursive fixed-pair rollouts from each eval context; no planning/ranking-regret estimand and no candidate inventory; macro arm unsupported (Phase 0), estimand dropped.

## Phase-0 family selection (data readiness only)

- Selected: Physion scenario family Dominoes (dynamics-training HDF5 release) - all five criteria verified.
- Rejected: ShapeStacks (ECCV 2018, Groth et al.) - failed criterion (i); release has no per-frame object-state time series (and the release server was unreachable at Phase-0 time).

## Membership

| Trial | Frames | Objects | Collision events | Windows |
| --- | --- | --- | --- | --- |
| pilot_dominoes_SJ020_d3chairs_o1plants_tdwroom_0032 | 151 | 10 | 334 | 40 |
| pilot_dominoes_SJ020_d3chairs_o1plants_tdwroom_0156 | 207 | 10 | 425 | 40 |
| pilot_dominoes_1mid_J025R45_o1full_tdwroom_0118 | 170 | 5 | 269 | 40 |
| pilot_dominoes_4midRM1_tdwroom_2_0040 | 151 | 6 | 73 | 40 |
| pilot_dominoes_4mid_boxroom_0020 | 242 | 7 | 780 | 40 |
| pilot_dominoes_default_boxroom_0009 | 178 | 6 | 462 | 40 |
| pilot_dominoes_1mid_J025R45_o1full_tdwroom_0153 | 151 | 5 | 157 | 40 |
| pilot_dominoes_default_boxroom_0126 | 182 | 6 | 280 | 40 |
| pilot_dominoes_1mid_J025R45_o1full_tdwroom_0037 | 151 | 5 | 105 | 40 |
| pilot_dominoes_default_boxroom_0042 | 151 | 6 | 290 | 40 |
| pilot_dominoes_SJ020_d3chairs_o1plants_tdwroom_0079 | 211 | 10 | 345 | 40 |
| pilot_dominoes_4midRM1_tdwroom_0036 | 170 | 6 | 359 | 40 |

## Recursive position MSE (mean over units; seeds listed individually)

| System | Seed | e=15 | e=30 | e=60 | e=120 | Typed failures |
| --- | --- | --- | --- | --- | --- | --- |
| continuous_h1 | 20260908 | 0.0073 | 0.0362 | 0.8520 | 345.1289 | 0 |
| continuous_h5 | 20260908 | 0.0005 | 0.0018 | 0.0061 | 0.0236 | 0 |
| continuous_h15 | 20260908 | 0.0000 | 0.0001 | 0.0005 | 0.0018 | 0 |
| hybrid_continuous_h1 | 20260908 | 0.0029 | 0.0159 | 0.2294 | 90.9424 | 0 |
| hybrid_continuous_h5 | 20260908 | 0.0001 | 0.0004 | 0.0016 | 0.0085 | 0 |
| hybrid_continuous_h15 | 20260908 | 0.0001 | 0.0001 | 0.0002 | 0.0007 | 0 |
| hybrid_micro_h1 | 20260908 | 0.0073 | 0.0500 | 1.0564 | 118.5686 | 0 |
| hybrid_micro_h5 | 20260908 | 0.0002 | 0.0008 | 0.0039 | 0.0260 | 0 |
| hybrid_micro_h15 | 20260908 | 0.0031 | 0.0003 | 0.0005 | 0.0016 | 0 |
| continuous_h1 | 20260909 | 0.0029 | 0.0209 | 0.3807 | 64.2916 | 0 |
| continuous_h5 | 20260909 | 0.0001 | 0.0005 | 0.0015 | 0.0091 | 0 |
| continuous_h15 | 20260909 | 0.0000 | 0.0001 | 0.0001 | 0.0004 | 0 |
| hybrid_continuous_h1 | 20260909 | 0.0057 | 0.0362 | 0.5586 | 137.2961 | 0 |
| hybrid_continuous_h5 | 20260909 | 0.0002 | 0.0007 | 0.0032 | 0.0178 | 0 |
| hybrid_continuous_h15 | 20260909 | 0.0000 | 0.0001 | 0.0003 | 0.0012 | 0 |
| hybrid_micro_h1 | 20260909 | 0.0076 | 0.0783 | 0.6444 | 252.6640 | 0 |
| hybrid_micro_h5 | 20260909 | 0.0004 | 0.0009 | 0.0046 | 0.0380 | 0 |
| hybrid_micro_h15 | 20260909 | 0.0025 | 0.0004 | 0.0006 | 0.0020 | 0 |
| continuous_h1 | 20260910 | 0.0047 | 0.0389 | 0.7054 | 138.3809 | 0 |
| continuous_h5 | 20260910 | 0.0003 | 0.0010 | 0.0034 | 0.0203 | 0 |
| continuous_h15 | 20260910 | 0.0001 | 0.0002 | 0.0004 | 0.0012 | 0 |
| hybrid_continuous_h1 | 20260910 | 0.0062 | 0.0409 | 1.2748 | 273.8525 | 0 |
| hybrid_continuous_h5 | 20260910 | 0.0003 | 0.0011 | 0.0040 | 0.0205 | 0 |
| hybrid_continuous_h15 | 20260910 | 0.0001 | 0.0002 | 0.0006 | 0.0019 | 0 |
| hybrid_micro_h1 | 20260910 | 0.0125 | 0.0808 | 0.8713 | 68.6988 | 0 |
| hybrid_micro_h5 | 20260910 | 0.0006 | 0.0014 | 0.0066 | 0.0392 | 0 |
| hybrid_micro_h15 | 20260910 | 0.0021 | 0.0004 | 0.0007 | 0.0023 | 0 |

## Paired contrasts on position MSE (positive favors the tested system; DESCRIPTIVE)

| Tested | Reference | Kind | h | Endpoint | Difference | Descriptive 95% interval |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid_continuous_h1 | continuous_h1 | training_effect | 1 | 15 | +0.0001 | [-0.0001, +0.0002] |
| hybrid_continuous_h1 | continuous_h1 | training_effect | 1 | 30 | +0.0010 | [-0.0003, +0.0022] |
| hybrid_continuous_h1 | continuous_h1 | training_effect | 1 | 60 | -0.0416 | [-0.0720, -0.0136] |
| hybrid_continuous_h1 | continuous_h1 | training_effect | 1 | 120 | +15.2368 | [+3.8383, +25.9710] |
| hybrid_continuous_h5 | continuous_h5 | training_effect | 5 | 15 | +0.0001 | [+0.0001, +0.0001] |
| hybrid_continuous_h5 | continuous_h5 | training_effect | 5 | 30 | +0.0004 | [+0.0003, +0.0004] |
| hybrid_continuous_h5 | continuous_h5 | training_effect | 5 | 60 | +0.0007 | [+0.0006, +0.0008] |
| hybrid_continuous_h5 | continuous_h5 | training_effect | 5 | 120 | +0.0020 | [+0.0015, +0.0025] |
| hybrid_continuous_h15 | continuous_h15 | training_effect | 15 | 15 | -0.0000 | [-0.0000, -0.0000] |
| hybrid_continuous_h15 | continuous_h15 | training_effect | 15 | 30 | -0.0000 | [-0.0000, -0.0000] |
| hybrid_continuous_h15 | continuous_h15 | training_effect | 15 | 60 | -0.0000 | [-0.0001, -0.0000] |
| hybrid_continuous_h15 | continuous_h15 | training_effect | 15 | 120 | -0.0001 | [-0.0002, -0.0001] |
| hybrid_micro_h1 | hybrid_continuous_h1 | symbolic_execution | 1 | 15 | -0.0042 | [-0.0046, -0.0038] |
| hybrid_micro_h1 | hybrid_continuous_h1 | symbolic_execution | 1 | 30 | -0.0387 | [-0.0411, -0.0364] |
| hybrid_micro_h1 | hybrid_continuous_h1 | symbolic_execution | 1 | 60 | -0.1698 | [-0.2493, -0.0882] |
| hybrid_micro_h1 | hybrid_continuous_h1 | symbolic_execution | 1 | 120 | +20.7198 | [+1.8540, +38.9578] |
| hybrid_micro_h5 | hybrid_continuous_h5 | symbolic_execution | 5 | 15 | -0.0002 | [-0.0002, -0.0002] |
| hybrid_micro_h5 | hybrid_continuous_h5 | symbolic_execution | 5 | 30 | -0.0003 | [-0.0004, -0.0002] |
| hybrid_micro_h5 | hybrid_continuous_h5 | symbolic_execution | 5 | 60 | -0.0021 | [-0.0023, -0.0018] |
| hybrid_micro_h5 | hybrid_continuous_h5 | symbolic_execution | 5 | 120 | -0.0188 | [-0.0198, -0.0180] |
| hybrid_micro_h15 | hybrid_continuous_h15 | symbolic_execution | 15 | 15 | -0.0025 | [-0.0028, -0.0023] |
| hybrid_micro_h15 | hybrid_continuous_h15 | symbolic_execution | 15 | 30 | -0.0002 | [-0.0003, -0.0002] |
| hybrid_micro_h15 | hybrid_continuous_h15 | symbolic_execution | 15 | 60 | -0.0002 | [-0.0003, -0.0002] |
| hybrid_micro_h15 | hybrid_continuous_h15 | symbolic_execution | 15 | 120 | -0.0007 | [-0.0008, -0.0006] |

## Contact-activity regime of the micro symbolic-execution effect (DESCRIPTIVE)

| h | Endpoint | Active units | Active difference | Still units | Still difference | Active-minus-still |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 15 | 35 | -0.0037 | 13 | -0.0054 | +0.0017 |
| 1 | 30 | 35 | -0.0426 | 13 | -0.0281 | -0.0145 |
| 1 | 60 | 35 | -0.3095 | 13 | +0.2062 | -0.5157 |
| 1 | 120 | 35 | +54.7996 | 13 | -71.0334 | +125.8330 |
| 5 | 15 | 35 | -0.0002 | 13 | -0.0003 | +0.0001 |
| 5 | 30 | 35 | -0.0002 | 13 | -0.0006 | +0.0004 |
| 5 | 60 | 35 | -0.0018 | 13 | -0.0028 | +0.0011 |
| 5 | 120 | 35 | -0.0186 | 13 | -0.0196 | +0.0010 |
| 15 | 15 | 35 | -0.0023 | 13 | -0.0031 | +0.0008 |
| 15 | 30 | 35 | -0.0002 | 13 | -0.0002 | +0.0000 |
| 15 | 60 | 35 | -0.0002 | 13 | -0.0003 | +0.0002 |
| 15 | 120 | 35 | -0.0006 | 13 | -0.0010 | +0.0005 |

## Dispositions

- Q1 signature replication: **not_supported_by_this_experiment**
- Q2 contact-activity regime: **supported**

Three families reported separately: a Physion-Dominoes outcome neither falsifies nor confirms the NovPhy (#77) or CLEVRER (#79) published verdicts.

## Limitations

- third-family replication: mechanism-shape evidence only, no leaderboard/VQA or perception claim; the three families are reported separately and none falsifies another
- engine-annotation features in, no visual perception on either arm; shared frozen parser contract
- descriptive paired bootstrap over 48 (trial, context) units per seed; no inferential superiority claim
- inference and symbolic-head work are not equalized across arms; recorded, not matched
- macro arm unsupported on Physion-Dominoes; its estimand dropped (Phase 0)
- carrier positions are engine transform origins and velocities are center-of-mass readouts; both carried verbatim under their published semantics
- frozen lower speed-tier edge is 0.0 on this family, so the velocity-bin micro channel is an at-rest/moving indicator; disclosed, not retuned
