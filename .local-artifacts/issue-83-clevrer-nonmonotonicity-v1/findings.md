# Issue-83 CLEVRER h15 non-monotonicity decomposition - findings

Diagnostics complete: True. Claim boundary: decomposition and sensitivity of the frozen issue-79 rollouts only; no re-freeze, amendment, or reinterpretation of the issue-79 published dispositions; both families (NovPhy / CLEVRER) stay reported separately; no new physics family, no planning/ranking estimand, no perception claim, no leaderboard/VQA claim; descriptive intervals only; either localization answer (concentrated or uniform) is a valid outcome

Endpoint semantics: window-relative or context-relative CLEVRER annotation frames; recursive fixed-pair rollouts from the frozen initial carriers, no truth resets; the issue-79 published verdicts are inputs and are never recomputed or amended.

## Membership and frozen segment census

| Scene | Objects | Collision frames | Cascade bounds (first, last) |
| --- | --- | --- | --- |
| 10000 | 4 | [34, 84] | (34, 84) |
| 10001 | 5 | [34, 58, 76] | (34, 76) |
| 10002 | 6 | [49, 69, 108, 117] | (49, 117) |
| 10003 | 4 | [28, 106] | (28, 106) |
| 10004 | 4 | [18, 26, 50] | (18, 50) |
| 10005 | 5 | [31, 67] | (31, 67) |
| 10006 | 4 | [34] | (34, 34) |
| 10007 | 6 | [25, 39] | (25, 39) |
| 10008 | 6 | [37, 65] | (37, 65) |
| 10009 | 4 | [25, 78, 93] | (25, 93) |
| 10010 | 5 | [31, 47, 52] | (31, 52) |
| 10011 | 5 | [32, 86, 97] | (32, 97) |

| Grid endpoint e | pre_first_collision | active_cascade | post_settlement |
| --- | --- | --- | --- |
| 15 | 47 | 1 | 0 |
| 30 | 24 | 24 | 0 |
| 60 | 0 | 32 | 16 |
| 90 | 0 | 16 | 32 |
| 120 | 0 | 0 | 48 |

## E1: segment-conditional recursive position MSE, h=15, seed mean (units per point; DESCRIPTIVE)

### grid traces (contexts 0..3, frames up to context+120)

| Arm | Segment | e=15 | e=30 | e=45 | e=60 | e=75 | e=90 | e=105 | e=120 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuous | pre_first_collision | 0.00132 (47) | 0.00152 (24) | 0.00258 (4) | n/a | n/a | n/a | n/a | n/a |
| continuous | active_cascade | 0.00115 (1) | 0.00180 (24) | 0.00178 (36) | 0.00144 (32) | 0.00211 (22) | 0.00216 (16) | 0.00268 (6) | n/a |
| continuous | post_settlement | n/a | n/a | 0.00101 (8) | 0.00112 (16) | 0.00155 (26) | 0.00204 (32) | 0.00513 (42) | 0.01156 (48) |
| hybrid | pre_first_collision | 0.00138 (47) | 0.00167 (24) | 0.00179 (4) | n/a | n/a | n/a | n/a | n/a |
| hybrid | active_cascade | 0.00135 (1) | 0.00203 (24) | 0.00183 (36) | 0.00159 (32) | 0.00207 (22) | 0.00289 (16) | 0.00455 (6) | n/a |
| hybrid | post_settlement | n/a | n/a | 0.00083 (8) | 0.00083 (16) | 0.00177 (26) | 0.00274 (32) | 0.00732 (42) | 0.01703 (48) |

### window traces (starts 0..39, frames up to start+60)

| Arm | Segment | e=15 | e=30 | e=45 | e=60 |
| --- | --- | --- | --- | --- | --- |
| continuous | pre_first_collision | 0.00385 (198) | 0.00140 (42) | 0.00258 (4) | n/a |
| continuous | active_cascade | 0.00334 (241) | 0.00215 (331) | 0.00119 (286) | 0.00176 (208) |
| continuous | post_settlement | 0.00043 (41) | 0.00051 (107) | 0.00090 (190) | 0.00126 (272) |
| hybrid | pre_first_collision | 0.00452 (198) | 0.00139 (42) | 0.00179 (4) | n/a |
| hybrid | active_cascade | 0.00352 (241) | 0.00211 (331) | 0.00134 (286) | 0.00189 (208) |
| hybrid | post_settlement | 0.00078 (41) | 0.00097 (107) | 0.00152 (190) | 0.00208 (272) |

## E2: post-settlement vs pre/active ratio of masked position MSE (DESCRIPTIVE)

Ratio rho = post_settlement mean / pre+active mean over units at the scored frame; the frozen uniformity band is [0.80, 1.25]; rho < 0.80 at the decision endpoint means the dip is concentrated in post-settlement segments.

| Kind | Arm | h | e | rho | 95% interval | post units | pre/active units |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grid | continuous | 1 | 15 | n/a | n/a | 0 | 48 |
| grid | continuous | 1 | 30 | n/a | n/a | 0 | 48 |
| grid | continuous | 1 | 60 | 1.050 | [0.903, 1.219] | 16 | 32 |
| grid | continuous | 1 | 90 | 1.050 | [0.878, 1.263] | 32 | 16 |
| grid | continuous | 1 | 120 | n/a | n/a | 48 | 0 |
| grid | continuous | 5 | 15 | n/a | n/a | 0 | 48 |
| grid | continuous | 5 | 30 | n/a | n/a | 0 | 48 |
| grid | continuous | 5 | 60 | 0.826 | [0.604, 1.177] | 16 | 32 |
| grid | continuous | 5 | 90 | 1.236 | [1.039, 1.474] | 32 | 16 |
| grid | continuous | 5 | 120 | n/a | n/a | 48 | 0 |
| grid | continuous | 15 | 15 | n/a | n/a | 0 | 48 |
| grid | continuous | 15 | 30 | n/a | n/a | 0 | 48 |
| grid | continuous | 15 | 60 | 0.777 | [0.611, 0.975] | 16 | 32 |
| grid | continuous | 15 | 90 | 0.945 | [0.689, 1.318] | 32 | 16 |
| grid | continuous | 15 | 120 | n/a | n/a | 48 | 0 |
| grid | hybrid | 1 | 15 | n/a | n/a | 0 | 48 |
| grid | hybrid | 1 | 30 | n/a | n/a | 0 | 48 |
| grid | hybrid | 1 | 60 | 0.996 | [0.688, 1.365] | 16 | 32 |
| grid | hybrid | 1 | 90 | 0.670 | [0.570, 0.791] | 32 | 16 |
| grid | hybrid | 1 | 120 | n/a | n/a | 48 | 0 |
| grid | hybrid | 5 | 15 | n/a | n/a | 0 | 48 |
| grid | hybrid | 5 | 30 | n/a | n/a | 0 | 48 |
| grid | hybrid | 5 | 60 | 0.733 | [0.630, 0.854] | 16 | 32 |
| grid | hybrid | 5 | 90 | 0.831 | [0.725, 0.959] | 32 | 16 |
| grid | hybrid | 5 | 120 | n/a | n/a | 48 | 0 |
| grid | hybrid | 15 | 15 | n/a | n/a | 0 | 48 |
| grid | hybrid | 15 | 30 | n/a | n/a | 0 | 48 |
| grid | hybrid | 15 | 60 | 0.518 | [0.423, 0.627] | 16 | 32 |
| grid | hybrid | 15 | 90 | 0.950 | [0.745, 1.218] | 32 | 16 |
| grid | hybrid | 15 | 120 | n/a | n/a | 48 | 0 |
| window | continuous | 1 | 15 | 0.580 | [0.519, 0.650] | 41 | 439 |
| window | continuous | 1 | 30 | 0.752 | [0.709, 0.798] | 107 | 373 |
| window | continuous | 1 | 60 | 0.964 | [0.926, 1.003] | 272 | 208 |
| window | continuous | 5 | 15 | 0.213 | [0.149, 0.331] | 41 | 439 |
| window | continuous | 5 | 30 | 0.354 | [0.278, 0.459] | 107 | 373 |
| window | continuous | 5 | 60 | 0.837 | [0.761, 0.921] | 272 | 208 |
| window | continuous | 15 | 15 | 0.121 | [0.080, 0.198] | 41 | 439 |
| window | continuous | 15 | 30 | 0.247 | [0.176, 0.366] | 107 | 373 |
| window | continuous | 15 | 60 | 0.720 | [0.595, 0.878] | 272 | 208 |
| window | hybrid | 1 | 15 | 0.388 | [0.328, 0.459] | 41 | 439 |
| window | hybrid | 1 | 30 | 0.652 | [0.575, 0.736] | 107 | 373 |
| window | hybrid | 1 | 60 | 0.928 | [0.865, 0.994] | 272 | 208 |
| window | hybrid | 5 | 15 | 0.242 | [0.180, 0.344] | 41 | 439 |
| window | hybrid | 5 | 30 | 0.306 | [0.236, 0.409] | 107 | 373 |
| window | hybrid | 5 | 60 | 0.997 | [0.917, 1.083] | 272 | 208 |
| window | hybrid | 15 | 15 | 0.196 | [0.130, 0.320] | 41 | 439 |
| window | hybrid | 15 | 30 | 0.478 | [0.366, 0.638] | 107 | 373 |
| window | hybrid | 15 | 60 | 1.100 | [0.954, 1.271] | 272 | 208 |

## E3: endpoint-grid and mask sensitivity of the recursive position MSE (DESCRIPTIVE)

| Arm | h | Grid | Mask | e=15 | e=30 | e=60 | e=90 | e=120 | monotone increasing | first violation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuous | 1 | anchor | inside_camera_view | 0.02019 | 0.09103 | 0.37074 | n/a | 13.53265 | True | - |
| continuous | 1 | anchor | all_frames | 0.19438 | 0.18840 | 0.39930 | n/a | 13.81078 | False | 15->30 |
| continuous | 1 | fine | inside_camera_view | 0.02019 | 0.09103 | 0.37074 | 1.63280 | 13.53265 | True | - |
| continuous | 1 | fine | all_frames | 0.19438 | 0.18840 | 0.39930 | 1.86629 | 13.81078 | False | 15->30 |
| continuous | 5 | anchor | inside_camera_view | 0.00189 | 0.00406 | 0.01144 | n/a | 0.04713 | True | - |
| continuous | 5 | anchor | all_frames | 0.17042 | 0.09241 | 0.04346 | n/a | 0.05696 | False | 15->30 |
| continuous | 5 | fine | inside_camera_view | 0.00189 | 0.00406 | 0.01144 | 0.02505 | 0.04713 | True | - |
| continuous | 5 | fine | all_frames | 0.17042 | 0.09241 | 0.04346 | 0.03900 | 0.05696 | False | 15->30 |
| continuous | 15 | anchor | inside_camera_view | 0.00132 | 0.00166 | 0.00133 | n/a | 0.01156 | False | 30->60 |
| continuous | 15 | anchor | all_frames | 0.16964 | 0.08553 | 0.03339 | n/a | 0.02002 | False | 15->30 |
| continuous | 15 | fine | inside_camera_view | 0.00132 | 0.00166 | 0.00133 | 0.00208 | 0.01156 | False | 30->60 |
| continuous | 15 | fine | all_frames | 0.16964 | 0.08553 | 0.03339 | 0.01828 | 0.02002 | False | 15->30 |
| hybrid | 1 | anchor | inside_camera_view | 0.01652 | 0.07558 | 0.29882 | n/a | 2.98228 | True | - |
| hybrid | 1 | anchor | all_frames | 0.19514 | 0.16933 | 0.36196 | n/a | 3.24343 | False | 15->30 |
| hybrid | 1 | fine | inside_camera_view | 0.01652 | 0.07558 | 0.29882 | 0.93881 | 2.98228 | True | - |
| hybrid | 1 | fine | all_frames | 0.19514 | 0.16933 | 0.36196 | 0.97414 | 3.24343 | False | 15->30 |
| hybrid | 5 | anchor | inside_camera_view | 0.00191 | 0.00497 | 0.00642 | n/a | 0.04440 | True | - |
| hybrid | 5 | anchor | all_frames | 0.16880 | 0.09295 | 0.03795 | n/a | 0.05876 | False | 15->30 |
| hybrid | 5 | fine | inside_camera_view | 0.00191 | 0.00497 | 0.00642 | 0.01771 | 0.04440 | True | - |
| hybrid | 5 | fine | all_frames | 0.16880 | 0.09295 | 0.03795 | 0.03679 | 0.05876 | False | 15->30 |
| hybrid | 15 | anchor | inside_camera_view | 0.00138 | 0.00185 | 0.00134 | n/a | 0.01703 | False | 30->60 |
| hybrid | 15 | anchor | all_frames | 0.17108 | 0.08328 | 0.03277 | n/a | 0.02687 | False | 15->30 |
| hybrid | 15 | fine | inside_camera_view | 0.00138 | 0.00185 | 0.00134 | 0.00279 | 0.01703 | False | 30->60 |
| hybrid | 15 | fine | all_frames | 0.17108 | 0.08328 | 0.03277 | 0.01978 | 0.02687 | False | 15->30 |

## Anchor consistency against the published issue-79 curves (provenance only)

| h | e | issue-83 probe | issue-79 published | relative difference |
| --- | --- | --- | --- | --- |
| 1 | 15 | 0.020187 | 0.020187 | 0.0000 |
| 1 | 30 | 0.091034 | 0.091034 | 0.0000 |
| 1 | 60 | 0.370743 | 0.370743 | 0.0000 |
| 1 | 120 | 13.532655 | 13.532652 | 0.0000 |
| 5 | 15 | 0.001888 | 0.001888 | 0.0000 |
| 5 | 30 | 0.004063 | 0.004063 | 0.0000 |
| 5 | 60 | 0.011445 | 0.011445 | 0.0000 |
| 5 | 120 | 0.047125 | 0.047125 | 0.0000 |
| 15 | 15 | 0.001317 | 0.001317 | 0.0000 |
| 15 | 30 | 0.001658 | 0.001658 | 0.0000 |
| 15 | 60 | 0.001334 | 0.001334 | 0.0000 |
| 15 | 120 | 0.011556 | 0.011556 | 0.0000 |

## Dispositions

- Q1 localization (regime property vs uniform): **supported** - dip concentrated in post_settlement segments
- Q2 artifact sensitivity (grid/mask): **not_supported_by_this_experiment** - all primary variants stay non-monotone

Both families are reported separately; a CLEVRER non-replication does not falsify the NovPhy boundary, and vice versa. Either localization answer is a valid outcome.

## Limitations

- decomposition of the frozen issue-79 rollouts; no re-freeze or amendment of any issue-79 verdict
- engine-annotation features in, no visual perception on either arm; shared frozen parser contract
- descriptive paired bootstrap over (scene, start-or-context) units; no inferential claim
- the hybrid arm runs its continuous-abstraction pair; the micro pair stays with issue-79 Q2
- segment labels derive from collision-event timelines, not from measured object speeds
