# Angular replay pilot findings

The frozen action-design viability screen **failed**. Four of five lineages are
comparable, but only two comparable lineages have a verified native-clear grid
action; the prospective requirement was at least three. This is not a #76
advancement result and authorizes neither training nor fresh evaluation.

All 65 assignments were retained: 64 passed capture/outcome integrity checks;
071-a04 remains the original collection failure with a 1e9 penalty. The entire
071 lineage is excluded from paired ranking. Eight other candidates in that
lineage also exceed the unchanged bird velocity/angular-velocity comparability
bounds. No birds, candidates, or failed states were removed.

| Training lineage | Comparable | Grid native clears | Distinct settled count costs |
| --- | --- | ---: | --- |
| 001 | yes | 0 | no |
| 071 | no | 0 | yes; not paired evidence |
| 141 | yes | 1 | yes |
| 211 | yes | 0 | no |
| 281 | yes | 1 | yes |

The two native clears are 141-a06, drag (-61,51), and 281-a03, drag (-76,26).
Their exact engine-terminal events are bound in the audit. Stable endpoints,
zero observed pigs without a clear, and censored captures are not wins. In
particular, 001-a05 and 001-a12 have zero active pigs at the censored endpoint
but retain failure penalties. No learned action selection was tested here.

Collection used 6,964.526332 wall seconds including the 144.730600-second smoke,
peaked at 1,825.28125 MiB worker-tree RSS, and retained 12,422,927,674 artifact
bytes. Validation and outcome reporting used 856.332360 CPU wall seconds plus
11.334173 seconds of prior smoke validation, leaving 932.333467 seconds of the
1,800-second offline allowance. No CUDA scoring, optimizer updates, new seeds,
replacement levels, or fresh evaluation occurred.

The [prospective protocol](issue-76-angular-replay-protocol.md),
[complete machine-readable audit](../data/issue-76-angular-replay/collection-validation.json),
and [accounting correction](../data/issue-76-angular-replay/accounting-correction.json)
retain the execution bindings, all comparisons, terminal evidence, and receipts.
Raw captures remain in the local artifact collection. The original-grid negative
findings are unchanged. This failed pilot is not followed by an automatic angle
or optimizer sweep; the next workflow needs a separately justified design.

## Complete assignment table

All releases are 600 ms with tap 0 ms. References are excluded from the twelve
grid candidates. Pigs/blocks are native active-lifecycle counts, not predictions.
The paired column is the whole-lineage comparability decision; it does not turn
censored or failed outcomes into usable task outcomes. Cost is the unchanged
1000 × pigs + blocks for usable stable/clear endpoints, otherwise 1e9.

| Assignment | Role | Drag | Outcome | Pigs | Blocks | Cost | Native clear | Paired |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 001-a00 | reference | (-80, 10) | native_time_window_limit | 1 | 1 | 1e9 | no | yes |
| 001-a01 | grid | (-80, 7) | stable_entered | 1 | 1 | 1001 | no | yes |
| 001-a02 | grid | (-78, 17) | stable_entered | 1 | 1 | 1001 | no | yes |
| 001-a03 | grid | (-76, 26) | stable_entered | 1 | 1 | 1001 | no | yes |
| 001-a04 | grid | (-72, 35) | native_time_window_limit | 1 | 1 | 1e9 | no | yes |
| 001-a05 | grid | (-67, 44) | native_time_window_limit | 0 | 1 | 1e9 | no | yes |
| 001-a06 | grid | (-61, 51) | stable_entered | 1 | 1 | 1001 | no | yes |
| 001-a07 | grid | (-55, 59) | stable_entered | 1 | 1 | 1001 | no | yes |
| 001-a08 | grid | (-47, 65) | stable_entered | 1 | 1 | 1001 | no | yes |
| 001-a09 | grid | (-39, 70) | native_time_window_limit | 1 | 1 | 1e9 | no | yes |
| 001-a10 | grid | (-30, 74) | native_time_window_limit | 1 | 1 | 1e9 | no | yes |
| 001-a11 | grid | (-21, 77) | native_time_window_limit | 1 | 1 | 1e9 | no | yes |
| 001-a12 | grid | (-11, 79) | native_time_window_limit | 0 | 1 | 1e9 | no | yes |
| 071-a00 | reference | (-80, 10) | stable_entered | 1 | 4 | 1004 | no | no |
| 071-a01 | grid | (-80, 7) | native_time_window_limit | 1 | 5 | 1e9 | no | no |
| 071-a02 | grid | (-78, 17) | native_time_window_limit | 1 | 5 | 1e9 | no | no |
| 071-a03 | grid | (-76, 26) | stable_entered | 1 | 5 | 1005 | no | no |
| 071-a04 | grid | (-72, 35) | replay did not complete within its supervised capture contract | — | — | 1e9 | no | no |
| 071-a05 | grid | (-67, 44) | stable_entered | 1 | 5 | 1005 | no | no |
| 071-a06 | grid | (-61, 51) | stable_entered | 1 | 5 | 1005 | no | no |
| 071-a07 | grid | (-55, 59) | stable_entered | 1 | 5 | 1005 | no | no |
| 071-a08 | grid | (-47, 65) | native_time_window_limit | 1 | 5 | 1e9 | no | no |
| 071-a09 | grid | (-39, 70) | stable_entered | 1 | 4 | 1004 | no | no |
| 071-a10 | grid | (-30, 74) | native_time_window_limit | 1 | 5 | 1e9 | no | no |
| 071-a11 | grid | (-21, 77) | stable_entered | 1 | 5 | 1005 | no | no |
| 071-a12 | grid | (-11, 79) | native_time_window_limit | 1 | 5 | 1e9 | no | no |
| 141-a00 | reference | (-80, 10) | level_fail | 1 | 5 | 1e9 | no | yes |
| 141-a01 | grid | (-80, 7) | level_fail | 1 | 5 | 1e9 | no | yes |
| 141-a02 | grid | (-78, 17) | level_fail | 1 | 6 | 1e9 | no | yes |
| 141-a03 | grid | (-76, 26) | stable_entered | 1 | 5 | 1005 | no | yes |
| 141-a04 | grid | (-72, 35) | native_time_window_limit | 1 | 5 | 1e9 | no | yes |
| 141-a05 | grid | (-67, 44) | native_time_window_limit | 1 | 6 | 1e9 | no | yes |
| 141-a06 | grid | (-61, 51) | level_clear | 0 | 6 | 6 | yes | yes |
| 141-a07 | grid | (-55, 59) | native_time_window_limit | 1 | 6 | 1e9 | no | yes |
| 141-a08 | grid | (-47, 65) | native_time_window_limit | 1 | 6 | 1e9 | no | yes |
| 141-a09 | grid | (-39, 70) | native_time_window_limit | 1 | 6 | 1e9 | no | yes |
| 141-a10 | grid | (-30, 74) | stable_entered | 1 | 6 | 1006 | no | yes |
| 141-a11 | grid | (-21, 77) | native_time_window_limit | 1 | 6 | 1e9 | no | yes |
| 141-a12 | grid | (-11, 79) | stable_entered | 1 | 6 | 1006 | no | yes |
| 211-a00 | reference | (-80, 10) | native_time_window_limit | 1 | 7 | 1e9 | no | yes |
| 211-a01 | grid | (-80, 7) | level_fail | 1 | 7 | 1e9 | no | yes |
| 211-a02 | grid | (-78, 17) | level_fail | 1 | 7 | 1e9 | no | yes |
| 211-a03 | grid | (-76, 26) | level_fail | 1 | 7 | 1e9 | no | yes |
| 211-a04 | grid | (-72, 35) | native_time_window_limit | 1 | 7 | 1e9 | no | yes |
| 211-a05 | grid | (-67, 44) | level_fail | 1 | 7 | 1e9 | no | yes |
| 211-a06 | grid | (-61, 51) | native_time_window_limit | 1 | 7 | 1e9 | no | yes |
| 211-a07 | grid | (-55, 59) | native_time_window_limit | 1 | 6 | 1e9 | no | yes |
| 211-a08 | grid | (-47, 65) | native_time_window_limit | 1 | 7 | 1e9 | no | yes |
| 211-a09 | grid | (-39, 70) | native_time_window_limit | 1 | 7 | 1e9 | no | yes |
| 211-a10 | grid | (-30, 74) | native_time_window_limit | 1 | 7 | 1e9 | no | yes |
| 211-a11 | grid | (-21, 77) | native_time_window_limit | 1 | 7 | 1e9 | no | yes |
| 211-a12 | grid | (-11, 79) | stable_entered | 1 | 6 | 1006 | no | yes |
| 281-a00 | reference | (-80, 10) | native_time_window_limit | 1 | 4 | 1e9 | no | yes |
| 281-a01 | grid | (-80, 7) | native_time_window_limit | 1 | 4 | 1e9 | no | yes |
| 281-a02 | grid | (-78, 17) | native_time_window_limit | 1 | 4 | 1e9 | no | yes |
| 281-a03 | grid | (-76, 26) | level_clear | 0 | 4 | 4 | yes | yes |
| 281-a04 | grid | (-72, 35) | native_time_window_limit | 1 | 4 | 1e9 | no | yes |
| 281-a05 | grid | (-67, 44) | stable_entered | 1 | 4 | 1004 | no | yes |
| 281-a06 | grid | (-61, 51) | stable_entered | 1 | 4 | 1004 | no | yes |
| 281-a07 | grid | (-55, 59) | stable_entered | 1 | 4 | 1004 | no | yes |
| 281-a08 | grid | (-47, 65) | stable_entered | 1 | 4 | 1004 | no | yes |
| 281-a09 | grid | (-39, 70) | native_time_window_limit | 1 | 4 | 1e9 | no | yes |
| 281-a10 | grid | (-30, 74) | native_time_window_limit | 1 | 4 | 1e9 | no | yes |
| 281-a11 | grid | (-21, 77) | native_time_window_limit | 1 | 4 | 1e9 | no | yes |
| 281-a12 | grid | (-11, 79) | native_time_window_limit | 1 | 4 | 1e9 | no | yes |

