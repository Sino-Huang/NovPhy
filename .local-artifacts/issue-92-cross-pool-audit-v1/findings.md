# Issue-92 WP3 cross-pool selection-validity audit - findings

Plan identity `issue-92-cross-pool-audit-v1`, schema `issue_92_cross_pool_audit_plan_v1`, version 1 (terminal), frozen 2026-09-22T17:35:14Z.

## Dispositions

- Q1 bounded-transfer breadth: **supported**
- Q2 cross-split selection validity: **supported**
- Q3 N2/N2n typed coverage: **supported**

## Q1 bounded-transfer ceiling (member unit)

- measurable members 46; admissible cells 575; typed failures 41
- member-unit ceiling 25/46 = 0.5435 [0.3913, 0.6957] (DESCRIPTIVE)
- successful cells 32 with ordinal histogram {"0": 1, "1": 1, "10": 3, "11": 3, "12": 6, "2": 3, "3": 3, "4": 5, "5": 3, "6": 4}

### 002-a10 typed-failure resolution

- coverage status `failed` (operationally_valid False); segment summary failure `native_time_window_limit` with event counts {"bird_exhaustion": 1, "bird_launched": 1, "collision": 199, "entity_death": 1, "entity_destroyed": 1, "pig_removed": 1, "stable_exited": 1}
- chunk recomputation: pig_removed on runtime:pig:0000 = True (1 events)
- exclusion rule: coverage-admissible join: under the #85/#87 semantics the cell carries no success value and is excluded from the ceiling; the cell, its evidence and this rule are published, never assumed away
- typed failures carrying an engine-truth success: 1 of 41 - the bounded pool's typed failures are NOT outcome-blind

## Q2 cross-split selection validity (conditions never pooled)

- cells 612 per condition over 17 states; pooled top-1 42/1224
- zero-shot (frozen arm): 13/612 = 0.0212 vs uniform chance 0.0407; paired difference -0.0195 [-0.0368, -0.0045] (member-clustered, DESCRIPTIVE)
- few-shot (adapted arm, not a frozen-system result): 29/612 = 0.0474
- #87-analogue systems: 2/153 per condition
- frozen-arm successes at ordinal 12: 11 of 13 on 4 states

## Q3 N2/N2n typed coverage

- N2: 186/208 admissible; ceiling 5/15 [0.1333, 0.6000]; type010102 half 0/7
- N2n: 70/104 typed failures; ceiling 0/5; no successful action exists in the pool, so no selection statement is estimable

## Pre-declared expectation checks

- 48 of 49 pre-declared expectations reproduce exactly; 1 delta(s) published below (never silently absorbed)
- DELTA constant_ordinal.n1.a02: pre-declared [2, 12], recomputed [1, 12] - the recomputed value is the retained-truth publication

## Constant-ordinal reference tables (the only admissible baselines)

- N1 (12 sealed ceiling states, WP1 verdict path): a04 5/12, a03 2/12, a05 2/12, a06 2/12, a02 1/12, a08 0/12, a12 0/12
- cross-split (17 states, availability varies): a12 3/15, a02 2/13, a03 1/13, a04 1/14, a06 1/16, a10 1/15, a08 0/17
- bounded pool (46 measurable members): a12 6/44, a04 5/44, a06 4/43, a02 3/43, a03 3/45, a05 3/45, a10 3/44, a11 3/44, a00 1/45, a01 1/45
- (i) on each pool a model-free constant policy beats every frozen ranker: on N1 a04 lands 5 of 12 ceiling states while all frozen systems land 0, and on the cross-split a constant a12 succeeds in 3 of 15 states where admissible against the frozen systems' 0.0212 cell rate
- (ii) no single constant transfers between pools: a04 is the best constant on N1 and scores 1/14 on the cross-split; a12 is the best constant on the cross-split and scores 0/12 on N1; a08 is retired as any kind of baseline (0/17 cross-split, 0/12 N1)
## Claim boundary

cross-pool audit over retained artifacts only; #76's, #77's and the n2-eval pool's published verdicts are inputs and are never recomputed, amended, or reinterpreted beyond the frozen recomputation declared here; #64/#65 stay sealed; the n2-eval system set (12 systems incl. adapted predictors and micro/macro ablations) is never mixed with #87's 4-system set in one rate; zero-shot and few-shot are reported separately and never pooled; no engine access, no rendering, no new data; intervals are DESCRIPTIVE

## Cost

- measured wall 18.023s of 300.0s cap; zero GPU, zero engine seconds
