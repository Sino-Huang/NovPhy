# Issue-82 oracle-ceiling diagnostic of the zero floor - findings

Plan: issue-82-zero-floor-oracle-v1 (frozen before outcome: True).

Claim boundary: oracle-ceiling diagnostic on the frozen #80 membership from existing #77/#80 artifacts; no new gameplay, captures, or renders; no multi-shot, adaptation, few-shot, zero-shot, or complete-gameplay claim; the oracle ceiling bounds what ANY planner could show on this membership and does not evaluate a planner; descriptive intervals only; the #80 pilot record and every prior disposition (#15/#72/#74/#75/#77/#78/#79/#80) stay unchanged; #64/#65 stay sealed and unauthorized

## Phase-0 success-predicate binding

Status: no_defensible_binding.

Stage 1 rule: stage 1 (candidate threshold): threshold = the maximum realized t=600 end-of-window replay count cost over the frozen admissible state-candidate cells whose issue-77 N1 replay native trace records pig removal (interaction_coverage pig_removed), i.e. the least inclusive monotone bound consistent with observed replay-side pig removals; if no frozen cell records pig removal, no candidate threshold exists

Stage 2 rule: stage 2 (defensibility gate, pre-declared before any state-level outcome): the stage-1 candidate threshold is a defensible binding only if the predicate 'realized cost <= threshold' agrees with the replay-side engine pig evidence on at least 0.50 of the frozen cells (the predicate must agree with its own anchoring executions at least half the time); a stage-1 threshold below the floor means the realized cost cannot express the success event and NO defensible binding exists

Stage 1 outcome: candidate threshold = realized cost <= 81.688258, set by 1 cell(s) (issue-77-n1-005-a07); 8/177 frozen cells record replay-side pig removal (their costs span [32.843, 81.688]); 98/177 replay segments are right-censored at the window limit.

Stage 2 outcome: observed agreement of the candidate predicate with the replay engine evidence 0.1299 (154 false-success cells, 0 missed-removal cells) vs the floor 0.50 and the always-failure baseline 0.9548; passed=False.

Declared divergence: The predicate is a REPLAY-LEVEL PROXY, not the closed-loop first-shot outcome: (1) the replay cost is read at the right-censored t=600 end-of-window offset and is not a settled cost, while gameplay outcomes are uncensored engine events; (2) the cost aggregates a parser-derived expected surviving-pig count and block count, not an engine death event; (3) the replay rows come from the issue-77 N1 campaign executions of the frozen actions, whereas closed-loop first-shot success was recorded by the #80 pilot on freshly materialized states; (4) the threshold is the least inclusive monotone bound consistent with observed replay-side pig removals, so its agreement with both replay engine evidence and the #80 closed-loop pilot outcomes is quantified and published alongside every oracle table

Closed-loop divergence of the stage-1 candidate predicate over the #80 pilot (45 valid cells, 3 typed): agreement 0.1111 (0 both-success, 40 predicate-only, 0 gameplay-only, 5 both-failure).

Blocker: the frozen estimand basis cannot express the success event: the stage-1 candidate threshold (realized cost <= 81.688258, the least inclusive bound consistent with the 8 replay-side pig removals observed among 177 frozen cells) agrees with the replay engine evidence on only 0.1299 of the cells - below the pre-declared 0.50 defensibility floor - because it marks 154 no-pig-removal cells as successes (their parser-derived pig component is ~0 although the pig survived); the pig-removed costs [32.843, 81.688] lie interior to the overall cost distribution, so no numeric threshold rule on the realized t=600 replay cost can separate replay-side pig removal from survival, and the predicate cannot defensibly proxy the closed-loop first-shot success

## Protocol note (future pilots only)

the #80 pilot retained 3 typed decision_failure: prior_candidate_absent cells (no-model-ordinal-prior on issue-77-n1-003-a00 across seeds 20260908/20260909/20260910; ordinal 8 is a typed-dropped branch of member issue-77-n1-003). pre-declared for FUTURE pilots only (never amended into the #80 pilot record): if the ordinal-8 candidate is not admissible on a state, the no-model ordinal prior falls back to the admissible ordinal nearest 8 by absolute distance, ties broken by the lower ordinal

## Compute accounting

GPU 3.269 s of the 360 s allowance (0.1 GPU-h); wall 3.4 s of 3600 s; decision frames parsed 15; parser calls 15; predictor loads 9; transition calls 3717; linear MACs 6661739088; model scoring slots 2601 state x candidate cells across system-seeds.

## Predicate-free descriptive tables

### Per-state best candidate by realized replay cost (sorted by state)

| State | Family | Member | Admissible | Best ordinal | Best branch | Best realized cost | Oracle indicator |
| --- | --- | --- | --- | --- | --- | --- | --- |
| issue-77-n1-001-a00 | type010103 | issue-77-n1-001 | 12 | 7 | issue-77-n1-001-a07 | 38.563 | n/a |
| issue-77-n1-001-a07 | type010103 | issue-77-n1-001 | 12 | 7 | issue-77-n1-001-a07 | 38.563 | n/a |
| issue-77-n1-003-a00 | type010103 | issue-77-n1-003 | 9 | 1 | issue-77-n1-003-a01 | 41.677 | n/a |
| issue-77-n1-003-a10 | type010103 | issue-77-n1-003 | 9 | 1 | issue-77-n1-003-a01 | 41.677 | n/a |
| issue-77-n1-004-a07 | type010103 | issue-77-n1-004 | 9 | 2 | issue-77-n1-004-a02 | 77.595 | n/a |
| issue-77-n1-005-a07 | type010103 | issue-77-n1-005 | 4 | 8 | issue-77-n1-005-a08 | 67.431 | n/a |
| issue-77-n1-006-a05 | type010103 | issue-77-n1-006 | 13 | 5 | issue-77-n1-006-a05 | 63.951 | n/a |
| issue-77-n1-006-a11 | type010103 | issue-77-n1-006 | 13 | 5 | issue-77-n1-006-a05 | 63.951 | n/a |
| issue-77-n1-007-a05 | type010103 | issue-77-n1-007 | 13 | 9 | issue-77-n1-007-a09 | 61.919 | n/a |
| issue-77-n1-007-a12 | type010103 | issue-77-n1-007 | 13 | 9 | issue-77-n1-007-a09 | 61.919 | n/a |
| issue-77-n1-008-a05 | type010103 | issue-77-n1-008 | 13 | 2 | issue-77-n1-008-a02 | 50.814 | n/a |
| issue-77-n1-008-a12 | type010103 | issue-77-n1-008 | 13 | 2 | issue-77-n1-008-a02 | 50.814 | n/a |
| issue-77-n1-009-a00 | type010105 | issue-77-n1-009 | 13 | 9 | issue-77-n1-009-a09 | 47.712 | n/a |
| issue-77-n1-009-a09 | type010105 | issue-77-n1-009 | 13 | 9 | issue-77-n1-009-a09 | 47.712 | n/a |
| issue-77-n1-010-a06 | type010105 | issue-77-n1-010 | 13 | 4 | issue-77-n1-010-a04 | 33.362 | n/a |
| issue-77-n1-011-a02 | type010105 | issue-77-n1-011 | 13 | 9 | issue-77-n1-011-a09 | 45.768 | n/a |
| issue-77-n1-011-a11 | type010105 | issue-77-n1-011 | 13 | 9 | issue-77-n1-011-a09 | 45.768 | n/a |
| issue-77-n1-012-a08 | type010105 | issue-77-n1-012 | 13 | 2 | issue-77-n1-012-a02 | 54.447 | n/a |
| issue-77-n1-013-a04 | type010105 | issue-77-n1-013 | 13 | 4 | issue-77-n1-013-a04 | 57.876 | n/a |
| issue-77-n1-014-a01 | type010105 | issue-77-n1-014 | 13 | 5 | issue-77-n1-014-a05 | 41.708 | n/a |
| issue-77-n1-014-a10 | type010105 | issue-77-n1-014 | 13 | 5 | issue-77-n1-014-a05 | 41.708 | n/a |
| issue-77-n1-015-a06 | type010105 | issue-77-n1-015 | 13 | 0 | issue-77-n1-015-a00 | 27.495 | n/a |
| issue-77-n1-016-a03 | type010105 | issue-77-n1-016 | 13 | 4 | issue-77-n1-016-a04 | 27.810 | n/a |
| issue-77-n1-016-a12 | type010105 | issue-77-n1-016 | 13 | 4 | issue-77-n1-016-a04 | 27.810 | n/a |

### System-vs-oracle cost gap (per seed; DESCRIPTIVE)

| System | Seed | States scored | Typed failures | Predicate-success states | Mean chosen cost | Mean oracle-best cost | Mean gap cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid-fixed-h1 | 20260908 | 24 | 0 | n/a | 59.785 | 48.252 | 11.532 |
| hybrid-fixed-h1 | 20260909 | 24 | 0 | n/a | 59.785 | 48.252 | 11.532 |
| hybrid-fixed-h1 | 20260910 | 24 | 0 | n/a | 58.567 | 48.252 | 10.315 |
| continuous-fixed-h1 | 20260908 | 24 | 0 | n/a | 52.663 | 48.252 | 4.411 |
| continuous-fixed-h1 | 20260909 | 24 | 0 | n/a | 57.767 | 48.252 | 9.515 |
| continuous-fixed-h1 | 20260910 | 24 | 0 | n/a | 66.450 | 48.252 | 18.198 |
| continuous-fixed-h5 | 20260908 | 24 | 0 | n/a | 53.626 | 48.252 | 5.374 |
| continuous-fixed-h5 | 20260909 | 24 | 0 | n/a | 66.450 | 48.252 | 18.198 |
| continuous-fixed-h5 | 20260910 | 24 | 0 | n/a | 57.767 | 48.252 | 9.515 |
| no-model-ordinal-prior | 20260908 | 22 | 2 | n/a | 68.772 | 48.850 | 19.923 |
| no-model-ordinal-prior | 20260909 | 22 | 2 | n/a | 68.772 | 48.850 | 19.923 |
| no-model-ordinal-prior | 20260910 | 22 | 2 | n/a | 68.772 | 48.850 | 19.923 |

### Threshold sensitivity (pre-declared grid; DESCRIPTIVE only)

| Threshold | Cells meeting the predicate (of 177) |
| --- | --- |
| 25 | 0 |
| 50 | 41 |
| 75 | 157 |
| 100 | 172 |
| 250 | 174 |
| 500 | 174 |
| 750 | 175 |
| 1000 | 177 |

## Dispositions

- Q1 (oracle ceiling nonzero): readiness_or_precision_insufficient - the oracle ceiling is not measurable without a defensible success predicate.
- Q2 (motivated follow-up): readiness_or_precision_insufficient - no branch can be typed from an unmeasurable ceiling; the predicate-free tables above are the descriptive input any follow-up must start from.

Ticket disposition: readiness_or_precision_insufficient (tokens: supported / not_supported_by_this_experiment / readiness_or_precision_insufficient). The blocked Phase-0 binding is a valid terminal outcome; nothing was forced.

## Limitations

- the success predicate is a proxy with quantified, non-zero divergence from both replay engine evidence and closed-loop first-shot outcomes
- the replay cost is right-censored at t=600 and is not a settled cost
- states sharing a source member share one physical initial state and candidate table; the bootstrap over 24 state identities is descriptive only
- the oracle ceiling is bounded by the frozen 13-candidate inventory reduced by typed branch failures; no unlisted action is considered
- model-system gap rows reuse the frozen #77/#80 checkpoints; no planner is evaluated and no competence is implied
