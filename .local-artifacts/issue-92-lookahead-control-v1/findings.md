# Issue-92 WP1b: lookahead control (decision-only endpoint arms) — findings

- identity `issue-92-lookahead-control-v1`, plan schema `issue_92_lookahead_control_plan_v1` version 1 (terminal), frozen 2026-09-22T16:44:41Z before any experiment statistic
- validation command: `python -u -m scripts.run_lookahead_control --validate`
- zero engine seconds, zero rendering, zero retraining; GPU flock held for every measured phase

## 1. The confound this package closes

the #87 decision rule rolls each candidate only pair.delta carrier steps (1 agent frame for *-fixed-h1, 25 for *-fixed-h5) while the engine-truth outcomes resolve 97.8-400.9 agent frames after the same sealed decision frame; no #87 arm ever rolls far enough to reach the moment the outcome is decided, so the zero-selection result is confounded with a lookahead of ~0.3-7% of the outcome horizon; this ticket adds decision-only endpoint arms at the frozen 225-step deployment endpoint and the 600-step (601-frame) observed-window endpoint.

Endpoint coverage (measured from the retained engine-channel pig-removal event steps):
- endpoint 225: reaches the outcome resolution on 8 of 13 successful slots
- endpoint 600: reaches the outcome resolution on 13 of 13 successful slots
- resolution offsets (agent frames after the sealed decision frame): issue-77-n1-001-a00/a06 353.3, issue-77-n1-001-a07/a06 353.3, issue-77-n1-005-a07/a07 370.1, issue-77-n1-006-a05/a05 400.9, issue-77-n1-006-a11/a05 400.9, issue-77-n1-010-a06/a02 97.8, issue-77-n1-011-a02/a04 101.9, issue-77-n1-011-a11/a04 101.9, issue-77-n1-012-a08/a04 124.8, issue-77-n1-014-a01/a04 103.3, issue-77-n1-014-a10/a04 103.3, issue-77-n1-016-a03/a03 122.4, issue-77-n1-016-a12/a03 122.4

## 2. Smoke evidence (pre-freeze; stop rule held)

- replication_control: ok True
- anchor_sha256_equality: ok True
- adaptive_cell_continuous: ok True
- adaptive_cell_hybrid: ok True

## 3. The endpoint ladder over the 36 ceiling cells per arm

Success band (N1, from the union verdict table): [2, 3, 4, 5, 6]; each sealed ceiling state has one successful ordinal among 9-13, so a single arm's 0/36 has p approximately (12/13)^36 = 0.056 under a uniform-selection null; the informative statistic is the joint ordinal-band evidence, never a single arm's count.

| arm | hits | rate | interval | chosen in band | chosen histogram | typed candidate failures | gpu s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| continuous-adaptive-e225 | 0/36 | 0.0000 | [0.0000, 0.0000] | 4 | {0: 18, 1: 2, 5: 4, 7: 5, 10: 2, 11: 2, 12: 3} | 0 | 12.5 |
| continuous-adaptive-e600 | 0/36 | 0.0000 | [0.0000, 0.0000] | 0 | {0: 36} | 0 | 29.0 |
| continuous-fixed-h1-e225 | 0/36 | 0.0000 | [0.0000, 0.0000] | 0 | {0: 28, 1: 2, 8: 1, 11: 2, 12: 3} | 0 | 17.7 |
| continuous-fixed-h1-e600 | 0/36 | 0.0000 | [0.0000, 0.0000] | 0 | {0: 35, 8: 1} | 0 | 47.0 |
| continuous-fixed-h15-e225 | 0/36 | 0.0000 | [0.0000, 0.0000] | 1 | {1: 17, 3: 1, 7: 10, 11: 2, 12: 6} | 0 | 1.2 |
| continuous-fixed-h15-e600 | 0/36 | 0.0000 | [0.0000, 0.0000] | 0 | {1: 14, 7: 12, 12: 10} | 0 | 3.2 |
| continuous-fixed-h5-e225 | 0/36 | 0.0000 | [0.0000, 0.0000] | 0 | {1: 12, 7: 14, 12: 10} | 0 | 3.6 |
| continuous-fixed-h5-e600 | 0/36 | 0.0000 | [0.0000, 0.0000] | 0 | {0: 32, 12: 4} | 0 | 9.4 |
| hybrid-adaptive-e225 | 0/36 | 0.0000 | [0.0000, 0.0000] | 2 | {0: 24, 1: 8, 5: 2, 12: 2} | 0 | 18.7 |
| hybrid-adaptive-e600 | 0/36 | 0.0000 | [0.0000, 0.0000] | 0 | {0: 27, 1: 5, 12: 4} | 0 | 37.9 |
| hybrid-fixed-h1-e225 | 0/36 | 0.0000 | [0.0000, 0.0000] | 0 | {0: 36} | 0 | 22.0 |
| hybrid-fixed-h1-e600 | 0/36 | 0.0000 | [0.0000, 0.0000] | 4 | {0: 30, 5: 4, 12: 2} | 0 | 58.8 |
| hybrid-fixed-h15-e225 | 2/36 | 0.0556 | [0.0000, 0.1429] | 3 | {1: 13, 3: 3, 10: 4, 12: 16} | 0 | 1.5 |
| hybrid-fixed-h15-e600 | 0/36 | 0.0000 | [0.0000, 0.0000] | 0 | {1: 31, 7: 2, 12: 3} | 0 | 4.0 |
| hybrid-fixed-h5-e225 | 0/36 | 0.0000 | [0.0000, 0.0000] | 0 | {1: 29, 7: 3, 12: 4} | 0 | 4.4 |
| hybrid-fixed-h5-e600 | 0/36 | 0.0000 | [0.0000, 0.0000] | 3 | {0: 26, 1: 6, 3: 3, 12: 1} | 0 | 11.8 |

## 4. Disposition

**supported** (any arm at the 0.5 margin: False; all arms below 0.5 with the interval below: True; guards ok: True).

## 5. Claim boundary (carried verbatim)

The adaptive arm selects its own prediction horizon and abstraction at each rollout step from a controller trained by dynamic programming on the same frozen world-model corpus; it does not learn from task outcomes, does not adapt within or across episodes, and chooses only among the frozen candidate inventory. The record therefore bounds decision-time horizon adaptivity, not policy learning.

decision-only rollouts of the frozen issue-77 N1 dynamics checkpoints from the sealed #87 anchors; no engine access, no rendering, no retraining, no gameplay; #87/#89/#90 published verdicts are inputs and are never recomputed, amended, or reinterpreted; #64/#65 stay sealed; single-shot, decision-only, development-lineage scoped; the cohort-v2 CEMPlanner gameplay-planning path is not admissible on this budget (it proposes continuous actions outside the frozen inventory, so the candidate-identity binding fails, and it binds a different predictor lineage); intervals are DESCRIPTIVE.
- decision-only endpoint arms: the rollout never touches the engine, so the arms measure selection from imagined rollouts, not executed gameplay
- the 225-step endpoint reaches the outcome horizon on only a subset of the successful slots; the 600-step endpoint covers all of them (coverage published from the retained engine-channel event steps)
- the adaptive arm bounds decision-time horizon adaptivity, not policy learning (claim-boundary sentence carried verbatim)
- the union verdict table is single-shot per (state, ordinal); per-slot execution variance is unobserved beyond the measured determinism statement of #92 WP1
- no content hash is recomputed after the freeze and no full-corpus integrity pass runs anywhere; inputs are verified at read time by their carried schema/identity/plan_identity fields
