# Issue-92 WP2b: cross-seed invariance probe — findings

- identity `issue-92-cross-seed-invariance-v1`, plan schema `issue_92_cross_seed_invariance_plan_v1` version 1 (terminal), frozen 2026-09-22T17:47:24Z before any experiment cell
- validation command: `python -u -m scripts.run_cross_seed_invariance_probe --validate`
- second seed rule: engine_seed + 1000000000 (NOVPHY_ENVIRONMENT_SEED initializes the Unity RNG only; ORACLE_SEED 20260908 is a protocol label, never an engine input)

## 1. Verdict agreement (the decision)

**26 of 26** second-seed verdicts agree with the retained union-table verdicts; flips: 0.

| kind | slot | retained | second seed | agrees | frame byte-equal |
| --- | --- | --- | --- | --- | --- |
| anchor | seed2--issue-77-n1-001-a00--a03 | False | False | True | True |
| anchor | seed2--issue-77-n1-003-a00--a00 | False | False | True | True |
| anchor | seed2--issue-77-n1-004-a07--a01 | False | False | True | False |
| anchor | seed2--issue-77-n1-006-a05--a00 | False | False | True | True |
| anchor | seed2--issue-77-n1-007-a05--a00 | False | False | True | True |
| anchor | seed2--issue-77-n1-008-a05--a00 | False | False | True | True |
| anchor | seed2--issue-77-n1-009-a00--a00 | False | False | True | False |
| anchor | seed2--issue-77-n1-010-a06--a00 | False | False | True | True |
| anchor | seed2--issue-77-n1-011-a02--a00 | False | False | True | False |
| anchor | seed2--issue-77-n1-012-a08--a00 | False | False | True | True |
| anchor | seed2--issue-77-n1-013-a04--a00 | False | False | True | True |
| anchor | seed2--issue-77-n1-014-a01--a00 | False | False | True | True |
| anchor | seed2--issue-77-n1-015-a06--a00 | False | False | True | False |
| anchor | seed2--issue-77-n1-016-a03--a00 | False | False | True | True |
| success | seed2--issue-77-n1-001-a00--a06 | True | True | True | True |
| success | seed2--issue-77-n1-006-a05--a05 | True | True | True | True |
| success | seed2--issue-77-n1-010-a06--a02 | True | True | True | True |
| success | seed2--issue-77-n1-011-a02--a04 | True | True | True | False |
| success | seed2--issue-77-n1-012-a08--a04 | True | True | True | True |
| success | seed2--issue-77-n1-014-a01--a04 | True | True | True | True |
| success | seed2--issue-77-n1-016-a03--a03 | True | True | True | True |
| control | seed2--issue-77-n1-003-a00--a01 | False | False | True | True |
| control | seed2--issue-77-n1-004-a07--a02 | False | False | True | False |
| control | seed2--issue-77-n1-007-a05--a01 | False | False | True | True |
| control | seed2--issue-77-n1-008-a05--a01 | False | False | True | True |
| control | seed2--issue-77-n1-009-a00--a01 | False | False | True | False |

## 2. Capture channel (DESCRIPTIVE)

Decision frames byte-identical to the retained first-seed frames: 19 of 26 executed slots. frames differing while verdicts agree is the informative separation (selection channel noisy, outcome channel invariant); byte-identical frames are the stronger invariance outcome; DESCRIPTIVE.

## 3. Selection re-scored on both frames (DESCRIPTIVE)

| system | seed | states | chosen-ordinal agreement |
| --- | --- | --- | --- |

## 4. Disposition

**supported** (flips 0; guards ok True; stop reason None).

## 5. Claim boundary

re-execution of 26 frozen (state, ordinal) slots at one frozen second seed value with the scenario authority, decision step, inventory and actions unchanged; single-shot, decision-only re-scoring, development/exposed N1 lineages only; #64/#65 stay sealed; #87/#89/#90 published verdicts are inputs and are never recomputed, amended, or reinterpreted; no retraining, no method changes, no multi-shot or complete-gameplay claim; capture-channel and selection comparisons are DESCRIPTIVE.
- one second seed value: the probe measures invariance across one frozen perturbation of the Unity RNG, not a distribution over seeds
- NOVPHY_ENVIRONMENT_SEED initializes the Unity RNG only; whatever the episode does not randomize is unchanged by construction, so byte-identical frames are a possible and informative outcome
- the re-scoring covers the 12 #87 model cells on the composition's distinct states; it is a selection-channel reading, not a new selection-validity estimate
- the remaining scope caps (single-shot, decision-only, development lineages, sealed benchmark untouched) stay as disclosed limitations
