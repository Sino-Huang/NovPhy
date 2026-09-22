# Issue-87 closed-loop oracle-ceiling probe - findings

Terminal protocol version: v3 (frozen 2026-09-22T02:45:51Z, before its scoring run). Ledger status: complete.

Claim boundary: bounded non-final diagnostic on development/exposed lineages; zero fresh captures; no multi-shot, adaptation, zero-shot, or complete-gameplay claim; cannot reopen #64/#65/#72/#75/#15/#76; the #80/#82/#85 published outcomes are read-only inputs, never re-run, amended, or reinterpreted; descriptive intervals only

## Q1 - closed-loop oracle ceiling over the 24 frozen states

Engine-truth ceiling states: 7/24 (pooled prevalence 0.3043); descriptive 95% interval [0.1304, 0.5217] over states (DESCRIPTIVE).
Oracle inventory: 289 scheduled candidate cells, 234 executed, 55 typed failures, 7 engine-truth-successful cells.

- type010103: 4/12 ceiling states, 4 successful cells of 111 executed (22 typed failures).
- type010105: 3/12 ceiling states, 3 successful cells of 123 executed (33 typed failures).

## Q2 - ranking gap on ceiling states (per frozen system)

| System | Ceiling states | Cells | Successes | Rate | Descriptive 95% interval |
| --- | --- | --- | --- | --- | --- |
| hybrid-fixed-h1 | 7 | 20 | 0 | 0.000 | 0.000, 0.000 |
| continuous-fixed-h1 | 7 | 20 | 0 | 0.000 | 0.000, 0.000 |
| continuous-fixed-h5 | 7 | 21 | 0 | 0.000 | 0.000, 0.000 |
| no-model-ordinal-prior | 7 | 21 | 0 | 0.000 | 0.000, 0.000 |
- Model systems pooled: mean 0.000, descriptive interval [0.000, 0.000] over ceiling states (DESCRIPTIVE; pre-declared Q2 margin 0.5).

## Dispositions

- q1_closed_loop_oracle_ceiling_nonzero: readiness_or_precision_insufficient
  - reason: no candidate executed on 1 state(s): ['issue-77-n1-005-a07']
- q2_ranking_gap_on_ceiling_states: supported
  - reason: every model system's pooled successful-selection rate over ceiling states x 3 seeds is < 0.5 and the pooled descriptive interval [0.0000, 0.0000] lies entirely below 0.5

Ticket disposition: readiness_or_precision_insufficient (tokens: supported / not_supported_by_this_experiment / readiness_or_precision_insufficient).

## Protocol chronology (iterative policy)

- plan-v1 (development, frozen 2026-09-22T02:17:27Z): oracle pass over every admissible candidate of the #80 frozen membership at seed 20260908 with the inherited #85 detection rule; DECISION-ONLY reactive scoring per system x seed x state anchored on each state's sealed pre-decision frame; bounded 2-cell rendered smoke on published #85 positive/negative slots; global caps 24 worker-hours wall / 3 GPU-hours active decision work; ports 28700+; 4 isolated workers Why: open the ticket under the iterative policy: nothing is yet observed, so v1 fixes the inherited measurement instruments and the bounded smoke that validates the harness before any long run Evidence: none yet (initial freeze, before the smoke)
- plan-v2 (terminal, frozen 2026-09-22T02:41:38Z): NO protocol element changed relative to plan-v1: the membership, the oracle slot inventory (289 scheduled candidate cells), oracle seed 20260908, the inherited #85 detection rule, the #85 reactive decision rule, the anchor rule, workers, port range, and caps are byte-identical. The pre-declared analysis margins were already fixed in plan-v1 (Q2 ranking-gap margin 0.5, Q1 interval-exclusion rule). What CHANGES with v2 is the status: development -> terminal, so this version is frozen with actual numeric values BEFORE its own scoring run, is executed exactly once, and admits no further outcome-conditioned modification. Reviewed smoke evidence embedded: 0/2 smoke cells executed real-rendered, 0/2 matched their frozen #85 expectations; channel verdicts and gallery media retained Why: the smoke validated the oracle harness end to end (real rendered execution, channel verdict, gallery media retention), so no instrument change is needed and the terminal version can be frozen before any terminal outcome exists Evidence: development/v1 records + data/issue-87-closed-loop-oracle/development/v1
- plan-v3 (terminal, frozen 2026-09-22T02:45:51Z): NO protocol element changed relative to plan-v1: the membership, the oracle slot inventory (289 scheduled candidate cells), oracle seed 20260908, the inherited #85 detection rule, the #85 reactive decision rule, the anchor rule, workers, port range, and caps are byte-identical. The pre-declared analysis margins were already fixed in plan-v1 (Q2 ranking-gap margin 0.5, Q1 interval-exclusion rule). What CHANGES with v2 is the status: development -> terminal, so this version is frozen with actual numeric values BEFORE its own scoring run, is executed exactly once, and admits no further outcome-conditioned modification. Reviewed smoke evidence embedded: 2/2 smoke cells executed real-rendered, 2/2 matched their frozen #85 expectations; channel verdicts and gallery media retained Why: the smoke validated the oracle harness end to end (real rendered execution, channel verdict, gallery media retention), so no instrument change is needed and the terminal version can be frozen before any terminal outcome exists Evidence: development/v1 records + data/issue-87-closed-loop-oracle/development/v1
- plan-v3 (terminal, frozen 2026-09-22T02:45:51Z): NO scientific or protocol element changed relative to plan-v2 (membership, 289-slot oracle inventory, oracle seed, detection rule, decision rule, anchor rule, margins, caps, workers, ports are byte-identical). The ONLY defect: plan-v2's embedded smoke evidence was read from the wrong records directory and reported '0/2 smoke cells executed, 0/2 matched' when the retained development records show 2/2 smoke cells executed real-rendered, 2/2 matched their frozen #85 expectations; channel verdicts and gallery media retained. plan-v2 is retained verbatim at protocols/plan-v2.json with its factual error; this version is the terminal freeze, made BEFORE any scoring-run execution exists (zero terminal oracle records at freeze time). Why: a frozen plan must not carry a factually wrong evidence field; correcting it before the scoring run preserves the freeze-before-outcome guarantee instead of compromising it Evidence: development/v1 records + data/issue-87-closed-loop-oracle/development/v1

## Development-version outcomes (EXPLORATORY; not citable as answers)

- [EXPLORATORY] smoke-v1--issue-77-n1-010-a06--a02: executed=True pig_removed=True expectation_matched=True failure=none
- [EXPLORATORY] smoke-v1--issue-77-n1-009-a00--a00: executed=True pig_removed=False expectation_matched=True failure=none

## Compute accounting

Global wall consumed 16466s of 86400s cap; active decision GPU seconds 6 of 10800s cap.
- continuous-fixed-h1: 69 scored cells, 855 transition calls, 1583631000 linear MACs, 1.0s cuda decision wall, candidate counts 9-13.
- continuous-fixed-h5: 69 scored cells, 4275 transition calls, 7918155000 linear MACs, 3.4s cuda decision wall, candidate counts 9-13.
- hybrid-fixed-h1: 69 scored cells, 855 transition calls, 1224743040 linear MACs, 1.5s cuda decision wall, candidate counts 9-13.
- no-model-ordinal-prior: 69 scored cells, 0 transition calls, 0 linear MACs, 0.3s cuda decision wall, candidate counts 9-13.
- continuous: frozen predictor parameters [1862396] across seeds ['20260908', '20260909', '20260910']
- hybrid: frozen predictor parameters [1837690] across seeds ['20260908', '20260909', '20260910']

## Limitations

- development/exposed N1 lineages only; zero fresh captures; not the sealed #64/#65 benchmark
- the engine-truth verdict covers the bounded native shot window per executed shot
- the oracle pass uses the single frozen seed 20260908; per-state indicators are one engine-truth draw per candidate slot
- states sharing a source member share one physical initial state and candidate table; the paired bootstrap over 24 state identities is descriptive only
- inference is NOT equalized across arms; per-arm decision compute is reported; the decision pass is DECISION-ONLY (no gameplay for system cells)
- the shared-GPU flock was not taken (batch coordination with the concurrent issue-88 ticket); GPU use is bounded to the short decision pass
