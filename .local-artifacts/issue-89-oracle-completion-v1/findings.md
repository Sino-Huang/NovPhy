# Issue-89 closed-loop oracle-ceiling completion pass - findings

Terminal protocol version: v3 (frozen 2026-09-22T09:50:17Z, before its scoring run). Ledger status: complete.

Claim boundary: bounded completion measurement on development/exposed lineages; zero fresh captures; no multi-shot, adaptation, zero-shot, or complete-gameplay claim; cannot reopen #64/#65/#72/#75/#15/#76; the #80/#82/#85/#87 published outcomes are read-only inputs, never re-run, amended, or reinterpreted; this ticket APPENDS the missing slots, it does not re-run #87; no system decision pass; descriptive intervals only

## Q1 - completion of the 55 typed-failure slots of #87

Completion inventory: 55 scheduled slots, 54 executed with an engine-truth channel verdict, 1 typed failures, 6 engine-truth-successful slots.
Residual typed failures outside the frozen declaration: 0.
Typed-unmeasurable declaration: issue-77-n1-005-a07 (1 slot(s) with the sealed-frame stability signature; stability evidence + media published).

## Q2 - ceiling re-estimate over the 24 frozen states

Engine-truth ceiling states: 12/23 measured states (pooled prevalence 0.5217); descriptive 95% interval [0.3043, 0.7391] over states (DESCRIPTIVE).
Excluded under the frozen typed-unmeasurable declaration: ['issue-77-n1-005-a07'].
Comparison to #87's published partial ceiling: 7/23 measured states, prevalence 0.3043, interval [0.1304, 0.5217] -> completed 12/23, prevalence 0.5217 (delta ceiling states 5, delta prevalence 0.2174).

## Dispositions

- q1_slot_completion: supported
  - reason: every scheduled slot carries an engine-truth channel verdict or is covered by the frozen typed-unmeasurable declaration
- q2_ceiling_reestimate: supported
  - reason: pooled prevalence 0.5217 > 0 and the descriptive 95% interval [0.3043, 0.7391] excludes 0 over the 23 measured states with ['issue-77-n1-005-a07'] excluded under the frozen typed-unmeasurable declaration

Ticket disposition: supported (tokens: supported / not_supported_by_this_experiment / readiness_or_precision_insufficient).

## Protocol chronology

- plan-v1 (development, frozen 2026-09-22T09:07:25Z): completion pass over exactly the 55 typed-failure oracle slots of the issue-87 terminal ledger (slot identities, actions, seed 20260908 and the inherited #85 detection rule verbatim; only execution infrastructure changes); frozen cold-start mitigation (staggered engine starts 45s, port warm-up probes 300s, extended connect/readiness bounds); frozen unreachability branch for issue-77-n1-005-a07; frozen disposition mapping; bounded 2-cell rendered smoke (1 former-timeout slot + 1 issue-77-n1-005-a07 slot); global caps 28800s wall / 1800s GPU; ports 28700+; 4 isolated workers Why: open the ticket under the frozen-before-scoring policy: nothing is yet observed, so v1 fixes the inherited measurement instruments, the mitigation, the unreachability branch and the bounded smoke that validates the harness before any long run Evidence: none yet (initial freeze, before the smoke)
- plan-v2 (development, frozen 2026-09-22T09:44:34Z): the bare-TCP port warm-up probes of plan-v1 are REMOVED: the v1 smoke showed they poison the engine session - the jar registers an anonymous probe connection as an agent and assigns the game window to the dead probe client, starving the real bridge (both v1 smoke cells typed-failed with TimeoutError at the 600s socket bound, wall ~622s each; engine logs show '2 agent/s has been connected'). The mitigation is now: serialized/staggered engine cold starts (45s dispatch spacing) plus the extended connect/readiness bounds, byte-identical to v1. NO other protocol element changed: the 55-slot completion membership, oracle seed 20260908, the inherited #85 detection rule, the unreachability branch, the disposition mapping, workers, port range, and caps are byte-identical. The v2 smoke re-validates the corrected harness on the same 2 frozen smoke slots before the terminal freeze. v1 is retained verbatim with its outcomes labelled EXPLORATORY. Why: a frozen mitigation element was shown to be self-defeating by the v1 smoke; correcting it during development (before any terminal freeze) preserves the freeze-before-scoring guarantee for the terminal version Evidence: 0/2 smoke cells executed real-rendered under the frozen mitigation, 0/2 matched their frozen structural expectations; channel verdicts and gallery media retained
- plan-v3 (terminal, frozen 2026-09-22T09:50:17Z): NO protocol element changed relative to plan-v2: the 55-slot completion membership, oracle seed 20260908, the inherited #85 detection rule, the corrected cold-start mitigation (staggered starts + extended bounds, no probes), the unreachability branch, the disposition mapping, workers, port range, and caps are byte-identical. What CHANGES with v3 is the status: development -> terminal, so this version is frozen with actual numeric values BEFORE its own scoring run, is executed exactly once, and admits no further outcome-conditioned modification. Reviewed smoke evidence embedded: 2/2 smoke cells executed real-rendered under the frozen mitigation, 2/2 matched their frozen structural expectations; channel verdicts and gallery media retained Why: the v2 smoke validated the corrected oracle harness end to end (real rendered execution under the frozen mitigation, channel verdict or protocol-covered stability signature, gallery media retention), so no instrument change is needed and the terminal version can be frozen before any terminal outcome exists Evidence: development/v2 records + data/issue-89-oracle-completion/development/v2

## Development-version outcomes (EXPLORATORY; not citable as answers)

- [EXPLORATORY] smoke-v1--issue-77-n1-001-a00--a00: executed=False pig_removed=None expectation_matched=False failure=TimeoutError: timed out
- [EXPLORATORY] smoke-v1--issue-77-n1-005-a07--a01: executed=False pig_removed=None expectation_matched=False failure=TimeoutError: timed out
- [EXPLORATORY] smoke-v2--issue-77-n1-001-a00--a00: executed=True pig_removed=False expectation_matched=True failure=none
- [EXPLORATORY] smoke-v2--issue-77-n1-005-a07--a01: executed=True pig_removed=False expectation_matched=True failure=none

## Compute accounting

Global wall consumed 4450s of 28800s cap; active decision GPU seconds 0 of 1800s cap (no decision pass in this ticket; the GPU cap is a guard).

## Limitations

- development/exposed N1 lineages only; zero fresh captures; not the sealed #64/#65 benchmark
- the engine-truth verdict covers the bounded native shot window per executed shot
- the oracle pass uses the single frozen seed 20260908; per-state indicators are one engine-truth draw per candidate slot
- states sharing a source member share one physical initial state and candidate table; the paired bootstrap over state identities is descriptive only
- this ticket APPENDS the 55 missing slots of #87; #87's published outcomes are read-only inputs, never amended or reinterpreted; no system decision pass here
- a typed-unmeasurable declaration for issue-77-n1-005-a07 excludes that state from the ceiling estimate with the exclusion declared
