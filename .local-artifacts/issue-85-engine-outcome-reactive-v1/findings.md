# Issue-85 engine-side outcome channel and bounded reactive re-diagnostic - findings

Diagnostics complete: False. Ledger status: complete.

Claim boundary: bounded non-final diagnostic on development lineages; no multi-shot, adaptation, zero-shot, or complete-gameplay claim; cannot reopen #64/#65/#72/#75/#15; the #80/#82 published typed outcomes are never re-run, amended, or reinterpreted; this is a NEW experiment with a NEW predicate; descriptive intervals only

## Phase A - the engine-side outcome channel

Detection rule: engine-truth first-shot success is read from the game's own state stream of the executed shot segment (the engine's canonical native fixed-step capture written during rendered play; no parser-derived cost enters the verdict). Primary event leg: pig_removed = the segment records at least one native macro event of type 'pig_removed', 'entity_death' or 'entity_destroyed' with at least one participant whose identity starts 'runtime:pig:' (the game's own pig-removal/death/destruction hooks). structure_destroyed is the same rule over 'runtime:block:' participants. Lifecycle leg (confirmatory): at least one 'runtime:pig:' entity whose per-sample lifecycle in the fixed-step state stream is 'destroyed'. The published verdict is the event leg; the lifecycle leg is published alongside for agreement review against the retained frames

Failure modes: declared failure modes: (1) a death event with an unresolvable participant identity (empty participants) would hide which entity died - detected by the lifecycle leg and typed as a channel anomaly; (2) a pig killed at the very end of the bounded native window may fire its death event before the registry marks the entity destroyed - event leg true with lifecycle leg false is a declared timing mode, recorded, NOT an anomaly; (3) the state stream records a destroyed pig while the event leg is silent - the actual missing-event failure mode - typed as a channel anomaly on that cell; (4) window censoring before any pig death leaves both legs false: the pig survived within the bounded window, not an anomaly

Channel verified: True.

- verification--positive--issue-77-n1-010-a06 role=positive expected pig_removed=True -> engine pig_removed=True consistency=agreement failure=none
- verification--positive--issue-77-n1-016-a12 role=positive expected pig_removed=True -> engine pig_removed=True consistency=agreement failure=none
- verification--negative--issue-77-n1-001-a00 role=negative expected pig_removed=False -> engine pig_removed=False consistency=agreement failure=none
- verification--negative--issue-77-n1-009-a00 role=negative expected pig_removed=False -> engine pig_removed=False consistency=agreement failure=none

## Phase C - #82 proxy predicate vs engine truth on the pilot (DESCRIPTIVE)

Rule: proxy success = executed realized end-of-window count cost <= the #82 published candidate threshold 81.68825840950012; engine truth = the engine-side channel verdict; DESCRIPTIVE. Compared 47 valid pilot cells (1 typed-failure cells excluded); agreement 0.1064 (both-success 0, proxy-only 42, engine-only 0, both-failure 5).

Paired proxy-minus-engine differences over pilot states (seeds averaged first; DESCRIPTIVE): mean +0.8958, descriptive 95% interval [+0.6875, +1.0000] over 4 state identities (positive = the proxy over-calls success).

## Engine-truth prevalence pilot gate

Prevalence 0.0000 (0/47 valid executions; 1 typed failures); floor 0.10; passed=False.

## Pilot compute accounting (executed cells; the gate barred Q1 scoring and the full run)

| System | Seed | Executed | Successes | Candidate counts | Transition calls | Linear MACs | Decision wall s | Engine wall s | Wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid-fixed-h1 | 20260908 | 4 | 0 | 9 12 13 | 47 | 67325056 | 0.546 | 112.6 | 183.2 |
| continuous-fixed-h1 | 20260908 | 4 | 0 | 9 12 13 | 47 | 87053400 | 0.526 | 129.6 | 199.7 |
| continuous-fixed-h5 | 20260908 | 4 | 0 | 9 12 13 | 235 | 435267000 | 0.537 | 130.5 | 198.1 |
| no-model-ordinal-prior | 20260908 | 4 | 0 | 9 12 13 | 0 | 0 | 0.367 | 117.9 | 187.3 |
| hybrid-fixed-h1 | 20260909 | 3 | 0 | 9 13 | 35 | 50135680 | 0.564 | 129.4 | 200.7 |
| continuous-fixed-h1 | 20260909 | 4 | 0 | 9 12 13 | 47 | 87053400 | 0.538 | 128.3 | 207.4 |
| continuous-fixed-h5 | 20260909 | 4 | 0 | 9 12 13 | 235 | 435267000 | 0.653 | 129.8 | 198.0 |
| no-model-ordinal-prior | 20260909 | 4 | 0 | 9 12 13 | 0 | 0 | 0.400 | 123.8 | 194.7 |
| hybrid-fixed-h1 | 20260910 | 4 | 0 | 9 12 13 | 47 | 67325056 | 0.616 | 114.5 | 190.3 |
| continuous-fixed-h1 | 20260910 | 4 | 0 | 9 12 13 | 47 | 87053400 | 0.512 | 137.0 | 205.3 |
| continuous-fixed-h5 | 20260910 | 4 | 0 | 9 12 13 | 235 | 435267000 | 0.577 | 128.9 | 198.8 |
| no-model-ordinal-prior | 20260910 | 4 | 0 | 9 12 13 | 0 | 0 | 0.413 | 120.1 | 194.4 |

- continuous: frozen predictor parameters [1862396] across seeds ['20260908', '20260909', '20260910']
- hybrid: frozen predictor parameters [1837690] across seeds ['20260908', '20260909', '20260910']
- continuous-fixed-h1: active predictor parameters per transition [1862396]; shared tables counted in full
- continuous-fixed-h5: active predictor parameters per transition [1862396]; shared tables counted in full
- hybrid-fixed-h1: active predictor parameters per transition [1441516]; shared tables counted in full
- no-model-ordinal-prior: active predictor parameters per transition [0]; shared tables counted in full

Per-cell wall/step/candidate rows are published in summary.json pilot_compute_accounting.per_cell. These are accounting tables, not Q1 scoring; Q1 stays barred by the gate.

Full-matrix execution incomplete; Q1 paired outcomes are not scored.

Ticket disposition: readiness_or_precision_insufficient (tokens: supported / not_supported_by_this_experiment / readiness_or_precision_insufficient).

- q1_engine_truth_reactive_differences: readiness_or_precision_insufficient
- q2_proxy_engine_agreement: supported

- development/exposed N1 lineages only; zero fresh captures; not the sealed #64/#65 benchmark
- the engine-truth verdict covers the bounded 12 s native shot window per executed shot
- regret references the t=600 end-of-window replay cost: right-censored, not a settled cost
- states sharing a source member share one physical initial state and candidate table; the paired bootstrap over 24 state identities is descriptive only
- inference is NOT equalized across arms; per-arm decision compute is reported
- no adaptive system is present; the #74 hybrid_adaptive wall context carries no penalty narrative
