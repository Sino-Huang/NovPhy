# Issue-92 WP1: selection-validity re-statement — findings

- identity `issue-92-selection-validity-v1`, plan schema `issue_92_selection_validity_plan_v1` version 1 (terminal), frozen 2026-09-22T14:54:42Z before any statistic
- validation command: `python -u -m scripts.run_selection_validity_restatement --validate`
- measured wall 0.389s of 300.0s cap; zero GPU seconds; every interval DESCRIPTIVE

## 1. Unit of analysis (the correction every headline number depends on)

The #87 membership is **24 states over 15 source members**; 9 members carry two states that are byte-identical duplicates — same frozen inventory element-for-element, same engine seed, same decision anchor — and their per-ordinal engine-truth verdict vectors are identical (9/9 pairs verified, 0 discordant entries). The member is the primary unit; state-level numbers remain as execution detail.

Duplicate-pair identity evidence:
- `issue-77-n1-001` ['issue-77-n1-001-a00', 'issue-77-n1-001-a07']: inventory identical True, engine seed identical True, decision anchor identical True, discordant verdict entries 0
- `issue-77-n1-003` ['issue-77-n1-003-a00', 'issue-77-n1-003-a10']: inventory identical True, engine seed identical True, decision anchor identical True, discordant verdict entries 0
- `issue-77-n1-006` ['issue-77-n1-006-a05', 'issue-77-n1-006-a11']: inventory identical True, engine seed identical True, decision anchor identical True, discordant verdict entries 0
- `issue-77-n1-007` ['issue-77-n1-007-a05', 'issue-77-n1-007-a12']: inventory identical True, engine seed identical True, decision anchor identical True, discordant verdict entries 0
- `issue-77-n1-008` ['issue-77-n1-008-a05', 'issue-77-n1-008-a12']: inventory identical True, engine seed identical True, decision anchor identical True, discordant verdict entries 0
- `issue-77-n1-009` ['issue-77-n1-009-a00', 'issue-77-n1-009-a09']: inventory identical True, engine seed identical True, decision anchor identical True, discordant verdict entries 0
- `issue-77-n1-011` ['issue-77-n1-011-a02', 'issue-77-n1-011-a11']: inventory identical True, engine seed identical True, decision anchor identical True, discordant verdict entries 0
- `issue-77-n1-014` ['issue-77-n1-014-a01', 'issue-77-n1-014-a10']: inventory identical True, engine seed identical True, decision anchor identical True, discordant verdict entries 0
- `issue-77-n1-016` ['issue-77-n1-016-a03', 'issue-77-n1-016-a12']: inventory identical True, engine seed identical True, decision anchor identical True, discordant verdict entries 0

## 2. Member-unit ceiling and the chance references

- **member-unit ceiling 7/14 = 0.5000**, descriptive 95% interval [0.2143, 0.7857] (10000 draws, seed 7201, member-clustered)
- cited state-unit context (never recomputed): #89 completed 12/23 = 0.5217 [0.30434782608695654, 0.7391304347826086] state-unit before deduplication; typed-unmeasurable ['issue-77-n1-005-a07']
- one uniform top-1 draw per ceiling member misses everywhere with probability **0.5671**; member-clustered 12-cell pooled chance 1.106e-03; one system x 3 seeds 1.824e-01; (12/13)^7 reference 0.5710
- breadth extrapolation (ESTIMATE): miss-all 0.297 at 15 ceiling members, 0.088 at 30 — breadth does not rescue a zero-selection count

## 3. Selection validity (the structural, clustering-invariant pathology)

- **AUC 0.4367** [0.4045, 0.4710] over 108 model cells (P(successful ordinal receives lower predicted cost)); member-clustered **0.4180** [0.3135, 0.5252] over 7 members
- top-1 **0/108**; top-3 **6/108 = 0.0556** against inventory-matched uniform chance 0.2340; top-3 paired difference -0.1859 [-0.2308, -0.1279] (member-clustered)
- success band (engine-truth successful ordinals over the 12 sealed ceiling states) = **[2, 3, 4, 5, 6]**; the systems' chosen ordinals lie in the band in **0 of 207** scored model cells; chosen support {7: 90, 8: 12, 9: 45, 10: 10, 11: 6, 12: 44}
- region counts on the sealed ceiling states: ordinals 0-1: 0/24 successes; ordinals 2-6: 12/60; ordinals 7-12: 0/70
- no-model ordinal prior (execution detail): chose ordinals {8: 36} on the ceiling states with 0/36 top-1 hits; its own ordinal 8 = (-47, 65) sits in the high-arc band and scores zero — the prior is retired as any kind of baseline

**Integrity disclosure (printed, never omitted).** The one excluded state `issue-77-n1-005-a07` — no sealed anchor under #89's frozen typed-unmeasurable declaration — succeeds at **ordinal 7**, inside the model-chosen ordinal support; its typed-failure cell (ordinal 2, sealed-frame stability violation) is retained in both #87 and #89 and carries no success value.

Exposure split (DESCRIPTIVE):
- training: ceiling members ['issue-77-n1-001', 'issue-77-n1-006', 'issue-77-n1-010', 'issue-77-n1-011', 'issue-77-n1-012'], 72 cells, AUC mean 0.4838, top-1 0/72, top-3 6/72 (0.0833)
- calibration: ceiling members ['issue-77-n1-014', 'issue-77-n1-016'], 36 cells, AUC mean 0.3426, top-1 0/36, top-3 0/36 (0.0000)

## 4. The ordering mechanism

- median |Spearman rho| between candidate ordinal and predicted cost: **0.8736** (per system {"continuous-fixed-h1": 0.48901098901098894, "continuous-fixed-h5": 0.6083916083916086, "hybrid-fixed-h1": 0.9945054945054945}); drag_x 0.8776, drag_y 0.8791
- predicted-cost variance decomposition over 2565 retained ranking rows: between-states **0.360**, between-actions **0.004**, residual 0.636
- mechanism sentence: *the frozen predictors learn state difficulty but order actions within a state by the sweep coordinate, which is why the successful action is never ranked first and discriminates at or below chance on the engine-truth channel*. The between-state component is large; state-blindness is NOT claimed.
- launch-regime reading (scoped to the N1 type010103/type010105 membership): the 13 candidates are a monotone launch-angle sweep (a0 = (-80, 10) flat to a12 = (-11, 79) steep); every engine-truth success sits at ordinals 2-6 (flat/mid direct shots) while all model systems choose ordinals 7-12 (high-arc lobs) in 207/207 model cells; the rule breaks ties toward the lower ordinal, so the high-ordinal concentration is a genuine preference, not tie-breaking.

## 5. Determinism (WP2 primary: a measured statement, never an assumption)

The frozen #77 replay branches and the #87/#89 closed-loop executions are separate process runs, months apart, under different runner modules, same engine seed and action: their engine-truth verdicts agree on **176/176 unique (member, ordinal) branches** and on **288/288 verdict slots**, with 0 channel anomalies and 0 discordant entries across the 9 duplicate pairs. The closed-loop pass is a deterministic re-observation given (level, engine seed, action).

## 6. Accounting corrections (before a referee finds them)

- membership: **15 source members**, not 12; 9 duplicate pairs disclosed above
- #87-only decision accounting: of 288 decision cells, 230 bound, 46 unbound, 12 typed decision failures (3 per system on issue-77-n1-005-a07)
- partial ceiling (pre-completion): 84 cells -> 82 bound, 2 unbindable
- completed ceiling: **144/144 bound, 0 successes** — the zero-selection sentence is valid only on this membership and only because the oracle table was completed by #89 afterwards
- decision-side provenance (recommended wording): "144 of 144 ceiling-state decision cells bound (12 ceiling states x 4 systems x 3 seeds), binding completed against the #89-extended oracle table; across all 24 states #87's decision pass binds 230 of 288 cells with 12 typed decision failures."
- ORACLE_SEED: the record `seed` field takes values [20260908, 20260909, 20260910] — a #74 matched training-seed protocol label (scripts/run_closed_loop_oracle_probe.py:126), not a random draw; the engine receives exactly one seed, NOVPHY_ENVIRONMENT_SEED = state['engine_seed'] (probe line 858; per member 764100001 + member ordinal: {"issue-77-n1-001": [764100001], "issue-77-n1-003": [764100003], "issue-77-n1-004": [764100004], "issue-77-n1-005": [764100005], "issue-77-n1-006": [764100006], "issue-77-n1-007": [764100007], "issue-77-n1-008": [764100008], "issue-77-n1-009": [764100009], "issue-77-n1-010": [764100010], "issue-77-n1-011": [764100011], "issue-77-n1-012": [764100012], "issue-77-n1-013": [764100013], "issue-77-n1-014": [764100014], "issue-77-n1-015": [764100015], "issue-77-n1-016": [764100016]})
- #90 token verbatim: audit verdict `indeterminate` (Q1 `readiness_or_precision_insufficient`); the audit detected outcome-correlated missingness (5 of 52 timed-out slots succeeded on re-execution against 7 of 234 originally executed), which required the completion pass — the protocol working, not a live threat

## 7. Verdicts

- q1_selection_validity: **supported** (conditions: AUC shift True, top-3 shift True, disjoint supports True; member-clustered AUC interval covers 0.5: True — published as-is under the frozen stop rule)
- q2_ordering_mechanism: **supported** (monotonicity True, action share True)
- q3_determinism: **supported**
- precision guards ok: True ({"bootstrap_usable_min_fraction": true, "ceiling_model_cells": true, "duplicate_discordant_entries": true, "duplicate_pairs": true, "replay_join_min_fraction": true, "scored_model_cells": true, "verdict_slots": true})

## 8. Claim boundary and limitations

re-statement and measurement over retained records only; #87's, #89's and #90's published verdicts are inputs and are never recomputed, amended, or reinterpreted; #64/#65 stay sealed; closed-loop claims stay single-shot, decision-only, development-lineage scoped; no engine access, no rendering, no new data; intervals are DESCRIPTIVE.
- development/exposed N1 lineages only; not the sealed #64/#65 benchmark
- the member unit removes the duplicate inflation but the 14 measurable members remain a small, clustered sample; every interval is DESCRIPTIVE
- the selection measurements bind the three frozen fixed-horizon model systems of #87 plus the ordinal prior; no adaptive or endpoint arm is scored here (ticket #92 WP1b)
- single oracle protocol label (ORACLE_SEED is a #74 matched training-seed label, not a random draw); the engine's only seed is the per-member NOVPHY_ENVIRONMENT_SEED
- the AUC cell-unit interval is resampled over 108 cells that cluster inside 7 members; the member-clustered interval is the honest precision statement and is published alongside
- no content hash is recomputed after the freeze and no full-corpus integrity pass runs anywhere; inputs are verified at read time by their carried schema/identity/plan_identity fields
