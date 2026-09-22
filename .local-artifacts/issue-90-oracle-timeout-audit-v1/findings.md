# Issue-90 timeout-outcome correlation audit for the #87 oracle pass (analysis-only)

Frozen protocol: plan.json version 1, frozen 2026-09-22T11:23:03Z before any audit statistic. Zero re-execution, zero rendering, zero GPU work; every input READ-ONLY. All intervals DESCRIPTIVE.

Audit verdict (frozen vocabulary): `indeterminate`
Q1 covariate-shift disposition: `readiness_or_precision_insufficient`

## Universe and strata

| stratum | slots | states | role |
| --- | --- | --- | --- |
| executed | 234 | 23 | reference stratum of every contrast |
| timeout | 52 | 18 | target stratum of every contrast |
| stability | 3 | 1 | separate typed stratum, never pooled into the timeout contrast |

Join coverage of the verdict-relevant sources: 1.000000 of 289 slots; the timeout stratum spans 18 states. Every slot carries a typed per-source join status in slot_join.csv.

## Q1 - frozen covariate contrasts (timeout minus executed, descriptive)

| contrast | covariate | target | reference | difference | descriptive 95% interval | margin | shift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| d1_normalized_cost_position | normalized_cost_position | 0.513695 | 0.515444 | -0.001749 | [-0.088374, 0.095617] | 0.100000 | no_shift |
| d2_frozen_replay_count_cost | frozen_replay_count_cost | 68.447531 | 77.044335 | -8.596805 | [-32.819035, 7.102048] | 4.228765 | borderline |
| d3_frozen_replay_pig_removed | frozen_replay_pig_removed | 0.096154 | 0.029915 | +0.066239 | [-0.005855, 0.139046] | 0.100000 | no_shift |
| d4_type010105_share | is_type010105 | 0.634615 | 0.525641 | +0.108974 | [-0.089437, 0.286821] | 0.150000 | no_shift |
| d5_state_best_observed_pig_removed | state_best_observed_pig_removed | 0.576923 | 0.534188 | +0.042735 | [-0.156210, 0.237711] | 0.100000 | no_shift |
| d6_ordinal_position | ordinal | 6.019231 | 5.918803 | +0.100427 | [-0.859934, 0.932201] | 1.000000 | descriptive_no_shift |

`shift` means the absolute difference reaches the frozen margin and the descriptive interval excludes 0; `borderline` means it reaches the margin with the interval including 0. Only the five verdict-relevant contrasts enter the verdict; d6_ordinal_position is reported for completeness because an inventory position is not outcome-linked by itself.

## Q2 - bias direction for #87's ceiling (typed, descriptive)

- Published #87 pooled ceiling: 7 of 23 measured states (0.304348); 24 scheduled states, unmeasured ['issue-77-n1-005-a07'].
- Published #87 family ceilings: type010103 4/12, type010105 3/12.

- `d1_normalized_cost_position` (no_shift, difference -0.001749, margin 0.100000, shift no_shift): no frozen bias direction is stated: this contrast shows no shift under the frozen margin and interval rule, so the timeout stratum is not distinguishable from the executed stratum on this covariate (DESCRIPTIVE)
- `d2_frozen_replay_count_cost` (timeout_lower, difference -8.596805, margin 4.228765, shift borderline): the raw frozen end-of-window count cost is lower for timed-out slots (more frozen progress); the unmeasured slots are disproportionately the better ones, so the published ceiling is plausibly understated (DESCRIPTIVE)
- `d3_frozen_replay_pig_removed` (no_shift, difference +0.066239, margin 0.100000, shift no_shift): no frozen bias direction is stated: this contrast shows no shift under the frozen margin and interval rule, so the timeout stratum is not distinguishable from the executed stratum on this covariate (DESCRIPTIVE)
- `d4_type010105_share` (no_shift, difference +0.108974, margin 0.150000, shift no_shift): no frozen bias direction is stated: this contrast shows no shift under the frozen margin and interval rule, so the timeout stratum is not distinguishable from the executed stratum on this covariate (DESCRIPTIVE)
- `d5_state_best_observed_pig_removed` (no_shift, difference +0.042735, margin 0.100000, shift no_shift): no frozen bias direction is stated: this contrast shows no shift under the frozen margin and interval rule, so the timeout stratum is not distinguishable from the executed stratum on this covariate (DESCRIPTIVE)
- `d6_ordinal_position` (no_shift, difference +0.100427, margin 1.000000, shift descriptive_no_shift): no bias direction is defined for an inventory position; the descriptive contrast is reported for completeness only (DESCRIPTIVE)

## Frozen indicator semantics (association context)

Within the executed stratum, slots whose frozen replay removed a pig show a published success rate of 1.000000 (n=7, states=7) against 0.000000 (n=227, states=23): difference +1.000000 with descriptive interval [1.000000, 1.000000]. This is context for the semantics of the frozen indicators and is never a verdict input.

## Mechanism inventory (descriptive; never a verdict input)

- executed: 234 slots over 23 states and 14 source members; wall seconds median 182.8 (max 322.4); engine wall seconds median 119.7; peak RSS median 1090.1 MiB; live_at_dispatch {'1': 1, '2': 12, '3': 77, '4': 144}; dispatch window 0.1 to 261.3 minutes after the first dispatch; published successes 7
- timeout: 52 slots over 18 states and 14 source members; wall seconds median 191.3 (max 191.4); engine wall seconds median not applicable; peak RSS median 954.3 MiB; live_at_dispatch {'1': 2, '2': 21, '3': 28, '4': 1}; dispatch window 0.0 to 244.6 minutes after the first dispatch; published successes 0
- stability: 3 slots over 1 state and 1 source member; wall seconds median 74.6 (max 74.9); engine wall seconds median not applicable; peak RSS median 990.5 MiB; live_at_dispatch {'2': 1, '4': 2}; dispatch window 44.1 to 46.7 minutes after the first dispatch; published successes 0

All-workers-live dispatch share (derived cold-start contention proxy): timeout 0.019231 versus executed 0.615385, difference -0.596154 with descriptive interval [-0.704817, -0.487555] (DESCRIPTIVE mechanism inventory).

## Stability stratum (typed, separate mechanism)

- slots ['oracle--issue-77-n1-005-a07--a01', 'oracle--issue-77-n1-005-a07--a02', 'oracle--issue-77-n1-005-a07--a07'] in states ['issue-77-n1-005-a07'] at ordinals [1, 2, 7]; signature 'sealed pre-decision state/RGB advanced during readiness or hold'. Typed note: reported as its own typed stratum with its frozen covariates; no interval is computed for a single-state stratum of 3 slots, and it is never pooled into the timeout contrast.
  - oracle--issue-77-n1-005-a07--a01: frozen cost 82.47732925415039, normalized position 1.0, frozen terminal kind 'stable_without_clear', frozen pig removal False, best-observed pig removal True
  - oracle--issue-77-n1-005-a07--a02: frozen cost 76.56465220451355, normalized position 0.6070423804066183, frozen terminal kind 'native_fail', frozen pig removal False, best-observed pig removal True
  - oracle--issue-77-n1-005-a07--a07: frozen cost 81.68825840950012, normalized position 0.9475582044814449, frozen terminal kind 'native_clear', frozen pig removal True, best-observed pig removal True

## Cited published cross-check

- source `issue-89-oracle-completion-v1` (`issue_89_oracle_completion_report_v1`); cited fields ['completion_pass.inventory', 'ceiling_reestimate.pooled', 'records/*.json outcome.first_shot_success', 'records/*.json issue87_typed_failure']
- published inventory: {"executed": 54, "residual_typed_failures_outside_declaration": [], "scheduled_cells": 55, "successful_cells": 6, "typed_failures": 1, "unreachability_declaration": {"consequence": "the ceiling questions are dispositioned over the 23 measured states with this exclusion declared", "declaration": "typed_unmeasurable", "media": "data/issue-89-oracle-completion/cells/", "rule": "frozen up front: if any issue-77-n1-005-a07 slot fails again with the sealed-frame stability signature ('sealed pre-decision state/RGB advanced during readiness or hold'), the state is declared typed-unmeasurable (the level physics is not static at the sealed decision step 30000, so verdicts cannot bind to the sealed pre-decision state); the stability evidence (paused before/after frames + records) and media are published, and the ceiling questions are dispositioned over the 23 measured states with the exclusion declared; the declaration triggers on the signature REGARDLESS of the other slots' verdicts", "signature": "sealed pre-decision state/RGB advanced during readiness or hold", "stability_failures": [{"failure": "ValueError: sealed pre-decision state/RGB advanced during readiness or hold", "identity": "oracle--issue-77-n1-005-a07--a02", "ordinal": 2}], "state": "issue-77-n1-005-a07"}}
- cited successes by #87 failure stratum: {"other": {"slots": 0, "successes": 0}, "stability": {"slots": 3, "successes": 1}, "timeout": {"slots": 52, "successes": 5}}
- attribution limit: a different execution infrastructure cannot be separated from slot selection by this row; it is a directional cross-check only

## Frozen protocol

- Frozen semantics: frozen replay evidence: for every audited slot the #77 N1 campaign replayed the same action from the same source-member decision state (native fixed step 30000, engine seed 764100001) BEFORE #87 executed anything, so the frozen pig-removal event, the frozen terminal kind and the frozen count cost are engine-truth pre-execution properties of the slot; the frozen count cost is the task objective (1000 * pig presence + block presence) at the frozen replay's observed window endpoint, where lower means more progress.
- Join keys: dispatch: issue-87 markers/<slot identity>.json dispatched_at_utc and receipts/<slot identity>.json peak_cpu_rss_mib; frozen_candidate_table: issue-80 candidate-outcomes/<source_member>.json row with the same ordinal (branch identity, action and count cost must agree with the issue-87 cell); frozen_replay: issue-77 N1 coverage branch keyed by the slot's branch identity, plus results/<branch identity>.json; headroom: issue-73 headroom registry keyed by its own state identity (typed unjoinable: the #73 registry holds issue-68 production-v2 calibration identities, disjoint from the issue-77 N1 membership identities, so the best-observed pig-removal indicator is computed from frozen #77 evidence instead, as the frozen substitution records); slot_identity: issue-87 oracle cell identity = state identity x candidate ordinal; state_to_source_member: issue-87 plan.json states[].source_member.
- Statistics: value(target stratum mean) minus value(reference stratum mean), where the mean runs over the slots of the stratum and boolean covariates are averaged as rates. Interval: state-clustered percentile bootstrap, 10000 draws, PCG64 seed 9001, unit state identity (every slot of a resampled state moves together), quantiles [0.025, 0.975], labelled DESCRIPTIVE; draws in which either side is empty are dropped and counted.
- Margins: d1_normalized_cost_position=0.1; d2_frozen_replay_count_cost=4.228764791041613; d3_frozen_replay_pig_removed=0.1; d4_type010105_share=0.15; d5_state_best_observed_pig_removed=0.1; d6_ordinal_position=1.0.
- Verdict mapping: shift_rule: a verdict-relevant contrast shows a shift iff the absolute difference is at least its margin AND its descriptive interval excludes 0; borderline_rule: a verdict-relevant contrast is borderline iff the absolute difference is at least its margin AND its interval includes 0; outcome_correlated: no precision guard tripped AND at least one verdict-relevant contrast shows a shift; outcome_blind: no precision guard tripped AND no verdict-relevant contrast shows a shift AND no verdict-relevant contrast is borderline; indeterminate: a precision guard tripped OR a verdict-relevant contrast is borderline.
- Precision guards: {"min_executed_slots": 100, "min_join_fraction": 0.95, "min_timeout_slots": 20, "min_timeout_states": 8, "min_usable_draws_fraction": 0.9}.

## Limitations and claim boundary

- development/exposed N1 lineages only; not the sealed #64/#65 benchmark
- the frozen covariates are engine truth for the frozen #77 replay, not for the #87 execution that timed out; per-slot execution variance between the two runs is unobserved
- states sharing a source member share one physical initial state and candidate table, so the 24 state identities carry fewer independent initial states; the state-clustered bootstrap is descriptive only
- the cold-start reading is inferred from published dispatch timestamps and per-cell wall seconds at one-second resolution; engine boot timestamps are not published, so live_at_dispatch is a proxy
- unobserved-confounder residual risk: any covariate not in the frozen list (per-worker engine identity, host load outside the published timestamps, port-level collisions) is unmeasured, and a timeout mechanism that acts through such a covariate would not be detected here
- no content hash is recomputed after the freeze and no full-corpus integrity pass runs anywhere; inputs are verified at read time by their carried schema/identity/plan_identity fields

- Claim boundary: descriptive audit of the published #87 oracle pass only; no re-execution, no rendering, no new data, zero GPU work; cannot reopen #87's or #89's published dispositions; no causal claim about the timeout mechanism beyond the frozen covariate set; intervals are DESCRIPTIVE.

## Protocol chronology

- version 1, frozen 2026-09-22T11:23:03Z before its scoring run: initial and terminal freeze: covariates, join keys, statistics, numeric margins, verdict mapping, bias-direction rules, precision guards and caps are published before any audit statistic exists. Why: standard freeze rule of the shared ADD-EXP protocol; the dry run is a structural inventory only and computes no statistic of any kind Evidence: no audit statistic has been computed at freeze time; the only quantity derived here is the raw-cost margin formula evaluated on the frozen candidate table, which involves no stratum and no contrast

## Compute accounting

- compute_statistics: 0.089s wall, peak RSS 686.4 MiB
- join_and_covariates: 0.312s wall, peak RSS 686.4 MiB
- Caps: 360.0s measured wall, 104857600 bytes derived artifacts, 0.0 GPU seconds.

## Reproduction

`python -u -m scripts.run_oracle_timeout_audit --validate` recomputes every published table from the frozen inputs and byte-compares it against the published artifacts.
