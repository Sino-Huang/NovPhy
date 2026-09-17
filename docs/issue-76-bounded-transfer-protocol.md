# #76 bounded input-corrected transfer study

Identity: `issue-76-bounded-transfer-v1`.

This is a strict bounded sibling of the parked §8 shared-development draft, NOT
its execution. It is an exploratory engineering study on development data only.
Its purpose is to decide whether correcting the demonstrated RGB input-contract
defect materially improves transfer while separating that effect from history,
architecture, loss and data changes. It is not readiness, fresh-access or
advancement evidence, and it does not change any earlier result or disposition.

## Estimand and labels

For each held-out base lineage, evaluate all 13 prospectively fixed actions using
only the assigned predecision RGB/history view and the action candidate. The unit
of analysis is the independent base lineage. The estimand is whether each system
selects one action whose observed branch reaches `native_clear` during its genuine
terminal or intact 12-second native observation window. The 13 action branches
are counterfactual choices within one group, not 13 independent units.

The predictor endpoint at 11,250 native steps (4.5 seconds) is only a predicted
latent feature supplied to the existing event readout. It is never relabelled as
a 12-second observed outcome. Preserve the four existing stop kinds exactly:
`native_clear`, `native_fail`, `stable_without_clear` and `right_censored`. There
is no `later_clear_by_deadline_label`. Stable and right-censored branches remain
typed as observed and are never converted to later-clear negatives.

## Membership and scope

Claims are limited to normal families `type010101` and `type010105`. For each
family reserve prospectively, without outcome screening:

- 20 readout-TRAIN lineages at deterministic ranks 0--19;
- 10 selector-TRAIN lineages at deterministic ranks 75--84; and
- 20 held-out MODEL_SELECTION lineages at deterministic ranks 120--139.

This yields 50 lineages per family, 100 independent base lineages total. Family
indexing follows the §8 five-family order. Lineage ordinals run 1..100 in
deterministic membership order; generation seeds are `761900000+ordinal` and
engine seeds `762000000+ordinal`, deliberately disjoint from the parked §8
ranges (`761700001..761700700` / `761800001..761800700`) so a later §8 can
never collide with this study. Bind exact template/workbook
constraints, XML, scenario lineage and seeds before rendering, and check metadata
disjointness against all prior fitting, exposure and reserved inventories; these
binding and disjointness conventions are adopted-from-§8.

Every lineage has the same 13 one-shot fixed actions: reference drag `(-80,10)`
followed by the 12 upward angles 5, 12, ..., 82 degrees at commanded radius 80
pixels, rounded coordinate-wise. There is no tap. Every branch requests
`release_time_ms=1000` and retains the recovered player's actual one-second
release semantics. Coordinate-wise rounding is adopted-from-§8. The complete
scheduled inventory is 1,300 captured branches.

TRAIN and MODEL_SELECTION roles remain distinct. Readout-TRAIN records fit the
conditioned event readout; selector-TRAIN records fit the selector; held-out
MODEL_SELECTION lineages provide the bounded transfer comparison. No lineage,
branch or role may be moved after outcomes or failures are observed.

## Input variants and contrasts

Produce three paired input views from the same native states and attach the same
native branch labels and failure mask to all views:

1. `legacy-single`: old HUD-inclusive request-72-equivalent RGB, latest frame
   only, represented as honest single-frame history.
2. `corrected-single`: corrected world-camera RGB at the identical decision
   state, latest frame only.
3. `corrected-history`: corrected world-camera RGB captured at actual native
   steps -100, -50 and 0 relative to the decision.

The latest corrected frame in `corrected-single` and `corrected-history` is the
same decision-state observation. Never repeat one image as pseudo-history and
never expose post-action frames as predecision input. A missing paired input
invalidates that branch for every arm; no arm receives a more favorable branch
set.

Freeze and report these paired contrasts:

- corrected-single minus legacy-single: RGB-path effect;
- corrected-history minus corrected-single: history effect; and
- corrected-history minus legacy-single: total input-contract effect.

The RGB-path contrast is within the new paired inventory. It is not a causal
contrast between v4 records and old 600 ms records.

## Model arms and matched exposure

The primary hybrid arms use the frozen balanced-native-v2 HYBRID predictor
ensemble seeds `760930001`, `760930002` and `760930003` from
`.local-artifacts/issue-76-balanced-native-v2/checkpoints/predictor-hybrid-<seed>.pt`.
For every model-backed arm, input variant and assigned predictor seed, retrain a
fresh instance of the existing conditioned event readout from scratch on the new
TRAIN records for 9,000 updates. Only adaptive execution trains a selector, for
2,000 updates after its readout updates. Do not reuse readout weights, or adaptive
selector weights, across input variants or arms. Freeze exact initialization
seeds, batches and RNG schedules in the source-bound plan before execution; this
prospective source/numeric freeze is adopted-from-§8.

Do not update or change the dynamics predictor, parser, history encoder,
architecture, loss or prediction horizon. No dynamics optimizer updates are
allowed. Scores are averaged over all three assigned seeds; there is no best-seed
selection.

The primary pure comparator uses frozen
`predictor-pure-760930001/2/3` checkpoints under the calibration-selected
fixed-250-continuous policy, averaging scores over all three seeds. Required
controls are:

- the same frozen hybrid checkpoints under the existing `fixed-50-macro` policy;
- a no-model `training_family_prior` derived only from the new TRAIN labels;
- candidate 0, the original action, as a disclosed secondary control; and
- the `legacy-single` hybrid arm required for RGB-path attribution.

Execute the complete arm-by-input matrix below. `Execute` means that the arm is
scored on that paired input view; no blank cell may be inferred or dropped.

| Arm | legacy-single | corrected-single | corrected-history |
| --- | --- | --- | --- |
| hybrid-adaptive | Execute | Execute | Execute |
| hybrid-fixed (`fixed-50-macro`) | Execute | Execute | Execute |
| pure (`fixed-250-continuous`) | Execute | Execute | Execute |
| no-model (`training_family_prior`) | Identical reference | Identical reference | Identical reference |
| candidate-0 (original action) | Identical reference | Identical reference | Identical reference |

For each hybrid-adaptive matrix cell and predictor seed, train the fresh
per-variant conditioned event readout for 9,000 updates and then the fresh
selector for 2,000 updates. The same-hybrid `fixed-50-macro` and pure
`fixed-250-continuous` controls train a fresh per-variant conditioned event
readout for 9,000 updates but train no selector; they select their pinned policy
directly. The no-model `training_family_prior` is computed once from TRAIN labels,
and candidate-0 is computed once from the fixed candidate list. Both are
input-invariant, consume no predecision input, have no conditioned
readout/selector path, and are duplicated as identical reference values across
all three input-variant columns. The dynamics predictor, parser and history
encoder are never retrained for any cell.

All arms use identical base lineages, action branches, stop labels, failure mask
and TRAIN/held-out split. Where applicable, model-backed arms use identical frozen
predictor seeds, initialization seeds, update counts, batches, RNG schedules and
readout capacity; adaptive cells use an identical selector schedule. The
paired-input rule applies before scoring: if one required view is missing, that
branch is unavailable to every arm.

The arm matrix's retained-result conversion and physical-frontier diagnostics are
source-bound respectively to `scripts/issue_76_bounded_transfer_metrics.py` and
`scripts/issue_76_bounded_transfer_frontier.py`, both in the plan's mandatory
source freeze.

## Execution and retention

Execution is not authorized by this document alone. Gate 3a must bind source,
numeric plan, player origin, membership, paired-render identity, action ordering,
failure rules and all metric implementations before capture dispatch. The passed
v4 shared-history smoke establishes only its bounded synchronization result; it
does not itself authorize this inventory or its fitting.

Execution envelope:

- eight isolated workers;
- 4,096 MiB per-worker process-tree RSS and 32,768 MiB aggregate RSS;
- 420 seconds per attempt;
- zero technical retries;
- 256 GiB minimum free disk before dispatch; and
- 150 GiB artifact cap, scaled from §8's 550 GiB for 9,100 branches.

The inventory requires 1,300 physical branch executions. Each physical branch
execution produces the three paired input views used by the matrix; these are
paired representations of one native branch, not 3,900 dispatches.

The combined active-collection cap is 345,600 seconds (96 hours),
adopted-from-§8 as an upper bound. Shot manifest deadline 180 seconds, history
readiness deadline 90 seconds, native timestep 0.0004 seconds, RGB stride 50
native steps, and at most 601 shot RGB frames plus the three assigned corrected
predecision history frames are adopted-from-§8. These limits do not promise
eightfold speedup.

Keep immutable player assets linked and keep each attempt's level XML, config,
logs, controls, ports, work directory and XDG state private. Charge shared inodes
once and disclose unique logical inode bytes rather than allocated filesystem
blocks; these storage conventions are adopted-from-§8. Retain every scheduled
branch, including valid outcomes, typed failures, partial artifacts and
unattempted members. Never replace, top up, delete or rerun a member because of
its outcome or operational failure.

## Validation and decision rule

The primary metric is `clear_hits`: one selected action per independent held-out
base lineage. An informative group is a prospectively scheduled, operationally
admissible lineage with at least one observed `native_clear` among its 13
branches. Freeze operational-admissibility and failure-mask computation before
access. Report both clear hits over all scheduled held-out groups and the clear
hit rate among informative groups. Retain and report every non-informative,
failed or incomplete group; none selects a replacement.

Freeze the result-to-metric adapter
`scripts/issue_76_bounded_transfer_metrics.py` and frontier module
`scripts/issue_76_bounded_transfer_frontier.py` among the plan's 41 mandatory
sources. The adapter's conversion of retained capture results into rows, labels,
validity and admissibility must preserve the four stop kinds exactly and apply
the paired-input rule that one missing view invalidates that branch for every
arm. The frontier module supplies the frozen physical-time, compute and
controller-trace definitions below.

At least 10 informative held-out groups are required for a decision-capable
result. Fewer than 10 is a coverage-insufficient stop, not evidence for or
against an arm.

Continue considering a larger corrected-data proposal only if every condition
below holds:

- at least 10 informative held-out groups;
- corrected-single hybrid has at least one more clear hit than legacy-single
  hybrid;
- corrected-history hybrid has at least one more clear hit than the independent
  pure comparator, the same-hybrid fixed control and the no-model prior;
- corrected-history is not worse than corrected-single;
- second-mode fraction is at least 0.05;
- second-horizon fraction is at least 0.05; and
- hybrid/pure linear-MAC ratio is at most 1.1.

Use the existing conditioned readout/selector definitions for second-mode,
second-horizon and linear-MAC accounting, and source-bind their exact numerators,
denominators and implementation before access. No post-access threshold or metric
definition changes are allowed. Passing these conditions justifies only a
separate user decision about a larger corrected-data proposal. It does not
authorize §8 or establish readiness, fresh generalization or advancement.

## Frontier metric protocol (R1 freeze)

Evaluate retained physical timelines on the common checkpoint grid `{0.5, 1.0,
2.0, 4.5, 8.0, 12.0}` seconds. A checkpoint is available exactly when the
retained observation window reaches that physical time; otherwise its
availability mask is false and no value is fabricated. All frontier consumers
must carry this mask.

The settled-state endpoint uses `v_settle=0.01 unity_unit/s` and
`omega_settle=0.01 degree/s`, derived from the player stability boundary at
`PhysicalSnapshotRuntime.cs:330-349`, and a continuous `w_settle=0.5 s` hold,
equal to 1,250 native steps at the frozen 0.0004 s timestep. `settled_at(t)` is
assigned only when every retained dynamic object's linear speed does not exceed
0.01 and its absolute angular speed does not exceed 0.01 throughout the complete
interval from checkpoint `t` through `t+0.5 s`, with continuous native-step
kinematic coverage. The only statuses are
`settled_at(t)`, `not_settled_censored` at 12 seconds and `unavailable` when the
required kinematics or continuous hold-window samples are absent. A non-settled
branch is never assigned an equilibrium label.

Regime segmentation is evaluation metadata only and is never a deployment
input. Segment the retained timeline over successive available checkpoint
windows using the exclusive precedence `collision_active` ->
`collapse_interaction_active` -> `settling` -> `quiescent`. `collision_active`
requires a new dynamic-object contact or a retained collision event involving a
dynamic object. Without that condition, `collapse_interaction_active` requires
at least two dynamic objects moving at or above 0.05 unity_unit/s in at least
50 percent of the window's retained samples. `settling` requires non-quiescent
motion after a prior retained event and a first-half to second-half mean-speed
reduction of at least 0.001 unity_unit/s. `quiescent` requires every retained
dynamic-object speed to be below 0.01 unity_unit/s. A window satisfying none of
these frozen exclusive definitions is invalid rather than relabelled.

Report cumulative error versus compute in two views. Area under the error curve
(`AUEC`) is trapezoidal integration over physical time using only available
common-checkpoint points, with its actual integrated-second span disclosed.
Error at equal cumulative MAC evaluates the requested cumulative-MAC points and
uses linear interpolation only within the observed MAC range; uncovered targets
are marked unavailable. Both views preserve the original typed label
`native_clear`, `native_fail`, `stable_without_clear` or `right_censored`.
Stable and censored observations are never retyped as failures.

For every decision, report the componentized compute ledger with these eight
categories separately: `perception`, `history_encoding`,
`predictor_transition`, `symbolic_heads_adapters`, `selector`, `readout`,
`infilling` and `action_scoring`. Each category reports MAC and FLOP-proxy counts
separately. Decision wall time is a separate field. Training compute is also
separate and reports MAC, FLOP-proxy, wall seconds and updates. Multiplicative
composite metrics, including decisions times MAC times wall time collapsed into
one number, are forbidden and remain unset.

The per-decision controller trace records `requested (delta_native_steps,
abstraction)`, `effective (delta_native_steps, abstraction)` and the boolean
`ranking_affect`. Delta is an integer and abstraction is a string. Oracle regret
is frozen as `not_computed` for this campaign because it has no legitimate
counterfactual oracle labels.

Physical-plausibility reporting uses only fields genuinely present in the
retained capture timeline. When contact separation is present, report minimum
contact separation and the count of negative-separation contact samples; report
the retained contact count and sampled lifecycle counts for `active`, `inactive`
and `destroyed` when those fields are present. Any absent field is
`not_available`. Floating remains `not_available` because the retained schema has
no authoritative floating predicate. Never infer physical plausibility from
latent MSE or other model-space error.

`scripts/issue_76_bounded_transfer_frontier.py` is one of the plan's 41 mandatory
`SOURCES`, alongside the result-to-metric adapter. Its contract and every numeric
definition above are frozen before campaign capture access and cannot change
after that access. Frontier results computed on previously exposed artifacts are
exploratory only.

## Honest failure and prohibited positive constructions

Stop the readout/selector strategy, or record the narrower stated finding, when
any of the following applies:

- fewer than 10 informative held-out groups;
- gains appear only in TRAIN loss, ranking or log-loss;
- corrected-history fails to beat the pure comparator or no-model prior;
- corrected-single does not improve on legacy-single, so RGB attribution fails;
- only history improves, which is an observability result rather than RGB
  causality; or
- a favorable claim requires replacements, dropped failures or post-access
  tuning.

The following positive constructions are prohibited:

- calling the 4.5 s endpoint a 12 s outcome;
- inventing later-clear labels;
- treating stable/censored as known failures;
- repeated-frame pseudo-history;
- post-action frames as input;
- counting 13 branches as 13 independent units;
- a v4-vs-old-600ms-records causal contrast;
- dropping zero-clear groups or failures; and
- treating smoke, TRAIN memorization or loss improvement as transfer success.

## Flags and disposition

Freeze these flags in the plan and every report:

- `capture_execution_authorized=false` until Gate 3a passes;
- `model_training_authorized=false`;
- `fresh_access=false`; and
- `advancement_authorized=false`.

The §8 shared-development draft stays parked regardless of this study's outcome
unless the user separately authorizes it. Neither a successful bounded result nor
the existence of retained artifacts changes that disposition.
