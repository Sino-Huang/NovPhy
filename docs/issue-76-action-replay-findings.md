# Action-replay collection findings

All 65 assigned training-only replays finished without a technical retry.
Collection exited successfully after 6,882.05 active wall seconds, with peak
worker-tree RSS 1,828.11 MiB and 12,259,919,755 artifact bytes.

The full validator passed all 65 capture contracts in 855.70 seconds. Its
machine-readable evidence is in
`data/issue-76-action-replay/collection-validation.json`. Four of five source
lineages passed the prospectively frozen operational comparability checks.
Lineage 071 failed on bird velocity/angular velocity in nine candidate captures;
the whole lineage is inadmissible for paired ranking. All its data and failed
comparisons remain retained. No tolerance was changed and birds were not omitted.

## Native endpoint inspection

A read-only inspection of each validated trace's last native chunk found no
native level-clear terminal among the 65 captures. Active pig/block counts below
use the final sample's `lifecycle == active` entities. These are evaluation
labels, not model inputs. Censored endpoints are not settled outcomes, and
native level failures retain the predeclared 1e9 ranking cost.

| Source lineage | Paired admissible | Grid actions with stable endpoints | Distinct stable pig/block counts |
| --- | --- | --- | --- |
| 001 | Yes | 5, 6, 7, 8, 11 | 1 pig, 1 block |
| 071 | No | 6, 9 | 1 pig, 5 blocks |
| 141 | Yes | 4, 5, 6, 9, 10, 12 | 1 pig, 6 blocks; action 12: 1 pig, 5 blocks |
| 211 | Yes | 5, 10, 11 | Action 5: 1 pig, 6 blocks; others: 1 pig, 7 blocks |
| 281 | Yes | 2, 5, 8, 11 | 1 pig, 4 blocks |

The action ordinals are the frozen 12-action grid, not the original-action
reference (ordinal 0). All other grid actions are censored or native level
failures. In particular, 001 action 9 has zero active pigs at the last observed
sample but is censored: it is not a verified clear or a settled successful
outcome and still receives the failure cost.

Only two admissible lineages distinguish settled grid outcomes by task counts,
each by one block. The other two distinguish stable observations from
failure/censor penalties only. A large penalty-regret improvement would not
establish a large pig/block improvement. The original-action reference is
outside the ranking pool; its outcome cannot be selected as a thirteenth action.

The endpoint inspection took 0.5872 CPU wall seconds, in addition to full
validation. Together these consume 856.2853 seconds of the 1,800-second offline
preparation allowance, before subsequent preparation work. This inspection did
not run predictors, fit models, change checkpoints or access fresh evaluation.

## Fixed-model ranking inspection

The frozen scoring run completed 54 model instances, 1,620 policy-lineage rows
and 19,440 candidate predictions in 177.40 active wall seconds, without nonfinite
prediction failures. These are fixed policies, not an adaptive controller.

Using the existing `TaskObjective.ranking_diagnostic`, each recipe/pair is
aggregated equally over three seeds and four comparable lineages. These twelve
decisions are not twelve independent scenarios. Lineage 071 remains saved but
excluded from paired summaries.

Among 81 hybrid recipe/pair combinations, the lowest observed mean normalized
penalty regret is approximately 0.666667 (`boundary-dynamics`, fixed-250-macro).
It selects failed/censored actions in 8/12 decisions; top1 is 2/12 and top3 is
8/12. Among 27 pure combinations, the best is `matched-dynamics`,
fixed-750-continuous: mean penalty regret 0.75, failed/censored selections 9/12,
top1 3/12, top3 9/12. These descriptive minima are selected on this diagnostic,
with selection optimism, not independently confirmed winning policies.

Inspection of all twelve fixed-action priors finds action 5 avoids failure-cost
selections on all four comparable lineages: top1 3/4 and average absolute settled
count regret 0.25 (one extra block on lineage 141). Its normalized penalty regret
is approximately 2.5e-10 because the denominator includes 1e9 failure costs; this
does not mean near-perfect gameplay. None of its outcomes is a verified clear.
This prior is also development-selected, not fresh evidence. Nonetheless, the
existing learned fixed rankings do not beat this simple prior on the diagnostic.

This read-only ranking inspection took 0.0681 CPU wall seconds and used saved
scores, not new inference. The complete machine-readable publication is
`data/issue-76-action-replay/rankings.json`: all 1,620 policy-lineage rows,
108 recipe/pair summaries and all twelve priors. Raw scores are retained under
`.local-artifacts/issue-76-action-replay-v1/fixed-action-scores/`.
Their complete 54-file archive is
`data/issue-76-action-replay/fixed-action-scores.tar.gz`; the measured scoring
budget is published in `data/issue-76-action-replay/scoring-completion.json`.

Across all admissible fixed-policy decisions (not independent replicates),
844/972 hybrid and 265/324 pure selections incur failure/censor penalties.
Selected predicted pig counts average 0.8662 and 0.8294 respectively; only
51/972 and 28/324 fall below 0.5. Thus wholesale predicted pig removal is not
established as the general explanation. Among selected stable outcomes, every
observed pig count is one; mean selected predicted-pig absolute error is 0.0807
for hybrid and 0.1200 for pure. These endpoint comparisons are descriptive:
predictions are at decision plus 4.5 seconds, while outcomes are at settlement.

The scorer predicts bounded presence/count costs, not the probability of a
native terminal, level failure or censoring by twelve physical seconds. The
realized ranking objective additionally assigns 1e9 to failed/censored outcomes.
This mismatch and the previously documented action-support gap must be addressed
in a prospective development design. The results alone do not establish which
is causal, justify changing the existing failure rule, or justify another
unchanged optimizer sweep. No new training or capture is authorized by this
findings document.

## Consequence for #76

These five training lineages do not establish adaptive gameplay advantage,
independent uncertainty, matched compute or the advancement gate. Fixed-model
rankings were measured only after the scoring implementation, checkpoint pool
and common genuine pre-decision anchors were frozen in local commit 7fc4b86.
Any ranking result must retain all failures/ties, separate absolute count regret
from failure penalties, and keep lineage 071's failed comparability visible.
The observed one-block headroom and zero verified clears must not be presented
as evidence that #76 already passes or that a learned gameplay improvement is
assured.
