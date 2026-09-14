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

## Consequence for #76

These five training lineages do not establish adaptive gameplay advantage,
independent uncertainty, matched compute or the advancement gate. No model
ranking has yet been measured. Before predictor scores, freeze the scoring
implementation, checkpoint pool and common genuine pre-decision anchor.
Any ranking result must retain all failures/ties, separate absolute count regret
from failure penalties, and keep lineage 071's failed comparability visible.
The observed one-block headroom and zero verified clears must not be presented
as evidence that #76 already passes or that a learned gameplay improvement is
assured.
