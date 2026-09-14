# Task-objective gap and next development direction

The angular pilot remains a failed feasibility screen. The following uses its
already-opened training outcomes only; no new model inference or capture was
performed. It identifies a concrete objective mismatch, not a hybrid advantage.

## Evidence

The deployed [TaskObjective](../world_model/planning/task_objective.py) reads
only bounded pig/block presence and returns `1000 * pigs + blocks`.
[Replay scoring](../world_model/planning/replay_action_scoring.py) applies it to
a recursive carrier at 11,250 native steps (4.5 physical seconds) from the
decision observation. It has no direct terminal-failure or censoring prediction.
The [native fitting code](../world_model/training/native_history_fit.py) trains
continuous dynamics on local/recursive carrier MSE and range excess, adds
symbolic losses for hybrid modes, and trains controller targets from carrier
error plus compute. Those objectives do not directly optimize gameplay success.
This does not mean latent states cannot indirectly encode relevant events.

The replay evaluation instead assigns 1e9 to failed/censored outcomes. To expose
the resulting distinction, a retrospective oracle diagnostic ranks every
assigned grid action by its **recorded** endpoint count cost, using the existing
lowest-ordinal tie rule. Reference actions are excluded and all five lineages
are retained, with 071 still excluded from paired conclusions.

| Lineage | Comparable | Count-only selection | Native outcome | Penalized regret | Count tie |
| --- | --- | --- | --- | ---: | --- |
| 001 | yes | a05 | censored | 999,998,999 | a05, a12; both censored |
| 071 | no | a09 | stable, not clear | 0 | none |
| 141 | yes | a06 | level clear | 0 | none |
| 211 | yes | a07 | censored | 999,998,994 | a07 censored, a12 stable |
| 281 | yes | a03 | level clear | 0 | none |

Thus exact recorded endpoint counts still select failures on two of four
comparable lineages. For 001 the lower count strictly favors censored outcomes;
for 211 the result depends on the disclosed tie rule. The
[machine-readable diagnostic](../data/issue-76-angular-replay/count-objective-gap.json)
binds these selections to the complete prior audit. Reproduction is sorting
each group's non-reference rows by `(observed_count_cost, candidate_ordinal)`
and evaluating the selected row's unchanged `ranking_cost`.

Actual endpoint times differ across actions. This is **not** an equal-time
dynamics comparison, an accuracy ceiling for the 4.5-second predictor, a
deployable oracle policy, or proof that this mismatch is the sole failure cause.
No confidence or seed-robustness claim follows from these four paired lineages.

## Next development direction, not an execution protocol

The next justified workstream is an event-aware, task-aligned prediction and
control contract shared by independently trained hybrid and pure-continuous
arms. It is not another learning-rate sweep or a post-hoc change to old results.

Before fitting or collecting anything for that workstream:

- Define a common physical deadline and distinguish clear-by-deadline,
  level-fail, stable-without-clear, and right-censored observations. A censored
  trace is negative for **observed clear by its deadline**, not evidence of
  never clearing or an invented terminal-time target.
- Audit training support for those targets by independent lineage and source
  role. The two angular positive lineages alone are not a training/readiness
  demonstration; neither outcomes nor geometry may choose favorable membership.
- Specify symmetric event/outcome supervision and capacity/compute accounting.
  Pure-continuous dynamics must remain independently trained without privileged
  symbolic inputs. The same-hybrid fixed-mode arm remains a separate control.
- Train action selection and adaptive mode/horizon choice against the declared
  task estimand, with compute charged, rather than silently substituting carrier
  MSE. Maintain separate dynamics, switching, and gameplay claim decisions.
- Freeze development checks and a stop rule before execution. Require actual
  gameplay and the strongest fixed/no-model controls before any fresh-access
  readiness/precision assessment. Do not authorize fresh data from a classifier
  score, training fit, or another small feasibility screen.

This document does not freeze those numeric choices or authorize a run. The
next implementation must first establish target availability and an exact,
costed training/development protocol. The #76 advancement gates, original-grid
negative results, and angular pilot's failed criterion remain unchanged.
