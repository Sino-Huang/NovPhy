# #76 findings: completed diagnostic, pre-access readiness stop

The full diagnostic completed and its original saved evidence validates exactly.
The disposition remains `readiness_or_precision_insufficient` **before fresh
access**, not `not_supported_by_this_experiment` for an unexecuted fresh study.
#75 remains a completed negative pilot. #64/#65 remain unauthorized and unrun.

The [working checklist](https://github.com/Sino-Huang/NovPhy/issues/76#issuecomment-5615651006)
distinguishes executed work, reporting closeout, and conditional future work.
The [local checklist](issue-76-todo.md) preserves the same accounting.

## Evidence and analysis provenance

- Original diagnostic: `.local-artifacts/issue-76-dynamics-diagnostic-v1/`.
- Original gallery/CSV/summary: `data/issue-76-review/`.
- Reporting supplement: `data/issue-76-closeout/{summary.json,comparisons.csv,findings.md}`.
- Original CSV validation correction: `validation-csv-repair.json` in the diagnostic directory.
- Supplement runner: `scripts/run_issue_76_closeout.py`.

The supplement completes missing comparator reporting after diagnostic
inspection. It is not a retroactively preregistered experiment, another model
repair, or a substitute for the original publication. It preserves original
plans, predictions, outcomes and review artifacts. Its JSON binds its exact
source, the original validation correction and predecessor selection. Each
publish/validate command first runs the original full saved-evidence validator.

All 34 additional comparisons are reported: all 16 fixed/adaptive systems versus
the frozen ordinal09 prior and uniform expected regret, plus original/covered
adaptive hybrid versus the independent continuous policy selected in #75.
The 33 original training/symbolic/adaptation contrasts are retained with
metric-specific descriptive decisions. No favorable comparison replaces another.

The unit is one of 24 predeclared calibration states (12 per current family).
Paired seed differences are averaged within state before a 10,000-draw state
bootstrap with RNG seed7201. JSON also retains each seed and both family
summaries. These are descriptive 95% intervals on already-opened development
data, without multiplicity-adjusted confirmation. Actions and seed repeats are
not independent lineages. An interval containing zero is not an equivalence test.

## 1. Hybrid training does not show a uniform advantage

These comparisons match horizon and execute continuous transitions in both
independently trained models. Positive improvement is reference minus hybrid:
positive means lower error/regret for the hybrid-trained checkpoint.

| Horizon | Regret improvement [descriptive 95% interval] | Offset225 carrier-MSE improvement [descriptive 95% interval] |
| --- | --- | --- |
| h1 | -0.027778 [-0.069444, 0.000000] | -216.416695 [-250.794489, -182.805729] |
| h5 | -0.055556 [-0.111111, -0.013889] | 0.005988 [0.001929, 0.010008] |
| h15 | 0.000000 [0.000000, 0.000000] | -0.001710 [-0.001923, -0.001499] |

At h5, lower recursive carrier error coexists with worse action ranking. At h1
and h15, offset225 carrier error is worse. Thus neither carrier improvement nor
a single fixed-horizon comparison establishes a general task advantage.
The complete five-endpoint and separate local-step results remain available;
the endpoint table above is not a claim about an entire error-growth curve.

## 2. Symbolic execution effects depend on mode, horizon and endpoint

The six same-checkpoint symbolic-versus-continuous contrasts are retained at all
five endpoints. Their directions are mixed. For example, macro h1 reduces
offset225 carrier MSE relative to the hybrid continuous h1 path, while its mean
regret difference is zero; micro h5 increases carrier error without changing
mean regret. Other effects differ by endpoint.

These findings separate actual execution mode from controller choice, but do
not establish universal symbolic benefit, reliable novelty semantics or a
uniquely joint/nonseparable mechanism. Both dynamics arms share a semantically
supervised CNN/object-centric carrier; this is not an end-to-end symbol-free test.

## 3. Adaptation does not establish the required strongest-baseline advantage

The continuous reference remains `continuous_h5`, selected using #75's full
200-state development evidence. It was not reselected on the 24-state diagnostic.
The no-model prior remains ordinal09, also frozen before #76; its development
selection optimism is explicitly retained.

| Tested | Reference | Regret improvement [descriptive 95% interval] |
| --- | --- | --- |
| Independent continuous h5 | Ordinal09 prior | -0.041667 [-0.152778, 0.069444] |
| Original hybrid adaptive | Ordinal09 prior | -0.069444 [-0.208333, 0.055556] |
| Covered hybrid adaptive | Ordinal09 prior | -0.069444 [-0.180556, 0.013889] |
| Original hybrid adaptive | Independent continuous h5 | -0.027778 [-0.083333, 0.027778] |
| Covered hybrid adaptive | Independent continuous h5 | -0.027778 [-0.083333, 0.027778] |

None of these intervals establishes improvement. Original and covered hybrid
adaptive mean regrets are both 0.361111 on this subset, compared with 0.333333
for continuous h5 and 0.291667 for the prior. Uniform expected regret is 0.354167;
it is an expectation across the 12 replayed actions, not a new random-policy
trial. All systems' prior/uniform comparisons and per-seed results are included,
not only the rows above.

Adaptive carrier error can improve over weak h1 fixed paths while remaining
worse than stronger h5/h15 controls. All 24 original/covered adaptation contrasts
are retained. Intermediate adaptive curves are unavailable, not interpolated.
Neither arbitrary mode variation nor a favorable comparison against a weak
control rescues the #75 gate.

## 4. Useful gameplay and broad NovPhy generalization remain unestablished

Only 10/24 states have informative action-regret variation; 2/24 have a
pig-removing/level-clearing candidate in the existing grid. Original and covered
hybrid adaptive select no pig-removing or level-clearing candidate in any of the
three seeds on this subset. The prior also selects none. The full report retains
every system's absolute counts, contacts/support changes, ties and failures.

These are selected existing single-shot replays, not new closed-loop gameplay
or multi-shot success measurements. Top1 ranking is not winning. Offset225 need
not be settlement, early endpoints can precede launch, and stable absorption
is explicitly labeled. Learned-carrier MSE is not an oracle physical violation.
Local observed-context accuracy cannot be substituted for recursive deployment.

The 80-template/40-pair/45-cell metadata inventory is complete. Actual model
evidence is still limited to normal single-force/multiple-forces states. The
rolling/falling/sliding and appearance-pair proposal is costed but not an
approved executable protocol. Novel forces, events, appearances and right-side
actions need the documented representation/observability/collector checks.
There is no result for zero-shot or few-shot novelty generalization, and no
evidence that normal-level selection caused the earlier hybrid failure.

## 5. Resource limits and original evidence are preserved

The 10,368-record diagnostic used 983.47 recorded active seconds and
769,148,024 derived bytes, with peak CPU RSS1542.59MiB and CUDA allocation36.37MiB.
It did not stop at a budget limit. The full supplement carries actual transition,
local, parser, decoder/adapter/controller and historical #74/#75 training-cost
references, including the failed correction. Linear MACs are not full FLOPs;
equal action counts are not matched end-to-end compute. Cached adaptive wall
times are not newly paired latency measurements. Validation/reporting work is
separate from the frozen diagnostic allowance.

## Reproduction and stop-branch handoff

Initialize `novphy` and source `env.sh`, then:

```bash
python -u -m scripts.run_issue_76_closeout --dry-run
python -u -m scripts.run_issue_76_closeout --publish --device cuda \
  2>&1 | tee -a data/issue-76-closeout-publish.log
python -u -m scripts.run_issue_76_closeout --validate --device cuda \
  2>&1 | tee -a data/issue-76-closeout-validate.log
python -m unittest tests.test_issue_76_closeout tests.test_issue_76_dynamics_diagnostic -q
```

Publication/validation log original per-seed/state progress. They recompute
reporting from existing evidence, not diagnostic predictions or game capture.
The original `--validate` command remains valid independently.

The issue's allowed stop branch does not require inventing a fresh sample-size
protocol, running unauthorized expanded experiments, or treating missing
archival as a pushed release. Exact fresh membership/power/margins/compute,
global disjointness, rendered compatibility and shared representation/training
changes remain conditional and unexecuted. No advancement gate was lowered.

Handoff: #35 consolidates this diagnostic and stop with distinct provenance and
claim labels; #36 handles archival with the required authority; #37 reports the
results and limits; #38 accounts for conditional scope. #64/#65 stay blocked/not
run. This does not complete those downstream tickets, authorize another repair
search, or overwrite #15/#72/#74/#75 conclusions.
