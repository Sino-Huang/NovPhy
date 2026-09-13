# Controller utility diagnosis: fix prediction quality before forcing switching

This exploratory CPU diagnosis uses existing training data only, with the
completed repaired-representation checkpoints from publication `850d777`.
Membership was chosen by metadata before these new scores: the first assigned
training lineage per family, IDs 001/071/141/211/281 under the
`issue-76-development-` prefix. All five were usable. All three seeds and both
independently fitted arms are retained; no fitting, captures, fresh access,
checkpoint selection, or deployment-policy changes occurred.

`scripts/diagnose_issue_76_controller_utility.py` archives all 30 arm/seed/lineage
records in `data/issue-76-controller-utility-diagnostic/report.json`, including
its exact source, membership, costs, local scores and full recursive fixed-pair
comparisons. The consolidated diagnostic took 13.223 CPU wall seconds with one
Torch thread, under its 180-second ceiling. Earlier read-only exploratory
commands led to this consolidated publication; this runtime is not total
research cost. The output is immutable, not an overwriteable latest result.

After activating `novphy` and sourcing `env.sh`, the executed commands were:

```bash
python -m unittest tests.test_issue_76_controller_utility
python -m scripts.diagnose_issue_76_controller_utility
```

Both focused tests passed. They check exact agreement with the real parent
teacher for both dynamics arms and refusal to invent a missing endpoint path.
The diagnostic also verifies exact parent-teacher label agreement for every
included training segment before interpreting its alternative scores.

## 1. Compute penalty suppresses small local symbolic gains

The original teacher minimizes observed-context, one-transition error plus
future observed-context value, not recursively predicted future error:

`h * .0004 * local_carrier_MSE + .01 * transition_linear_MACs / 2063232`.

Each seed contributes 1,350 teacher rows across the selected training segments.
The original hybrid teacher uses continuous mode on all 1,350 for every seed.
Holding predictions and DP paths fixed while setting only the diagnostic
compute coefficient to zero changes the label counts:

| Seed | Original symbolic labels | Zero-penalty micro labels | Zero-penalty macro labels | Zero-penalty continuous labels |
| --- | ---: | ---: | ---: | ---: |
| 760930001 | 0 | 1350 | 0 | 0 |
| 760930002 | 0 | 1273 | 1 | 76 |
| 760930003 | 0 | 1321 | 23 | 6 |

The micro penalty exceeds the continuous penalty by .002625473 per transition.
For seed 1 at horizon 750, mean local carrier MSE is .004340783 continuous
versus .004172473 micro. Its weighted mean local gain is only .000050493,
roughly 52 times smaller than the extra compute penalty. A positive local
symbolic gain therefore need not produce a cost-effective teacher choice.
This is evidence about the present utility scale, not permission to declare
symbolic computation free or lower the advancement compute constraint.

## 2. Removing compute cost does not repair recursive drift

With zero compute penalty, hybrid teachers mostly choose micro horizon 50:
905/855/893 rows for seeds 1/2/3. Yet full first-shot recursive predictions
at that fixed pair have very large error. Zero-penalty pure teachers likewise
prefer short horizons. This demonstrates why forcing symbolic labels or
simply removing the cost term is not a justified deployment repair.

A separate diagnostic replaces the error term with full recursively predicted
endpoint error, keeping the original per-transition compute penalty:

`4.5 * endpoint_carrier_MSE + .01 * total_transition_linear_MACs / 2063232`.

It scores every fixed pair at the same exact first-shot 11,250-step endpoint,
with no truth resets. This is a different error objective, explicitly not the
parent DP objective or a new advancement criterion. It selects fixed-750-micro
for all 15 hybrid seed/lineage cases and fixed-750-continuous for all 15 pure
cases. Thus recursive utility can justify micro's extra compute in these
examples, but its uniformly fixed winner does not establish useful adaptation.

The unchanged-carrier comparison is especially important. Even the hybrid's
fixed-750-micro prediction beats unchanged-carrier MSE in only 0/5, 2/5, and
3/5 cases by seed: 5/15 total. The independent pure fixed-750 prediction beats
unchanged carrier in all 15. This is a representation-space diagnostic, not
a no-model gameplay baseline or an independent scenario sample of size 15.
It identifies a dynamics-quality limitation that changing the controller alone
cannot resolve.

## 3. Exact-label accuracy overstates some distillation errors

On these same training inputs, hybrid exact-label accuracies are approximately
.516/.516/.534. However, mean regret under the unchanged local-DP teacher is
.000695/.001529/.000654; corresponding pure values are
.000689/.000680/.000572. Many best/second-best gaps are below .0001. Therefore
low exact-label accuracy alone is not proof that every disagreement is costly.
Some errors are material under this utility: hybrid maximum regrets are
.029581/.044257/.178404. Full per-lineage summaries retain these differences.

This refines, rather than erases, the previous finding of weak label fitting.
Do not optimize a classification metric while ignoring decision regret,
predicted-context distribution shift, or actual gameplay performance.

## Next development decision

Do not retrain a controller solely to increase symbolic counts. Prediction
quality, recursive drift, and teacher utility need separate attention. A
prospective dynamics-training intervention should keep the repaired common
representation, independent pure arm, all seeds, physical endpoints and
negative results fixed. Family-blocked predictor minibatches and four-step
training versus up to 225 recursive deployment steps are candidate causes,
not established causes of this dynamics gap. A matched sampler test can
isolate the first without changing architecture, utility weights or rollout
targets; it must retain equal scheduled/applied updates, valid example counts,
and failures. Longer recursive training would be a distinct intervention.

No fresh/final evidence was opened. This diagnosis supplies neither a supported
#76 disposition nor #64/#65 authorization. Actual paired gameplay, practical
margins, uncertainty, strongest controls/prior and matched full compute remain
required; fixed-pair carrier gains cannot substitute for them.
