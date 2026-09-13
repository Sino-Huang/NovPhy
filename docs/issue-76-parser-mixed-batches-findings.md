# Mixed batches restore responsiveness but do not meet the frozen criteria

The `issue-76-parser-mixed-batches-v1` experiment completed all three fits and
all 900 assigned paired evaluation records. Publication and exact record/plan
validation pass at `data/issue-76-parser-mixed-batches/report.json`. Each seed
retained 1,500 scheduled updates, 1,488 applied updates, 12 skipped assignments,
and the original 47,616 real image/label exposures. No original evidence changed.

| Seed | Interleaved model-selection block MAE | Mixed MAE | Mixed training-probe count span |
| --- | ---: | ---: | ---: |
| 760930001 | 1.37895 | 0.90125 | 5.71333 |
| 760930002 | 1.33410 | 0.74310 | 6.44130 |
| 760930003 | 1.32850 | 0.85592 | 5.97669 |

All seeds exceed the required count span of 3 and improve task-slot presence
Brier error, but none halves its interleaved-control model-selection block MAE.
**The predeclared combined criterion fails for all three seeds.** Responsiveness
has improved materially; that does not justify relabeling this as a ready parser
or an advancement pass. These are development comparisons, not fresh evidence.

Mixed model-selection task-presence Brier errors are .09811, .08376, and .09472,
versus interleaved controls .14162, .13795, and .13575. Present task-slot center
MAEs remain .07811, .07889, and .08206. Training block MAEs are .54716, .28138,
and .33455: the remaining gap to model selection matters, so more optimization
cannot simply be assumed to fix generalization.

New fitting, buffer loading, and scoring consumed 103.0783, 102.7748, and
102.4343 active seconds (308.2874 total). Parent costs remain in the bound
reference publication and control summaries. No additional optimizer step was
run after inspection, and none of these parser-only checkpoints is deployable.

## A separate rare-class failure remains

All three models classify all 27 sampled pig-absent model-selection frames as
pig-present, while classifying all 1,573 sampled pig-present frames correctly.
The 27 absent frames belong to **one** scenario lineage,
`issue-76-development-069`; they are not 27 independent evaluation cases.

On the sampled training frames, false-positive counts are 209, 196, and 192
out of 217 pig-absent frames; false-negative counts are 1, 4, and 31 out of
7,719 pig-present frames. Negative training frames occur in nine lineages:
002, 022, 024, 028, 032, 036, 046, 048, and 300. Low aggregate pig error would
conceal this failure to recognize removal.

Read-only inspection of lineage 069's original agent screenshots at first-shot
frames 0 and 96 shows a visible pig initially and its absence at frame 96.
The prepared negative label corresponds to native fixed step 20,510; the
retained source trace records `pig_removed` and `entity_death`. This supports
the label for that diagnostic example; it is not an outcome-conditioned new
capture or a general label-integrity proof.

That shot's original result remains a censored `native_time_window_limit`
without an observed native terminal. Pig removal or the visible score does not
authorize rewriting its frozen gameplay outcome. The separate native-terminal
audit and all old negative evidence remain unchanged.

The next targeted diagnosis should distinguish rare-class learning from the
remaining generalization/optimization problem, with any new recipe and budget
frozen before fitting. Do not weaken the failed thresholds, force symbolic use,
select a favorable seed, or open fresh data. #76 still has no supported
advancement disposition, and #64/#65 remain unauthorized.
