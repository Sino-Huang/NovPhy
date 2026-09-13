# Pig-loss balancing repairs the sampled removal-recognition failure

All three seeds pass the prospectively frozen mechanistic criteria in
`docs/issue-76-balanced-pig-protocol.md`. Publication and exact plan/result
validation pass at `data/issue-76-balanced-pig/report.json`. This is development
evidence, not an advancement disposition or a deployable complete checkpoint.

The training-exposure audit found only 1,321, 1,360, and 1,341 pig-absent draws
out of 47,616 per seed, across nine training lineages. The matched intervention
balanced only pig BCE and pig-count squared error using training-derived weights;
images, batches, initialization, optimizer and update counts were unchanged.
Every seed completed 1,500 scheduled / 1,488 applied / 12 skipped updates and
all 300 assigned evaluation records. No seed, example or checkpoint was selected.

| Seed | Training FP / absent | Training FN / present | Model-selection FP / absent | Model-selection FN / present |
| --- | ---: | ---: | ---: | ---: |
| 760930001 | 0 / 217 | 102 / 7,719 | 0 / 27 | 13 / 1,573 |
| 760930002 | 0 / 217 | 71 / 7,719 | 0 / 27 | 5 / 1,573 |
| 760930003 | 0 / 217 | 77 / 7,719 | 0 / 27 | 5 / 1,573 |

The mixed controls had 209/196/192 training false positives and 27/27
model-selection false positives for every seed. The new models trade some
false negatives for removal recognition. Training sensitivities are
.98679/.99080/.99002; model-selection sensitivities are .99174/.99682/.99682.
All specificity values are 1.0 on these samples. Thus improvement is not an
always-absent classifier artifact. Class weighting can change calibration;
these numbers do not establish calibrated uncertainty.

| Seed | Mixed model-selection block MAE | Balanced MAE | Mixed task Brier | Balanced task Brier |
| --- | ---: | ---: | ---: | ---: |
| 760930001 | .90125 | .95381 | .09811 | .10097 |
| 760930002 | .74310 | .74055 | .08376 | .08142 |
| 760930003 | .85592 | .80156 | .09472 | .08607 |

All collateral-error ratios meet the frozen <=1.1 control bound, including
seed 1's small deterioration. Balanced training block MAEs are .39272, .29334,
and .28515. The remaining training/model-selection gap and substantial absolute
count errors are not repaired by this result. Neither the older mixed-batch
criterion nor any #76 requirement is replaced by the new mechanistic criterion.

All 27 pig-absent model-selection frames still come from ONE scenario lineage.
Repeated frames and model seeds are not independent evidence of broad rare-event
generalization. The result supports class imbalance as a contributor to the
sampled learning failure, not as the sole cause of all model failures.

New active fitting/loading/scoring costs are 103.7702, 103.1800, and 103.0312
seconds (309.9815 total), plus 8.7434 CPU seconds for the separate exposure audit.
Bound parent costs remain in the publication. No update extension, automatic
retry, new capture, fresh access, or original-evidence rewrite occurred.
Four focused unit tests (including original-loss/gradient equivalence and exact
sample accounting) and eight parent native-fit/order tests passed.

Next, the improved training recipe needs a prospective common-representation
and paired dynamics/controller readiness study with actual learned gameplay,
not immediate fresh evaluation. Remaining requirements include a strong
independently trained continuous baseline, useful adaptive switching, recursive
prediction quality, uncertainty, and full matched compute. No supported #76
advancement disposition exists; #64/#65 remain unauthorized.
