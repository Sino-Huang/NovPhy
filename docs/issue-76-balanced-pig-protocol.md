# Prospective training-derived pig-loss balancing diagnostic

The exact training-exposure audit counts 47,616 samples per seed, including
duplicated draws. Pig-positive/negative counts are 46,295/1,321, 46,256/1,360,
and 46,275/1,341 for seeds 760930001/2/3. All negative exposures belong to the
same nine training lineages. These counts come only from original training
labels, not calibration or model selection. Audit source, members, counts,
and measured CPU cost are retained in `data/issue-76-pig-exposure/report.json`.

## Single intervention

For each seed use class weight N/(2*N_class), frozen before optimization:
positive/negative weights .5142671994815855/18.022710068130205,
.5147007955724663/17.50588235294118, and
.5144894651539709/17.75391498881432. Apply these weights only to the per-example
pig-slot BCE contribution (4/22 times BCE) and pig-count squared error. The
mean weighted pig contribution replaces its original mean. All other objective
terms remain unchanged; one CNN forward supplies both the original objective
and the correction. Unit weights must reproduce original loss and gradients.

Reuse the exact mixed-batch control's original images, targets, transpose,
six passes, seed initialization, 64x96 architecture, 1,500 scheduled updates,
1,488 applied updates, 12 skipped slots, AdamW .001, weight decay .0001, and
gradient cap 1. No new examples, resampling, augmentation, optimization extension,
architecture search, or checkpoint selection. All three seeds are required.
Do not modify frozen parent source or replace any earlier negative result.

## Measurement and interpretation

Retain all 250 training and 50 model-selection assignments per seed, including
unusable cases, and the original exact evaluation frame indices. Reuse bound
mixed control measurements. Report all per-episode metrics and family/role
summaries, plus pig specificity (1-FP/negative) and sensitivity (1-FN/positive).
Every seed must have specificity >=.5 and sensitivity >=.95 on BOTH training
and model selection, with model-selection family-equal block MAE and task-slot
Brier each <=1.1 times its mixed control. A constant-absent classifier fails
sensitivity. This diagnoses learning tradeoffs, not calibrated probabilities:
class weighting may change probability calibration and that must be disclosed.

These are new mechanistic criteria, not a replacement for the previously failed
mixed-batch criteria or #76 advancement requirements. Model-selection negatives
are just 27 frames from ONE lineage; even passing does not establish robust
rare-event generalization. No claim of independence for repeated frames/seeds.
No calibration labels enter fitting. No gameplay/history/dynamics/controller
fit, deployment, fresh access, or #64/#65 authorization follows from this test.

## Execution and budget

Bind this source/protocol, full parent plan/source, all three mixed controls,
exposure audit and coefficients before fitting in
`.local-artifacts/issue-76-balanced-pig-v1/plan.json`. New allowance: 900 active
seconds per seed including buffer loads, fit and scoring (2,700 total), 12 GiB
CPU RSS, 8 GiB allocated GPU, and 1 GiB new artifacts. Charge the completed
training-label audit separately and retain all prior costs. Reuse the persistent
budget/checkpoint machinery; no concurrent GPU research job or automatic retry.
No hashes, raw-corpus reread, new captures, or source-shard regeneration.

Activate novphy, source env.sh, run `python -m unittest -q
tests.test_issue_76_balanced_pig tests.test_issue_76_pig_exposure`, then
`python -m scripts.run_issue_76_balanced_pig` with `--dry-run`, `--prepare`,
`--run`, `--publish`, and `--validate`. Record the prospective freeze before run.
