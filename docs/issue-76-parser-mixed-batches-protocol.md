# Prospective exposure-matched mixed-minibatch test

The order-only probe was insufficient, while the five-image mixed-batch check
showed that the unchanged parser can fit those known inputs. This experiment
tests within-minibatch composition at full training exposure. Neither earlier
result is replaced, and this is not a fresh evaluation or an optimizer sweep.

## Single change

Keep all 250 original training assignments and the exact original sampled
image/label multiset for each seed and each of the six passes. Generate each
episode's original 32 samples using its original update number, exactly as in
the preceding order probe. Arrange usable episodes in that probe's frozen
family-interleaved order. Stack their sampled tensors as [248 episodes, 32 rows],
transpose these axes, then regroup into 248 batches of 32. Apply the same
transposition to images, presence labels, and centers; duplicated sampled frames
remain duplicated. There are no new labels, artificial images, or replacements.

Retain the original two unusable assignments as two skipped update slots at the
end of every 250-slot pass. Thus each seed still has 1,500 scheduled updates,
1,488 optimizer updates, 12 skips, and exactly 47,616 real image/label exposures.
Batch contents and update order intentionally differ; do not claim that each
old batch is preserved. Test the multiset and input/target alignment before fit.

Use the original three seeds, initialization, NativeVisualParser, 64×96 RGB,
perception_loss, AdamW .001, weight decay .0001, and gradient cap 1. No extra
updates, architecture/resolution choices, augmentation, checkpoint selection,
new captures, or history/dynamics/controller fitting. Five-image overfit weights
are not reused. The whole-episode interleaved fits are the primary paired
controls; retain the original ordered controls as well.

## Measurement and decision

Reuse the preceding probe's exact 250 training and 50 model-selection assignments
and exact up-to-32 actual evaluation frame indices per episode. No calibration
data or model-selection labels enter an optimizer. Reuse unchanged control
measurements, bound in the new plan, rather than recomputing them. Keep all failed
assignments. Record all per-episode metrics, family/role means, and pig-positive,
pig-negative, false-positive, and false-negative sampled-frame counts.

Every seed must meet all three mechanistic development criteria:

- family-equal model-selection block MAE <=50% of its interleaved control;
- first-frame predicted count span >=3 across the same five training probes;
- family-equal task-slot presence Brier error no worse than that control.

Report all seeds and cells. These criteria neither weaken nor replace #76's
gameplay, strong independently trained continuous, same-checkpoint control,
useful symbolic switching, uncertainty, validity, matched-compute, and fresh
advancement requirements. A favorable parser result still needs a separately
frozen symmetric full refit and actual gameplay readiness; these parser-only
checkpoints cannot be deployed as complete models. No result opens fresh data.

## Budget and provenance

Root `.local-artifacts/issue-76-parser-mixed-batches-v1`; preserve every parent
source, checkpoint, cache, record, and budget. Bind parent kernel source, the
preceding probe's source and controls, all assignments, frame indices, order,
this protocol, and the new executable before fitting. No new hashes or
full-corpus integrity passes. The original shards are read for sampled training
data, not regenerated. Cache only one pass's sampled tensors in CPU memory.

Allowance: 900 active seconds per seed including buffer loading, fitting, and
new scoring (2,700 total), 12 GiB CPU RSS, 8 GiB allocated GPU, and 1 GiB new
artifacts. Previous fitting/scoring costs remain in the bound control summaries
and reference publication. Use the existing persistent FitBudget and retain
stops/unclean interruptions; no concurrent GPU research job or automatic retry.

After activating `novphy` and sourcing `env.sh`, run the unit test
`python -m unittest -q tests.test_issue_76_mixed_batches`, then use
`python -m scripts.run_issue_76_mixed_batches` with `--dry-run`, `--prepare`,
`--run`, `--publish`, and `--validate`. Record the prospective freeze before run.
