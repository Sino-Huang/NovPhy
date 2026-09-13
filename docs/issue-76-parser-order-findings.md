# Parser ordering alone did not repair readiness

The source-bound `issue-76-parser-order-v1` experiment completed all three
1,500-update fits and all 900 assigned paired evaluation records. Each seed
retained exactly 1,488 applied batches and 12 unusable skips. The published
report at `data/issue-76-parser-order/report.json` validates against its frozen
plan, per-seed records, and completed resource budgets.

| Seed | Ordered model-selection block MAE | Interleaved MAE | Interleaved training-probe count span |
| --- | ---: | ---: | ---: |
| 760930001 | 1.48591 | 1.37895 | 0.48084 |
| 760930002 | 1.48575 | 1.33410 | 0.90285 |
| 760930003 | 1.49639 | 1.32850 | 0.87541 |

These are family-equal means over the 50 assigned model-selection episodes,
using the same prospectively chosen actual frames for both models. The required
MAE reduction was 50% and required count span was at least 3, for every seed.
**No seed passed.** The modest improvements do not establish that ordering has
no effect; they establish that this order-only intervention was insufficient at
the declared budget. Do not promote this parser as a ready deployment component.

New fitting plus paired scoring used 89.5462, 89.6172, and 89.5919 active seconds,
respectively (268.7552 seconds total). No original model or budget was changed;
there were no new captures, optimizer alternatives, update extensions, or fresh
access. The report retains the original common-model budgets separately.

## Additional diagnosis, not another candidate fit

The original first-shot screenshots for training lineages 001 and 141 visibly
contain the blocks and pigs. The objects are small even at 640×480; resolution
is therefore a possible limitation, not an established cause. No enlarged-image
training or architecture change was made.

A CPU backward-pass probe on the same five initial training images used in the
earlier collapse diagnosis found nonzero gradients in the encoder, backbone,
slot fusion, queries, and prediction heads. For seed 760930001, image-encoder
gradient norms were 33.9015 at original initialization, 0.03516 in the completed
ordered parser, and 2.84510 in the completed interleaved parser. Five-image losses
were 11.2077, 6.0963, and 7.2066. This 4.04-second read-only probe changed no model
weights and rules out a simply frozen/disconnected image path on these inputs.

The frozen sampled evaluation contains 7,719 pig-present and 217 pig-absent
training frames, plus 1,573 present and 27 absent model-selection frames.
These are descriptive sampled-frame counts, not independent gameplay episodes
or native win counts. The earlier concern that absent-class support might be
zero is not borne out by these measurements.

Next, freeze a tiny mixed-training-batch learnability check before another full
refit. Such a check can test whether the existing low-resolution parser can
fit known training images at all; it cannot establish generalization, gameplay,
useful symbolic switching, or an advancement pass. Preserve the original v2
negative evidence and this unsuccessful order-only experiment. #76 remains
without a supported advancement disposition; #64/#65 are not authorized.
