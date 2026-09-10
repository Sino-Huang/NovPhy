# #70: image-blind parser diagnosis and v4 repair

The completed v3 objective audit remains a valid negative result: all 200 states
had tied parsed scores; on 81 count-discriminating states, regret was 1.0 and
top-3 was 4/81. Gameplay remains blocked. This was not a CEM execution failure.

## Reproduction and mechanism

Direct evaluation of the saved parser on 120 different real training images
showed that the first image projection was entirely negative. Its following
ReLU produced all zeros, and gradients into that projection were zero. Later
layers could still reduce aggregate loss by fitting per-slot frequencies, but
could no longer distinguish images. All 20 epochs completed without detecting
this failure. The earlier two-update smoke test was insufficient.

A fresh, seed-matched, training-only reproduction collapsed by epoch 2. The new
image-sensitivity detector reported exactly zero presence/pig/block ranges.
Normalization plus leaky activations alone avoided exact zeros but retained very
weak sensitivity; it was not considered sufficient for the repair.

## Implemented plan

- Preserve original v1/v2 parser architectures and every existing result.
- Use a separately identified normalized-slot architecture: standardize RGB/Sobel
  features per pixel, then LayerNorm and LeakyReLU(0.1) after each dense image
  projection. Retain nonlinear slot fusion and the existing supervised losses.
- Fit input means and population standard deviations once from first/last
  sampled frames of 16 uniformly spaced **training** lineages, with scale floor
  0.05. Store these as checkpoint buffers; inference never refits statistics.
- Log maximum presence variation and pig/block count ranges on that fixed probe
  each epoch. Stop before checkpointing an image-blind/nonfinite epoch (maximum
  presence range <=1e-5). This detects collapse, not predictive accuracy.
- Retain the full 18-slot vocabulary and 236-value downstream representation.
  Use new parser/carrier identities and v4 artifact directories. Retrain all
  downstream world models only if the unchanged objective audit passes.
- Reuse complete v3 image/label shards read-only, checking training membership,
  release bindings, vocabulary, sampling and image dimensions. No hashes, data
  copies, new gameplay collection, or reuse of failed model weights is involved.
  Keep the v3 shards on disk while the v4 manifest references them.

After two training-only epochs with standardization, the fixed probe had maximum
presence range 0.9583, pig-count range 0.1041 and block-count range 3.6484, compared
with zero for the old parser. This is evidence against the diagnosed collapse,
**not** evidence that calibration or gameplay will pass. No calibration labels
were used to select the repair; objective thresholds remain unchanged.

Regression coverage includes negative-projection gradients, full-resolution
image-to-independent-presence learning, normalization checkpoint round-trip,
image-blind parser detection, read-only shard reuse/mismatch rejection, and
final-epoch resume. Verification passed: 75 focused tests, the public no-write
dry run, real CUDA parser/carrier/world-model smoke, and a real one-epoch training
checkpoint/reload/resume check in a temporary directory. Reloaded sensitivity
matched exactly; temporary weights were removed after the test.
The long 20-epoch repair and subsequent evaluation remain
operator work; bounded training probes do not replace them.

## Operator commands

The v4 plan and reused data manifest are prepared. Start the full resumable run:

```bash
python -u -m scripts.run_issue_70_parser_repair \
  --run-repair --device cuda \
  2>&1 | tee -a data/issue-70-parser-repair-v4.log
```

It logs stages, epochs, frames, losses, sensitivity and scoring progress. Only
after `pilot_allowed=True`:

```bash
python -u -m scripts.run_issue_70_parser_repair \
  --run-pilot --device cuda --start-display \
  2>&1 | tee -a data/issue-70-pilot-v4.log
```

Review `data/issue-70-pilot-audit-v4/`. Then publish and validate:

```bash
python -u -m scripts.run_issue_70_parser_repair \
  --publish 2>&1 | tee -a data/issue-70-publish-v4.log
python -u -m scripts.run_issue_70_parser_repair \
  --validate 2>&1 | tee -a data/issue-70-validate-v4.log
```

A completed negative objective/ranking gate skips gameplay and can be published
and validated. An exception (including image-sensitivity failure) instead needs
investigation before continuing. This repair does not authorize #64 or open any
final evaluation. Keep #70 open pending the operator's new results.
