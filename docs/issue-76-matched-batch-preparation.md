# Exact-example batch regrouping preparation

This is implementation validation, not a fitted experiment or advancement
result. No optimizer, capture, or fresh evaluation was run for this check.

The preceding controller-utility diagnostic motivates testing predictor quality
before changing the controller cost objective. One candidate is the original
family-blocked batching. `scripts/issue_76_matched_batches.py` reproduces every
original sampled `(training-entry index, start)` using the parent's update-local
generator. It then permutes those rows only within their exact transition pair.
The original skipped update positions and abstraction/horizon sequence remain
unchanged. Local RNGs do not perturb predictor initialization.

Validation against the repaired-representation training assignment, all three
seeds (760930001–760930003), and both independently fitted arms found:

- Exactly 5,952 non-skipped batches and 48 skipped positions per arm/seed.
- Exactly 190,464 sampled starts per arm/seed, with identical multiplicities
  within every transition pair before and after regrouping.
- All 5,952 regrouped batches contain multiple training lineages.
- On the first assigned usable training lineage, 18 batch comparisons covering
  both arms' update schedules match the existing sampler's tensors exactly.

Two automated tests also cover deterministic regrouping, unchanged global RNG,
per-pair sample multiplicities, skipped updates, symbolic-target exclusion from
the pure arm, and exact masks/actions/targets across repeated fixed-step
coordinates in different shot segments. Run with the initialized `novphy`
environment and sourced `env.sh`:

```
python -m unittest tests.test_issue_76_matched_batches
```

Next: bind a separate prospective fit plan, keep the parent checkpoints and
training loss unchanged, cache the assigned metadata/carriers in memory, and
fit fresh paired predictors using these schedules. The indexed batch helper
currently prioritizes matching the original target construction; it performs
no image loading. No claim about improved predictor quality follows from
these sampling checks alone.
