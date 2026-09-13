# Targeted endpoint continuation

The completed targeted-local stage retains all 36 endpoint comparisons but
fails pure-seed-1 endpoint retention and pure-seed-2 local retention. The
[paired joint probe](issue-76-joint-interventions-findings.md) found that adding
endpoint MSE at horizon 750 improves both anchor metrics from the original
weights in all 12 paired horizon-750 cells, and endpoint error versus control
in all 12. This supports a new candidate, not a full-fit or advancement claim.

## Frozen recipe

Continue all six completed targeted-local checkpoints, paired seeds 760930001,
760930002 and 760930003. Preserve model parameters and complete AdamW history,
learning rate 0.00003, weight decay 0.0001, gradient clip 1, common 352-value
carrier, matched capacities, action encoding and every mode/horizon.

Add one unit-weight MSE term for the fully recursive 11,250-native-step
endpoint only at horizon 750. Keep local weight 10 at that horizon and local
weight 1 at horizons 50/250. Keep all previous recursive, bound and symbolic
terms unchanged. The added endpoint has no truth resets or truncated gradient
chain. Supervise only rows with an actually observed endpoint. If none are
available in a batch, retain its original available local/recursive/symbolic
loss; do not invent a target or skip otherwise usable supervision.

Use the existing mixed-start schedule: half nominal observed shot boundaries,
half uniform frames, then exact within-pair mixing. Keep all 250 assigned
training lineages and both unusable assignments. Use no held-out data for
fitting. Run exactly 900 scheduled positions per arm/seed, absolute indices
15,600..16,499: 893 applied updates and seven skips, 28,576 starts with 14,288
nominal starts of each kind. This count follows actual unusable indices
56/132 and the partial pass; neither assignment is replaced.

The 900-position duration is a prospective engineering choice of roughly
three and a half lineage passes for this narrow continuation. It is not a
power guarantee from a one-step probe. Use only the final scheduled checkpoint;
no outcome-dependent extension, early stopping or checkpoint selection.

## Unchanged and stronger requirements

Recompute all 30 assigned training diagnostic records and 180 policy curves.
All 36 policy/seed endpoint comparisons require mean MSE <=0.8 times unchanged
carrier and at least four of five individual wins. Preserve every boundary,
mixed, full-duration, lower-rate and boundary-focused accuracy-retention
requirement. Additionally require all six horizon-750 anchors to retain both
local and endpoint error within 1.10 times their targeted-local source.
This keeps the new stronger pure-seed-3 endpoint rather than discarding it.
Every arm/seed must qualify. No seed, mode, horizon or lineage is dropped.

## Execution and accounting

Preflight all source checkpoints and actual paired horizon-level sampled-start
multiplicities before fitting. Archive source, protocol, plan and preflight
before `--run`. Initialize novphy and source env.sh. Run sequential CUDA fits
with 900 active seconds per arm/seed, 300 CPU seconds for preflight and each
seed's preparation, 600 CPU seconds for diagnostics, and 180 CPU seconds for
independent validation. Keep 12 GiB RSS, 8 GiB allocated CUDA and 1 GiB new
local artifacts. Preserve all parent, branch, representation, diagnostic and
validation costs, without counting shared reused work twice.

Independent validation must exactly recompute curves and decisions, compare
frozen/public/source-bound plans, verify all checkpoint counts and optimizer
settings and actual per-parameter step increments, and check the complete
11-job resource inventory. No capture, fresh/final access or controller fitting
occurs here. A training qualification cannot substitute for actual adaptive
gameplay against strongest pure, same-hybrid fixed and strong prior controls,
useful adaptation, practical margins with uncertainty and matched compute.
