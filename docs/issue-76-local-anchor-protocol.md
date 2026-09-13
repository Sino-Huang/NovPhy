# Mixed-start continuation with targeted local supervision

The validated boundary-only continuation improved the two residual pure local
errors but did not qualify and introduced other regressions. Preserve that
result, its costs and its stronger endpoint references. Start this separate
candidate from all six lower-rate checkpoints, not a seed-dependent mixture
of stages. The lower-rate stage retained all 36 endpoint comparisons and
qualified four of six arm/seed combinations, versus two for boundary-only.

## One prospective recipe

Use paired seeds 760930001, 760930002 and 760930003 and both independently
trained arms. Retain model parameters and complete AdamW history, learning
rate 0.00003, weight decay 0.0001 and gradient clip 1. Shared 352-value carrier,
capacity matching, action encoding and all modes/horizons remain unchanged.

The sole change relative to lower-rate training is local-loss weight 10 at
horizon 750, for every hybrid mode and for pure continuous. Weights remain 1
at horizons 50 and 250. Full recursive loss, carrier-bound penalty and hybrid
symbolic loss keep their original weights. No oracle information is added.
Local supervision remains the mean of the first four observed adjacent
transitions, not a selected diagnostic lineage or a recursive truth reset.

The paired one-step probe showed strong horizon-750 local improvements in the
two failing pure seeds while improving their endpoints. Weighting every
horizon also produced an 8.581% local regression for pure seed 760930003 at
horizon 250; this targeted recipe does not change that horizon's weight.
Shared-parameter interference remains possible. These observations justify
a candidate, not a guarantee of qualified training or gameplay performance.

Restore the original 50% boundary / 50% uniform-start schedule and exact
within-pair mixing across all 250 assigned training lineages. Neither of the
two unusable assignments is replaced. Use all actual shot boundaries, not
just the five diagnostic first shots. Continue exactly 1,800 scheduled
positions, absolute 13,800..15,599: 1,785 applied and 15 skipped updates,
57,120 starts per arm/seed, half nominal boundary and half nominal uniform.
Uniform draws may also land on boundaries; report this separately.

These positions are a new branch from the lower-rate checkpoints, not a
continuation of the boundary-only optimizer. The same assignment window is
intentional; all costs of both branches are charged. Use the final scheduled
checkpoint only, with no early stopping, extension or best-checkpoint search.
The fixed length allows roughly seven assigned-lineage passes; it is not a
power claim inferred from a single finite step.

## Gates and costs

Preserve every earlier curve and requirement: all 36 policy/seed endpoint
comparisons need mean MSE <=0.8 times unchanged carrier and >=4/5 wins.
Retain boundary initial/endpoint accuracy, strongest boundary/mixed pure
endpoints, full-duration pure endpoints, and both lower-rate anchor initial
and endpoint accuracy within 1.10. Additionally require all six horizon-750
anchors to retain both boundary-focused initial and endpoint accuracy within
1.10. This keeps stronger references regardless of source-branch selection.
All six arm/seed qualifications must pass; no seed, policy or lineage drops.

Before fitting, audit all real schedules for exact paired horizon-level
lineage/start multiplicities and inspect source checkpoint completion/state.
Archive this protocol, source, public plan and preflight. Initialize novphy
and source env.sh. Sequential CUDA fits have 1,800 active seconds each;
preflight and each preparation have 300 CPU seconds; diagnostics 600 CPU
seconds; independent validation 180 CPU seconds. Keep 12 GiB RSS, 8 GiB
allocated CUDA and 1 GiB new local artifacts. Retain all parent/branch,
representation, probe and validation costs. Do not count reused costs twice.

After all six fits, replay all 30 diagnostic records and 180 policy curves.
Independent validation must exactly reproduce curves/decisions and check
source-bound plans, all checkpoint counters, optimizer settings and actual
per-parameter step increments, and the complete 11-job budget inventory.
No new capture, fresh/final access or controller fitting occurs in this stage.

Training qualification is not advancement. Development readiness, controllers,
actual adaptive gameplay against strongest pure, same-hybrid fixed and strong
prior controls, practical margins with uncertainty, useful nondegenerate
adaptation, and matched-compute constraints remain mandatory for #76.
