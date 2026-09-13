# Prospective exact-example mixed predictor fitting

Test whether mixing lineages within predictor batches reduces the recursive
error found in the repaired-representation checkpoints. This changes batch
grouping/order only: no added examples, new architecture, loss, labels, parser,
history, controller objective, or rollout horizon. Batch-wise loss normalization
still follows the unchanged parent objective; regrouping can change which
available targets share a normalization denominator.

Freeze the executable and this protocol before fitting. Source is the complete
`issue-76-repaired-representation-v1` plan and its completed common checkpoints.
Use all three seeds 760930001–3 and independently initialized hybrid/pure arms,
with the parent's widths and initial seeds. Reuse its common carriers without
updates; retain original common costs, including their inherited parser costs
in the source protocol. No new data or fresh access, and no concurrent GPU fit.

For each arm/seed reconstruct all 6,000 original update-local draws of 32 starts
from the 250 assigned training lineages. Preserve the 48 skipped updates and
all 190,464 sampled `(lineage, start)` multiplicities within each exact
abstraction/horizon pair. Deterministically permute rows within each pair using
local generator seed `seed + 1009 * (100000 + pair_group_index)`, where groups
follow first appearance in the original schedule. Partition into 32-row batches
at the original non-skipped updates of that pair. Do not redistribute across
pairs or insert replacements for failed assignments.

Fit all six predictors for 6,000 scheduled updates, AdamW lr .0003, weight decay
.0001, gradient cap 1, parent four-offset local/recursive losses and hybrid
symbolic losses. Pure receives only carrier, mask, and action tensors. Explicit
indices retain exact observed endpoints within the same shot; missing endpoints
stay masked and zero-filled. Cache metadata and carriers in CPU memory, not RGB
or pre-materialized batches. Each arm has a separate optimizer and checkpoint;
no warm start, seed selection, adaptive stopping, or additional updates.

After all six fits, compare each against its own unchanged parent checkpoint on
the first assigned training lineage in each of the five families. Use the first
shot and exact 11,250-fixed-step endpoint; missing endpoints fail qualification,
not replacement. Retain all nine hybrid and three pure fixed-policy recursive
carrier MSEs plus the unchanged-carrier diagnostic. No controller is fitted or
used in this stage, and no fixed policy is called a gameplay policy.

Qualification requires, separately in every seed, hybrid fixed-750-micro mean
MSE at most 80% of its parent and lower error than the unchanged carrier on at
least four of five lineages. Pure fixed-750-continuous mean MSE must be at most
110% of its parent in every seed. These are diagnostic progression criteria,
not the advancement gate or evidence of held-out generalization. Failure
remains published without retry or extending this fit. Passing permits planning
the next development controller/readiness experiment, not fresh evaluation.

Root `.local-artifacts/issue-76-matched-dynamics-v1`; publication
`data/issue-76-matched-dynamics`. Per-seed CPU preparation ceiling 300 active
seconds; per-arm/seed predictor ceiling 1,800 active seconds; full CPU diagnostic
ceiling 600 active seconds. CPU RSS <=12 GiB, allocated GPU <=8 GiB, new artifacts
<=1 GiB. Persistent budget accounting charges preparation, fitting, and diagnostic
work separately; common training is reused, not free. Interrupted fits resume
their exact schedule/checkpoint; failed or exceeded fits need explicit audit.
Expected runtime is several minutes to tens of minutes. These are ceilings, not
work targets, and unused time is not transferred.

Initialize `novphy` and source `env.sh`. Run module without flags for a no-write
plan check; `--prepare` freezes and publishes the prospective plan. Commit/push
and record the freeze in #76 before `--run --device cuda`. Publish all 30 paired
qualification records, six summaries, final checkpoints, and measured costs.
No gameplay, fresh-experiment, #64/#65, or advancement authorization follows
from this diagnostic implementation.
