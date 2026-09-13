# Gradient diagnosis: conflict is not the general explanation

All 72 frozen-weight probes completed under source `da969c0`. Every local /
recursive / symbolic sum matched the production loss, and checkpoint tensors
remained exactly unchanged. The probe used 44.078 active seconds, peak RSS
1,547.723 MiB and allocated CUDA 175.497 MiB. Full records are in
[the report](../data/issue-76-gradient-tradeoff/report.json).

At the boundary checkpoints, none of the 36 total gradients opposes the
horizon-750 first-step gradient, and none of the local/recursive gradient
pairs conflicts. At the full-duration checkpoints, only 2/36 total gradients
oppose that anchor, and only 2/36 local/recursive pairs conflict. These are
different cells: cross-horizon conflicts occur in pure seed760930003 at
horizons50/250; within-horizon conflicts occur in pure seed760930002 at
horizons50/250. Thus direct conflict is not a general explanation at these
five training anchors and two checkpoint snapshots.

Recursive gradient norms nevertheless dominate: at the completed checkpoints
their ratio to local norms ranges from approximately11.7 to1001.3. Positive
alignment does not mean balanced influence or guarantee finite-step descent.

The retained AdamW delta predicts increased anchor loss in11/36 boundary
cells (8 horizon50, 3 horizon250) and4/36 full-duration cells (2 at each of
those horizons). Eleven boundary cells and two completed-stage cells reverse
the current ordinary negative-gradient direction. This identifies an
optimizer-direction issue in some cells, but does not isolate momentum from
second-moment preconditioning, prove a whole-fit cause, or explain every
observed regression. All24 horizon750 AdamW directions predict local descent.

Next distinguish first-order direction from finite-step overshoot using
temporary parameter perturbations at the completed checkpoints. No automatic
optimizer reset or new training run is justified by this diagnostic alone.
All conclusions concern the declared training anchors, not held-out
generalization, gameplay, or #76 advancement. The failed stage stays failed.
