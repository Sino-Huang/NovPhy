# Finite-step diagnosis: smaller steps remove some, not all, regressions

All 36 assigned completed-checkpoint cells and 72 temporary perturbations
completed under frozen source `f3e6077`. No checkpoint was written, no optimizer
step was applied, and exact temporary parameter restoration passed after
every pair. The probe used 23.563 active seconds, peak RSS 1,507.949 MiB and
allocated CUDA 145.259 MiB. Full results are in
[the report](../data/issue-76-finite-step/report.json).

| Measurement | Original-sized delta | One-tenth delta |
| --- | ---: | ---: |
| Total-objective increases | 2/36 | 0/36 |
| Horizon-750 anchor-loss increases | 7/36 | 4/36 |

All 36 total directional derivatives predict descent. The original-sized
step nevertheless increases total loss for pure seed 760930001 at horizon750
and pure seed 760930003 at horizon50. One-tenth steps improve both. This
demonstrates finite-step overshoot on these declared training batches.

Three additional anchor regressions also disappear with the smaller step:
seed760930001's horizon50 hybrid micro, hybrid macro and pure continuous
directions predict anchor descent but their original-sized deltas increase
anchor loss. Four anchor regressions remain at both scales: pure seeds
760930002/3 at horizons50/250. Those four are already predicted by their
directional derivatives, so reducing step size does not change their sign.
Every horizon750 delta improves its own declared anchor at both scales.

The evidence supports a separately frozen lower-learning-rate continuation,
not an optimizer reset, architectural cause claim, or guaranteed remedy.
In particular, one-tenth step size improves every current total objective,
but that is not proof of local retention after many updates, useful recursive
prediction, generalization, or gameplay. These probes use five diagnostic
training anchors, not the complete mixed production-batch distribution.
No candidate is fitted or selected from these temporary perturbations.

The next continuation must include all six checkpoints, preserve all source
moments and exposure pairing, explicitly change the loaded AdamW learning
rate (passing a constructor rate alone is overwritten by loading its state),
and retain the original boundary/mixed accuracy references and all-policy
qualification criteria. It must have a new source-bound plan and stage-local
budget, not extend or reinterpret the failed full-duration stage. #76's
gameplay advancement gate remains unmet; #64/#65 remain unauthorized.
