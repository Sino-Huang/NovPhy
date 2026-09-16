# Conditioned event engineering findings

The frozen conditioned-readout stage completed all six heads and six selectors
without numerical or resource errors. All completed model/Adam tensors were
finite; 66,000 scheduled optimizer updates were applied.

Training action ranking is now learned: both adaptive arms and every fixed
pair hit all 22 informative training groups. The retained calibration result
was hybrid 0/11 informative groups and pure 2/11. The exposed engineering
model-selection result was hybrid adaptive 1/10 informative groups, selected
hybrid fixed 2/10, primary pure 1/10, and no-model prior 3/10 (33 total groups).
Thus the conditioned fitting remedy fixed training learnability but did not
transfer action utility across lineages.

Hybrid mode use remained non-degenerate (second-mode fraction 0.1553), but
horizon use collapsed almost entirely to 750 steps (second-horizon fraction
0.00052). Actual adaptive MAC was 0.2499 times the calibration-selected primary
pure fixed-250 policy. Finite/useful/mode/compute checks passed; all relative
utility checks and the horizon check failed. No readiness or advancement claim
follows, and all earlier failures remain unchanged.

The remaining bottleneck is transfer/coverage, not more optimizer iterations
on the same 25 training clear-event records. Before another implementation or
larger collection, inspect training/cohort coverage, capture timing and target
semantics, and whether geometry/relational structure supplies a justified
inductive bias. Any larger training or readiness collection needs a separate
numeric, outcome-independent, source-bound protocol. Old exposed engineering
outcomes cannot be relabeled as new readiness evidence.
