# Conditioned event-readout engineering intervention

Retain the completed balanced full-rollout engineering stage and all of its
failed utility results. Reuse its exact three-seed fixed endpoints, four
predicted trajectory checkpoints, initial carriers, actions, validated event
targets, and role inventory. Dynamics, perception, trajectory rollout code,
selector architecture/imitation rule, compute coefficient, comparator
selection, and engineering thresholds do not change.

The training-only differential probe established that the shallow unconditioned
readout had not learned training ranking. A state-action multiplicative readout
with stronger action scaling learned all 22 informative training lineages;
neither calibration nor model-selection outcomes selected that recipe.

Use a 256-dimensional SiLU state projection of initial plus fixed endpoint,
modulated multiplicatively by one plus a tanh action projection. Follow it by
a 256-dimensional SiLU hidden layer and four stop-kind logits. Multiply only
the readout's two drag channels by six, converting the frozen frame-height
action encoding to the 80-pixel drag scale; dynamics still receives its original
action code. Both arms use the exact same readout architecture and parameter
count, initialized from their shared seed without warm starts.

Fit 9,000 AdamW readout updates at lr 0.001 in each arm/seed. Each update uses
all valid training records and cycles fixed pairs, giving identical total
record exposure to both arms (1,000 sweeps per hybrid pair; the three pure pairs
cycle through the same 9,000 updates). Classification is the mean of globally
class-weighted per-record cross-entropy, not a separately normalized weighted
mean per lineage. Average the existing within-lineage clear-ranking term over
all valid training groups, keeping non-informative groups' ranking term zero.
Use the existing finite guarded norm-clipping fitter. Invalid captures retain
their missing targets and never enter the optimizer.

Train the original outcome-class/compute-cost pair teacher and trajectory-state
selector for the unchanged 2,000 updates. Deploy the same repeated adaptive
rollout through 11,250 native steps and the same ensemble/calibration selection
and actual primary-pure MAC comparison. All source, work, failure, and role
bindings are frozen before held-out engineering scoring.

The old model-selection cohort is exposed and remains engineering-only. A
positive result warrants collecting a newly frozen disjoint development
readiness cohort; it never opens fresh evaluation or passes advancement by
itself. No threshold changes, seed deletion, or outcome-based technical retries
are permitted within this frozen stage.
