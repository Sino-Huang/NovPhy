# Prospective event-ranking model protocol

The complete causal event-development collection and its once-only audit are
the sole data source for this stage.  The audit must report all 2,600 assigned
records, a complete inventory, no resource failure, and `data_ready=true`.
This stage fits only the 100 training lineages.  Fifty calibration lineages
choose fixed controls; the 50 model-selection lineages are scored only after
all weights and calibration choices are frozen.  No fresh data are opened.

For paired seed 760930001--3, reuse the completed shared RGB/history checkpoint
and the prospectively retained 50/50 local-anchor/endpoint-anchor predictor
midpoint for each independently trained hybrid and pure-continuous arm.  No
parent parameter is updated.  Each valid decision RGB is encoded using only
the deployed agent image and timestamp.  In reset-comparable lineages, every
candidate uses the reference decision carrier; other lineages remain eligible
only for individual stop-kind supervision.

For every action, compute one exact native transition for every permitted
mode/horizon pair.  A common 256-wide event readout receives the initial
carrier, pair-specific successor, and five-value legal action context, and
predicts one of native-clear, native-fail, stable-without-clear, or intact
right-censoring.  Both arms use the identical readout architecture.  Fit 4,500
AdamW updates per arm/seed (lr 0.001, weight decay 0.0001, gradient clip 1),
rotating all training lineages and all available pairs.  Loss is inverse-count
weighted stop-kind cross entropy plus a clear-action listwise term only for
fully reset-comparable lineages with an observed clear.

Fit the existing parameter-matched controller architecture for 2,000 AdamW
updates per arm/seed (lr 0.001, weight decay 0.0001, gradient clip 1).  Its
training-only teacher is the pair with minimum observed-class loss from the
frozen event readout.  Runtime inference receives only the current carrier,
candidate action, and the declared remaining horizon.  Engine events and
outcomes never enter deployment inputs.

Average the three paired seed probabilities.  Calibration chooses one fixed
pair per arm by clear-action hits, then stop-kind log loss, with declared pair
order breaking ties.  Calibration also selects the primary independently
trained pure-continuous comparator between its adaptive policy and its selected
fixed policy by the same criteria, with adaptive first on an exact tie.  The
no-model control is selected between the per-family, Laplace-smoothed training
clear-rate action prior and the original-action policy using those same
calibration criteria; action-score ties use ordinal order.  Model-selection
readiness requires complete finite scores, at least one additional clear-action
hit by adaptive hybrid over the selected primary pure comparator, the
same-hybrid fixed control, and the selected no-model control, nonzero useful clears,
at least 5% second-mode and second-horizon use, and a hybrid/pure linear-MAC
ratio no greater than 1.10.  This is a development readiness screen, not the
fresh advancement decision.  Passing only permits freezing a separately
powered once-only fresh protocol.

Mode/horizon-use and deployment-compute checks cover exactly the same
reset-comparable model-selection groups used for clear-action ranking; invalid
or noncomparable assignments cannot contribute apparent policy diversity.

Each feature-preparation job has 3,600 active seconds.  Each event readout and
selector fit has 1,800 active seconds.  Existing 12 GiB RSS and 8 GiB allocated
CUDA ceilings apply.  All six paired jobs, unavailable records, work counts,
selection decisions, and failures are retained.  No automatic replacement,
seed dropping, threshold change, or hyperparameter sweep follows a failure.
