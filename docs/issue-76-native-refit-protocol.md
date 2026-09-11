# #76 native refit: symmetric implementation contract before fitting

This refit uses only the prospectively frozen native-development collection.
Engineering C1/C2/native/C4 observations are excluded from fitting; synthetic
optimizer smokes are separate test evidence. Calibration/model-selection roles
cannot enter any representation, predictor or controller optimizer.

## Common representation costs and supervision

Instantiate the existing small RGB convolutional slot backbone without its
relation and macro heads: 240,906 trainable visual parameters, from the prior
290,702-parameter parser. Fit only physical-slot presence, projected centers and
object kinds; do not teach either shared encoder symbolic contact/support/macro
predicates. The removal affects both arms identically and avoids an unused
symbolic readout in the pure method. No old parser or checkpoint is modified.

The common observation/action GRU has 59,520 parameters. Its training-only
next-visual auxiliary linear readout has 71,272 parameters (300-value current
carrier plus requested observed time delta to 236 values). Common deployment
therefore has 300,426 parameters; common fitting has 371,698. These counts were
calculated from metadata-only model construction before implementation, not
selected by outcome search. Report actual memory, wall and device costs too.

For each paired seed 760930001–3: train visual perception for 1,500 AdamW updates
(lr0.001, weight decay0.0001, batch32), then freeze it; cache its 236-value visual
carrier for each assigned usable segment; train memory for 500 AdamW updates
(same lr/decay) on a full observed gameplay-episode sequence per update. The
memory auxiliary loss predicts the next actual observed visual state conditioned
on that target's time difference. It does not use future inputs to construct
the current memory. A new episode resets state; shots and missing-time gaps do
not. Action-only events occur once after each pre-intervention observation.

Sample training lineages in frozen ordinal rotation; each update's frame choices
use its recorded seed/update counter. Do not replace an unusable assigned lineage
with a favorable one. A skipped update remains skipped in both paired methods'
exposure ledger. History can train on intact censored windows but never invents
a next observation after the final recorded endpoint. All RGB and all native
oracle steps are retained; prepared model-specific shards may be derived.

Physical-slot centers use body position, or enabled non-trigger collider geometry
for static entities, projected with training capture metadata. Entities outside
the fixed 18-slot vocabulary remain in raw truth, not hidden extra agent inputs.
Native macro labels become available only once an actual stable-entered/exited
event establishes the state; structure instability additionally requires the
immediately preceding native support set. This is a new native-event-based
label contract, not the old two-tick derivation. Only the hybrid optimizer uses
these symbolic labels. Full-world topology is not asserted from a projected
18-slot relation matrix.

## Paired predictors and controllers

Both independent predictors receive the same frozen 300-value carrier and known
five-value shot context. Repeating that context during prediction does not
execute another action or update real observation memory. Hybrid width384 has
1,903,290 parameters; pure width456 has1,920,828 (0.9215% difference), selected
by the already declared count-only rule. No dead padding, inherited hybrid
weights, pure symbolic heads or pure mode embedding.

Each arm/seed has6,000 AdamW updates (lr0.0003, decay0.0001), batch32 sampled
positions within the same rotated training lineage. Horizons cycle50/250/750
native steps identically in both arms; hybrid modes cycle across all nine
mode/horizon pairs. Use up to four exact local/recursive endpoints with masks,
full300-value MSE and0.01 mean-square excess beyond absolute carrier value2.
The hybrid additionally uses its selected symbolic loss with explicit target
availability; pure training never receives that symbolic batch dictionary.

Controllers predict permitted mode/horizon choices from current300-carrier,
known shot context and remaining native horizon only. Hybrid has nine outputs;
pure has three. Match total controller parameters within2% with a count-only
width choice and no dead logits. Fit each for2,000 AdamW updates (lr0.001,
decay0.0001), batch32. Offline teacher costs use actual training endpoints plus
fixed0.01 normalized transition-MAC cost, with native-time-weighted local MSE
and dynamic programming to the available window endpoint (at most11,250 steps).
Targets from absent endpoints are masked. Training oracle labels never enter
deployed controller inference.
The common MAC normalization reference is1,852,800 (the declared full-width
hybrid micro transition), identical for both arms; do not normalize each arm
by its own maximum and erase their actual cost difference. All optimizers use
gradient-norm clipping at1, symmetrically.

Retain strongest fixed pure and fixed hybrid baselines from all three/nine
choices, plus adaptive pure/hybrid and a no-model reference. Calibration chooses
fixed baselines; model selection scores the frozen choices. Tie order is the
declared pair ordering, not a favorable seed. Equal-time recursive comparisons
request11,250 steps/4.5s; shorter physical endpoints are unavailable, not padded
with terminal absorption. Report all assigned denominators and availability.

## Resource limits, progress and scientific interpretation

Per seed, common fit/cache/history share3600 active seconds. Each predictor
arm/seed gets3600s; controller fitting and its readiness diagnostics share1800s
per arm/seed. These are ceilings of3h common+6h predictors+3h controllers, with
no transfer between arms. Count GPU-resident active wall, including input work,
and report CUDA allocation/reservation and CPU preprocessing separately. Limit
fitting to8GiB allocated CUDA and12GiB process-tree RSS, and new working data to
256GiB. Stop, preserve partial checkpoints and report incomplete jobs on limits.
CPU shard preparation uses the unspent portion of the12-hour collection window;
its persistent budget is separate from the GPU stages and included in reporting.
Resume continues the saved optimizer/update schedule and consumed budget; it
does not reset allowances. No hyperparameter sweep or seed replacement.
A Ctrl-C requests a clean pause at a completed update/shard boundary. An unclean
hard kill or failed optimizer job is retained for audit, not silently replayed.

Preparation, fitting and diagnostic commands must print progress/elapsed/ETA
and be dry-run/synthetic-smoke tested before handoff. A dry-run cannot certify
uncollected data or unfitted weights. Missing/corrupt capture failures stay in
the data report; require at least90% usable assigned lineages in each role and
family before fitting, without replacing any. Report exact effective exposure.

Offline prediction/available-symbol validity is a prerequisite diagnostic, not
actual policy gameplay success, ranking power or authorization for fresh access.
Require complete matched jobs and at least90% model-selection endpoint coverage,
finite predictions, bounded-error/calibration audits and a viable legal gameplay
candidate before a fresh statistical protocol can be frozen. No fitted or
synthetic artifact alone can set #64/#65 supported. Durable archival, actual
gameplay/baseline/useful-mode evidence and fresh power/multiplicity/margins remain
separate gates; absent evidence stays absent rather than being defaulted true.
