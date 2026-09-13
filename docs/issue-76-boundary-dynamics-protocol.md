# Prospective shot-start-balanced predictor fitting

The exact-example mixed trial failed hybrid qualification. Its frozen-weight
diagnostic found large first-transition error and very sparse exposure to the
shot-start states that initialize deployment predictions. Test one change:
replace half of uniformly sampled training starts with observed shot starts.
This is an experiment on start-distribution mismatch, not an assumption that
the cause or advancement outcome is already established.

## Training contract

Reuse the completed common representation and metadata/carriers of
`issue-76-repaired-representation-v1`, all 250 assigned training lineages
(248 usable), and all three seeds 760930001–3. No parser/history updates, new
capture, new target derivation, held-out optimizer input, or fresh access.
Retain common fitting costs as additional standalone costs, not free reuse.
The complete parent source plan remains at its published artifact path; this
plan records the used data bindings, membership, capacity, costs, executable
source, and compact comparison records without recursively copying the entire
parent plan.

Reconstruct each of the original 6,000 scheduled updates and its 32 uniform
starts. For each non-skipped update, replace the 16 even-numbered rows' starts
with uniform draws from that assigned lineage's observed shot-segment start
indices, using the local generator seed `seed + 1009 * (1000000 + update)`.
Keep all odd-numbered rows exactly unchanged. Do this BEFORE the existing
within-pair mixing, which then uses its original seed and group order.

Thus both arms receive identical sampled lineage/start multiplicities at each
horizon, while hybrid examples are divided among its three abstraction modes.
Keep the original pair sequence and all 48 skipped updates. Each arm/seed has
5,952 applied updates, 190,464 total samples, 95,232 deliberately drawn boundary
samples, and 95,232 retained uniform draws. Some uniform draws can also happen
to be boundaries. Do not claim preservation of the previous start multiset.
Shot starts come from all observed shot segments, without selecting on success,
terminal status, or prediction error. Missing exact future targets remain
masked within the same segment; no invented endpoints or cross-shot targets.

Instantiate fresh independent hybrid and pure predictors using the same parent
architectures, count-matched widths, and initial seeds. No warm start or seed
selection. Keep all objectives and four local/recursive offsets unchanged,
AdamW lr .0003, weight decay .0001, gradient clipping at 1, batch size 32,
6,000 scheduled updates. Pure fitting receives no symbolic target dictionary.
Only sampling changes; architecture, loss, and controller incentives do not.

## Frozen diagnostic and decisions

Fit all six predictors before scoring. Compare them against all six completed
`issue-76-matched-dynamics-v1` predictors using the already published exact
curves, on the same first-assigned training lineage in each of five families.
Use the first shot; retain all nine hybrid and three pure fixed-policy curves.
Record separate observed-context local and fully recursive errors at each
policy's steps 1/2/4/8 and reachable common elapsed steps
750/1500/3000/6000/11250. Missing observed endpoints stay unavailable. No truth
resets occur in recursive predictions. These are training diagnostics, not
held-out generalization, actual gameplay success, or advancement evidence.

For each hybrid seed separately, qualification requires:

- Fixed-750-micro first-transition mean MSE <=50% of the matched reference.
- Fixed-750-micro 11,250-step mean MSE <=80% of that reference.
- Lower endpoint error than the unchanged carrier on at least 4/5 lineages.

Each pure seed's fixed-750-continuous endpoint mean MSE must be <=110% of its
matched reference. All required endpoints must be available. All three hybrid
seeds and all three pure seeds must meet their requirements; no averaging away
a failed seed. Report first-transition and endpoint decisions separately.

A passing diagnostic only supports planning the next development readiness
test. Short-horizon drift remains explicitly visible in the full curves and
cannot be hidden by the horizon-750 contrast. Useful joint mode/horizon use,
strong controls, gameplay, uncertainty, and matched compute remain separate
requirements. No controller fitting or fresh access is included in this stage.
Publish failed qualification without extending this fit or changing thresholds.

## Execution and cost

Root `.local-artifacts/issue-76-boundary-dynamics-v1`, publication
`data/issue-76-boundary-dynamics`. CPU metadata/carrier preparation <=300 active
seconds per seed; predictor fitting <=1,800 active seconds per arm/seed; full
CPU diagnostic <=600 active seconds. CPU RSS <=12 GiB, allocated GPU <=8 GiB,
new artifacts <=1 GiB. These ceilings do not transfer between fits. Expect about
30 minutes based on the completed matched trial, with no concurrent GPU
research job. Preserve checkpoints, failures, skips, and persistent cost logs.
Clean interruptions resume saved updates; failures require audit, not replay.

Initialize `novphy` and source `env.sh`. Run focused tests and the no-write module
invocation, then `--prepare`. Commit/push the source-bound plan and record it in
#76 BEFORE `--run --device cuda`. Publish every assigned diagnostic record and
all six checkpoint counters and costs. No #64/#65 authorization is implied.
