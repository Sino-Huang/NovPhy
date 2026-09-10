# #76 next-stage proposal: canonical physics/history and symmetric refit

This is a costed proposal for direction, **not an approved execution freeze**.
The completed [compatibility findings](issue-76-compatibility-findings.md) show
that neither a longer run nor a cosmetic pig replacement fixes readiness.
Do not open fresh data, change the old checkpoints, or advance #64/#65 from it.

## Recommended scope

First restore canonical asset/behavior semantics in a separately versioned
instrumented player. Recover original assets and the actual big-pig hit-count
behavior from the bundled reference game. Audit normal **and** novel properties;
do not merely import PinkBigPig while leaving the normal control on different
mass, damping and behavior. Preserve the old player and every C1 failure.

Verify resource registration, generated entity identities, collider/rigid-body
properties, distinct-bird hit/death behavior, aligned agent observations and
request-71 events using bounded engineering fixtures. Changes to the canonical
normal/novel comparison must be declared, not called appearance-only by default.
This is an asset/behavior migration, not permission to alter novelty definitions.

Then specify a shared agent-observable history contract. Previous bird impacts
cannot be supplied as true engine counters to one arm. Both dynamics approaches
must receive the same permitted observation/action history and missing-data
handling. Freeze that representation and its training method before collecting
its training/development data or fitting either model. The current two-frame
carrier is not silently assumed sufficient for hit-count dynamics.

Fit the common perception/history component symmetrically, then the independently
trained pure-continuous and hybrid predictors with paired seeds, equal declared
data/update exposure, and their matched controllers. Keep target-novelty fitting
out of any zero-shot claim. A few-shot study would require its own separate
training/update and held-out novel-lineage protocol.

Run a new, explicitly bounded readiness pilot against the strongest permitted
continuous, fixed-hybrid and no-model controls. This is a new candidate under a
new runtime/representation, not a reinterpretation of #75. If it fails, stop;
there is no automatic optimizer sweep or fresh evaluation.

## Proposed resource envelope to approve

These are upper planning allowances, not measured performance or power claims.
The larger stage exceeds C1's 2GiB/30-minute capture allowance, so it should not
be launched under the earlier smoke authorization.

| Work | Proposed allowance |
| --- | --- |
| Canonical asset/behavior port, semantic fixtures and player build | 8 engineering hours; up to 8 separately frozen engineering captures, 30 minutes capture wall, 3GiB aggregate RSS, 2GiB data |
| Shared observation-history contract, refit/data integration and regression tests | 16 further engineering hours |
| New training/development collection | At most 350 base lineages, up to 3 shots per lineage; 12 hours active wall; at most 4 isolated workers and 12GiB aggregate RSS |
| Shared perception/history fitting | 3 GPU-hours |
| Paired pure/hybrid predictor fitting | 6 GPU-hours total, allocated equally across the two arms and three paired seeds |
| Matched controller fitting and readiness scoring | 3 GPU-hours total, with symmetric per-arm allocations |
| New captured/derived working data | 256GiB maximum; stop rather than discard failures or enlarge the cap |

Exact per-cell/role counts, reserved seeds, frame-selection rules, architecture,
updates, margins and readiness decisions still need a source-bound protocol
after the asset/behavior contract is established. The caps above must not be
mistaken for that protocol. In particular, 350 base lineages is a development
ceiling, not a justified fresh or sealed sample size.

The C1 smoke measured roughly 16–24 seconds per one-shot attempt in this
environment, but canonical multi-shot behavior and larger history storage may
cost more. The collection/storage allowances therefore require operator
agreement and a new bounded real smoke. Full collection and fitting commands
should be left to the operator after tests, no-write dry-run and that smoke,
with foreground progress/ETA and deterministic resume.

## Archive and fresh-test ordering

Confirm permission to commit/push the scoped #70–#76 prerequisites and new work,
and specify the durable large-asset archive destination. Source/checkpoint/repair
and failed-run inventories must have an explicit verified archival record before
fresh access; neither `.local-artifacts` nor a flag constitutes a pushed release.

Only after a viable candidate and archived assets exist can the fresh numerical
protocol, paired success power/precision, practical margins, multiplicity,
matched compute, once-only disjoint generation and gameplay execution be frozen
and run. Only its exactly validated supported disposition can advance #64/#65.
No fresh or sealed outcome access is needed to approve or implement the stages
above.
