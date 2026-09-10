# #76 Phase A execution contract

Prospective implementation protocol, 2026-09-10. This freezes the explanatory
diagnostic and metadata inventory, not a new superiority test or a fresh cohort.
The accepted #75 disposition is `not_supported_by_this_pilot`. Its thresholds,
results and source-repair receipt remain unchanged. No #64/#65 authorization.

## A0.1: fixed-pair dynamics versus adaptation

Select24 already-opened calibration states from the #72/#74 inventory: within
each of `type010101` and `type010102`, sort by full state identity and take indices
`round(j * (n-1) / 11)` for j=0..11. Freeze all exact identities, generation seeds,
12 candidate identities/actions and observation-manifest references before image
parsing or diagnostic scoring. No outcomes influence membership; missing targets
or failed predictions remain in the report, not replaced.

Use paired model seeds20260908/20260909/20260910 from #74. Fixed systems:

- independent continuous dynamics at h1,h5,h15;
- the hybrid checkpoint at each h1,h5,h15 and each continuous,micro,macro mode.

All models start from the same B=1 initial agent carrier for a state. Reuse the
accepted #70 CNN,236-value carrier, action encoding and12-action grid. No gradients,
fitting, new encoder, altered score or optimizer search. Predict recursively to225
fixed steps with no intermediate truth resets, retaining predictions at elapsed
steps15,30,60,150,225. Fixed-pair execution reaches each endpoint exactly.

At each retained time t, also predict a separate LOCAL transition of duration h
from the observed carrier at t-h to the target at t. Local context observations
are evaluation-only, never inputs to counterfactual action ranking. At t-h=0 use
the common initial carrier. The two diagnostics must remain labeled separately.

Reuse unchanged #74 continuous fixed-horizon and hybrid continuous-h15 endpoint
scores for audit/comparison, not as substitutes for missing intermediate curves.
Report original and covered adaptive controllers from #74/#75 at225 using their
saved outcomes/traces. No fabricated/interpolated adaptive intermediate curves.

### Target and timing semantics

Read only agent PNGs required for the listed endpoints, local contexts and their
immediate predecessor frames. Use the original carrier builder. Preserve the
#70 endpoint parser batch convention: sorted unique indices
`(0, p-1, p, last-1, last)` for each observed target position p. The CNN uses
per-image normalization; cached parse outputs may be reused only for the same
frame and batch size. The runtime initial carrier still uses B=1 and the separate
B=3 SAME-image cache audit, never a batch of future images.

Recheck the reconstructed225 target against its archived carrier at atol/rtol1e-5
and retain the archived225 tensor exactly. Terminal absorption is allowed only
after a recorded stable stopping condition; disclose requested time, actual last
observed time, capture/manifest identities and absorbed-target status. Nonstable
truncations are unavailable, not a fabricated complete horizon. Record actual
launch, collision and stable-terminal offsets from the existing capture. Do not
assume the previous96-trace launch sample covers every selected replay.

Aggregate carrier MSE uses all236 values, preserving #74's definition. Additional
field errors use TARGET availability masks (never prediction masks): presence
probability/presence-available; kind index/expected-kind probability/kind-available;
center x/y/center-available; motion x/y/motion-available. The two global history/
elapsed-time features are reported separately. Also report availability-bit MSE
and parsed pig/block count MAE. No post-outcome normalization, weight tuning or
rescaling. Undefined masked errors remain null with available counts.

These targets are learned-perception carriers, not oracle physics states. MSE
does not imply a physical violation or reveal which unreported fields failed.
The common225-step endpoint can precede settling; action outcomes below are
settled outcomes and are not mislabeled225-step engine truth.

### Outcomes, contrasts, uncertainty and compute

Rank candidates by the unchanged `1000*pig_probability_sum + block_probability_sum`
objective at225, tie by original candidate ordinal. Reuse the existing normalized
accepted settled-count-cost regret (failed prediction/state regret1), top1/top3,
realized pig removal/level-clear/contact/support outcomes and best-observed grid
headroom. Frozen no-model comparator is ordinal09, with its development-selection
optimism disclosed. No gameplay-success claim from replay top1 or motion alone.

Report separately: hybrid-training effects at the same fixed continuous h;
symbolic execution effects within the same hybrid checkpoint and same h; original
and covered adaptive policies against fixed controls. Give all seeds, families,
tied/informative states, prediction failures and selected outcomes. Do not select
a new deployable checkpoint/controller from the diagnostic.

Equal-weight candidate means within each state, then equal-weight states; seed
repeats are clustered within state. Descriptive paired-state bootstrap10000 draws,
seed7201, averaging seed contrasts within state first. These reused-development
contrasts do not establish fresh confirmation or uniquely joint mechanisms.

Record transitions, requested/effective horizons, decoder/adapter calls, parser
work, controller calls and executed linear MACs, synchronized inference wall,
local-diagnostic overhead and cached-source work separately. Linear MACs are NOT
full FLOPs or equal total deployment compute. No infilling occurs. Retain existing
training-cost references; new fitting cost is zero. Failed partial traces remain
partial and their total work is not inferred as zero.

One cumulative3600-second active-command allowance covers target parsing and new
diagnostics, persisted across resumes and checked at saved units. Derived output
allowance2GiB, CPU RSS3GiB, CUDA allocation2GiB. No automatic budget increase.
Lost work between a hard kill and the last saved unit is explicitly unmeasured.
Validation/metadata/smoke cost is separate. A budget stop permits an honest partial
diagnostic report, never favorable partial-data inference.

## A0.2: metadata inventory, not a performance test

Read all80 local templates and the40 paired workbook rows. Emit all45 scenario ×
novelty-level cells (five scenarios, levels0..8) and40 normal/novel counterpart
pairs. Bind template paths and source text, workbook row/coordinate/restriction
values, canonical authored object IDs, object types/counts, slingshot position,
novelty mechanism, event timing and runtime-observability limitations.

Use the existing template reader and authored-ID rules. Compare against the exact
frozen18-slot vocabulary without dropping entities. Static slot compatibility is
necessary, not proof of CNN recognition, object coverage after generation, action
reachability or game/capture support. Record the normal-only two-family collector
restriction, new object appearances, special agents/forces, right-side slingshot
action incompatibility and absent event/force state. Engine labels are metadata,
never privileged runtime model inputs.

No generated-level outcomes, prior success-filter CSVs, model-selection/final
images or runtime game sessions are read by this inventory. Prior exposure is
known from the #62–#75 lineage contracts only; templates elsewhere in the repo
are NOT declared globally unexposed without a future partition audit.

## A0.3 and fresh-evaluation feasibility

Publish a small metadata-selected recommendation for additional normal rolling/
falling/sliding and matched appearance/physical-novelty inspections, with unsupported
slots/action/state requirements explicit. This recommendation is not membership
for a live experiment. Give a costed implementation/compatibility-smoke proposal
that requires explicit approval and a new exact freeze before any new capture.
Keep zero-shot transfer separate from equal-budget adaptation on novel examples.

Because #75 failed the frozen candidate screen and assets remain unarchived,
this implementation must report `readiness_or_precision_insufficient` BEFORE
fresh access, even if a diagnostic curve is favorable. Broadening training,
repairing the carrier, changing the controller objective or collecting additional
novelty outcomes cannot be hidden inside this diagnostic. No numerical fresh
population/sample-size/advance claim is invented for a test that cannot yet run.

Every original #76 requirement will have an explicit ledger entry: implemented
and measured, pending operator diagnostic, stopped at declared budget, or
conditional/not executed with a concrete blocking reason. Phase B captures,
paired fresh success/precision testing, fresh action search, graphics/process
isolation and archive-before-access remain unexecuted behind the same gate.
Their absence is NOT recorded as passing or completing a fresh experiment.

## Handoff and preservation

Use new `.local-artifacts/issue-76-dynamics-diagnostic-v1/` and
`data/issue-76-review/`. Save machine-readable plans/results, CSV/SVG error curves,
an HTML review index and links to existing agent frames/videos. No new rendered
frames or WebMs are claimed. Existing code, plans, checkpoints and outcomes remain
unchanged; new source text is frozen without hashes. No commit/push or archived
release is claimed without separate authority.
