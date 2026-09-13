# Prospective paired downstream refit with repaired shared perception

The previous complete native refit failed readiness: nearly constant perception,
large recursive hybrid error, and no symbolic controller targets. Subsequent
matched development tests restored input responsiveness with mixed batches and
sampled pig-removal recognition with training-derived pig-loss balancing. This
experiment tests their combined downstream effect. It does not isolate one of
those two perception changes from the other or replace any earlier result.

## Shared change and fixed downstream recipe

Reuse ALL three final balanced parser checkpoints, seeds 760930001–3, from
`issue-76-balanced-pig-v1`, after its exact validated publication. No extra parser
updates or best-seed choice. Bind the complete source plan, checkpoint paths,
three-seed summaries and original parser fit costs before new fitting. Both
independent dynamics arms receive the same resulting parser/history checkpoint
within each seed. Shared perception remains semantically supervised; the pure
dynamics arm is not claimed to be end-to-end symbol-free.

Use the same 350 assigned native-development members: 250 training, 50
calibration, 50 model selection. Preserve every failed assignment. Read original
prepared RGB when producing the new visual caches. Derive small metadata-only
shards by retaining every original tensor except RGB, with unchanged values,
indices, segments, role and member identity. Record the original source plan
beside the new derived plan identity. No raw trace re-derivation, new labels,
image duplication, new hashes or corpus-integrity reread.

Instantiate the same 22-slot / 352-value common architecture with the same
seed initialization. Loading the repaired parser does not draw extra random
values: its constructor consumes the same initialization as the original
parser, preserving the following history initialization. Cache all usable
members' visual carriers, train the unchanged next-observation history objective
for 500 scheduled ordinal-rotation updates at lr .001, then cache histories.
Only training members enter the history optimizer; calibration/model-selection
caches are inference only. The parser has no symbolic readout/supervision.

Reuse the parent implementation unchanged for all downstream work: independent
hybrid and pure dynamics, original count-matched widths and 6,000 scheduled
updates per arm/seed at lr .0003, batch 32, identical source starts/horizons,
four exact local/recursive target offsets and the existing objectives. Pure
fitting receives no symbolic target dictionary. Keep the original horizon/mode
schedule; no forced symbolic use or altered cost tradeoff. Rebuild training-only
controller targets and fit 2,000 scheduled updates per arm/seed at lr .001,
batch 32. AdamW weight decay .0001 and gradient cap 1 throughout. All three seeds,
all six independent predictors, all six controllers; no optimizer/architecture
sweep, checkpoint selection or extension after looking at results.

## Measurement and interpretation

Use every original assigned calibration/model-selection member, its first-shot
anchor, and the same exact 11,250-native-step / 4.5-second endpoint. Missing
endpoints stay unavailable, never padded or relabeled as terminals. Freeze all
nine hybrid fixed pairs, three pure fixed horizons and both adaptive policies.
Retain calibration-selected fixed baselines and all individual policy results.
No selection on model-selection scores or favorable seed. Retain the unchanged
carrier no-model diagnostic; it is not the strongest no-model gameplay policy.

Report the existing recursive carrier, visual and memory errors, engine-relative
presence/center/pig/block errors, carrier-bound excess, prediction failures,
available symbolic Brier/availability, full per-step work and executed joint
mode/horizon counts. Carrier targets themselves change when perception changes:
lower old-versus-new carrier MSE is not an apples-to-apples physical improvement.
The prespecified physical comparisons are engine-relative block/pig/presence
errors for each arm's adaptive and fixed-750-continuous policies against their
corresponding original-refit policies, all three seeds and all available members.
Use these alongside within-new-refit fixed/adaptive and no-change contrasts.
Keep other fixed-pair scores visible; do not choose a favorable contrast later.

This is a development readiness diagnosis, not the once-only fresh experiment.
The parent's 90% endpoint-coverage check remains a data-availability check, not
proof of useful dynamics, gameplay or advancement. Do not label finite but
inaccurate predictions as ready. A new complete common checkpoint can be used
in a separately frozen development gameplay test, but no fresh access follows
automatically. Useful switching, strong independent pure and same-checkpoint
controls, strongest action prior, uncertainty, practical margins, complete
matched compute and fresh precision/design remain separate requirements.

## Budget, resumability and publication

Root `.local-artifacts/issue-76-repaired-representation-v1`, publication
`data/issue-76-repaired-representation`. Freeze this protocol and executable,
parent kernel source and membership, parser plans/checkpoint paths and prior
costs BEFORE execution. The earlier failures and raw/censored outcomes remain
unchanged. No collection or fresh/final access; #64/#65 remain unauthorized.

Metadata preparation: 300 active CPU seconds. New common cache/history: 3,600
active seconds per seed; predictors: 3,600 per arm/seed; controllers plus
diagnostics: 1,800 per arm/seed. These are ceilings, not required work. Expected
active runtime is roughly an hour based on the completed original fit. No
automatic retry or transfer of unused budgets. Charge parser reuse and original
preprocessing separately, retaining the parent control costs; a standalone model
must not report reused training as free. New artifacts <=8 GiB, CPU RSS <=12 GiB,
allocated GPU <=8 GiB. Retain checkpoints, skips, failures and persistent costs.
The existing training kernels also retain their original overall storage guard.
No concurrent GPU research job.

Publish all 600 assigned diagnostic records using the unchanged compacting
function, with complete step records retained in a deterministic gzip archive.
Validate publication, full archive, inventory and source bindings exactly.
The unchanged legacy collection success field is not authoritative; the separate
native-terminal audit remains 6/350 fixed-policy development wins, not learned
gameplay evidence. Do not infer a win from predicted/observed pig absence.

Activate novphy and source env.sh before Python. Run focused tests and the
no-write `python -m scripts.run_issue_76_repaired_representation --dry-run`, then
`--prepare`, record the prospective freeze, and `--run`. `--publish` and
`--validate` operate only on completed results. Clean interruptions resume the
saved schedule; unclean or failed optimizer jobs require audit, not replay.
