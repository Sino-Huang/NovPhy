# Prospective five-image learnability check

Freeze this protocol and executable before optimization. The v2 and order-only
experiments remain negative evidence. This is a single training-only diagnostic,
not a new candidate selection sweep or a replacement advancement experiment.

Use first frame index 0 of the first usable training episode in each sorted
family: issue-76-development-001, 071, 141, 211, 281. These are the five images
already used in the collapse and gradient probes, selected by metadata rather
than outcomes. Reuse their prepared 64×96 RGB images and native labels; no
new rendering, resizing, label derivation, calibration, or model-selection read.

Initialize the unchanged NativeVisualParser with seed 760930001. Perform exactly
1,000 updates on the same mixed five-image batch, using the original
perception_loss, AdamW at .001, weight decay .0001, and gradient norm cap 1.
No pretrained weights, architecture changes, alternate resolutions, optimizer
variants, best-checkpoint selection, or extra updates after inspection.

Record initial and final loss, all five predicted/true block counts, task-slot
presence Brier error, pig count error, and present task-slot center error. A
minimal batch fit succeeds only if final block-count MAE is at most .1 **and**
task-slot presence Brier error is at most .01. Evaluate the final update, not an
earlier favorable snapshot. These thresholds test fitting known images, not
generalization, physical dynamics, causal history, useful symbolic switching,
gameplay, or advancement. The resulting weights are ineligible for deployment.

If it succeeds, low-resolution input and this architecture can at least fit
these known cases; diagnose the production batching/optimization regime next.
If it fails, preserve that result and investigate the representation/optimization
failure rather than launching another full refit blindly. One seed is intentional
for this minimal mechanistic check, not a substitute for the three-seed study.

Separate root `.local-artifacts/issue-76-parser-learnability-v1`, source-bound to
this protocol, executable/helpers, parent kernel source, and exact selected
entries. Allowance: 180 active seconds including loading, fitting, and scoring;
12 GiB CPU RSS, 8 GiB allocated GPU memory, 256 MiB new artifacts. Existing
FitBudget records all active work and preserves interrupted/stopped checkpoints.
No concurrent GPU research job, new hashes, or full-corpus integrity passes.

After activating `novphy` and sourcing `env.sh`, use
`python -m scripts.run_issue_76_parser_learnability` with `--dry-run`, then
`--prepare`, `--run`, `--publish`, and `--validate`, in that order. Record the
prospective protocol before `--run`. No outcome authorizes #64 or #65.
