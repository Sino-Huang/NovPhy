# Issue 70: small-CNN perception check, then optional full rerun

The repair workflow now covers data preparation, real parser training, parser
validation, regenerated carriers, matched world-model retraining, and the #70
ranking/gameplay rerun. It keeps the old parser, checkpoints and v1 results intact.
No new game capture is needed for training.

## Recovery for the failed v6 gameplay capture

The initial v6 pilot accidentally launched Unity with `-nographics`. All 48
trials failed before capturing a shot. The fix keeps graphics enabled on the
virtual display; it does not change the CNN, world models or planner decisions.
See `issue-70-render-capture-repair.md` for evidence and the disclosed retry plan.

For this existing failed run, **do not retrain or delete its results**. Run:

```bash
python -u -m scripts.run_issue_70_parser_repair \
  --repair-pilot --device cuda --start-display \
  2>&1 | tee -a data/issue-70-pilot-v6-rendered.log
```

This repeats all frozen trials once under `experiment/pilot-rendered-v1/`,
preserving the old failures. Its gallery is `data/issue-70-pilot-audit-v6-rendered/`.
After completion and video review:

```bash
python -u -m scripts.run_issue_70_parser_repair \
  --publish 2>&1 | tee -a data/issue-70-publish-v6-rendered.log
python -u -m scripts.run_issue_70_parser_repair \
  --validate 2>&1 | tee -a data/issue-70-validate-v6-rendered.log
```

Once the repair plan exists, those commands automatically use the corrected
pilot and `data/runtime_evidence/issue-70/action-design-v6-rendered.json`, which
discloses the original failures and additional technical attempt. A new zero-shot
capture failure pauses the pilot immediately. Inspect it before resuming; cached
failures stay in the accounting and are not automatically retried. An all-capture-
failure run cannot be published as a gameplay comparison.

The bounded live check is `--smoke-pilot --device cuda --start-display`. It records
the first frozen level with fixed prior and corrected CEM in a separate diagnostic
directory, without counting those shots as pilot results. Its review gallery is
`data/issue-70-pilot-rendering-smoke-v1/index.html`.

## Original v6 training workflow

V6 replaces fixed RGB/Sobel features and the flattened-image MLP with a small
learned RGB CNN: three 3x3 convolutions (16/32/64 channels, strides 1/2/2), GroupNorm
and LeakyReLU, adaptive pooling to 4x6, then projection to 128 features. Existing
slot fusion and prediction heads remain. RGB is scaled to [-1,1]; there are no
pretrained weights, per-pixel fitted statistics, new dependencies or downloads.
All CNN and head weights train from scratch under one fixed schedule.

The 96x64 input images, 18 slots, 236-value downstream interface, training-only
balanced losses, candidate inventories and objective gate are unchanged. The
CNN has a separate architecture/parser/carrier identity. Old MLP checkpoints
cannot be loaded as CNN checkpoints or reused as CNN-carrier world models.

V6 reads the v5 data manifest, which points to the original v3 shards. Keep the
prior artifact directories: no data is copied or recollected, and no new hashes
are introduced. Previous evidence remains untouched. The historical comparison
against v5 is exploratory, not a matched causal backbone ablation: the training
schedule and initialization differ. The plan is in `issue-70-cnn-backbone-plan.md`.

## Start the repair

Activate the `novphy` conda environment and source `env.sh`, then run:

```bash
python -u -m scripts.run_issue_70_parser_repair \
  --run-perception --device cuda \
  2>&1 | tee -a data/issue-70-parser-repair-v6.log
```

This trains and audits perception only. It stops after Stage A even if the audit
passes: no carrier rebuild, world-model retraining or gameplay starts. It reports
parameter count, stages, epoch/frame/loss progress, estimated training time
remaining, per-class diagnostics and endpoint progress. Rerunning resumes cached
data and completed parser epochs.
An interrupted parser epoch repeats that epoch using its fixed shuffle seed.
Initial data preparation is CPU/disk work; CUDA is used during training and
parser/model inference, so an initially idle GPU is expected.

Each epoch logs image sensitivity on the first/last sampled frame of 16 uniformly
spaced training lineages. A nonfinite or <=1e-5 maximum presence-probability range
stops training before that epoch is checkpointed. Pig/block count ranges are also
reported. This is a collapse detector, not an accuracy or generalization claim.
The existing calibration objective gate is unchanged and still controls whether
world-model training may begin. A sensitivity-check exception needs investigation;
it is not a completed negative calibration audit to publish.

Each epoch also reports separate pig/block present and absent accuracy on all
prepared frames of 128 uniformly spaced training lineages, plus count MAE. This
is training diagnostics, not independent evaluation. Calibration prints per-true-
count prediction/MAE groups and writes `experiment/parser-count-diagnostics.json`.
`experiment/backbone-comparison.json` records the CNN and preserved v5 rankings.

Only after `perception complete passed=True`, explicitly start the expensive
downstream workflow if you want to proceed:

```bash
python -u -m scripts.run_issue_70_parser_repair \
  --run-repair --device cuda \
  2>&1 | tee -a data/issue-70-downstream-v6.log
```

It reuses the completed CNN/audit and then rebuilds carriers and trains the nine
matched world models. This can take much longer than the perception check.

The stages are:

1. Freeze the repair plan against the accepted #62 training release.
2. Prepare bounded parser-training shards from all 3,000 training lineages.
3. Freeze training-only class weights, then train the CNN from scratch for 20 epochs.
4. Prepare a new #70 rerun bound to that parser and nine future model checkpoints.
5. Parse the existing #68 calibration endpoints into a separate cache.
6. Apply the unchanged objective audit. **Stop if it fails, or if using --run-perception.**
7. Rebuild complete h15 training carriers with the repaired parser.
8. Train the three-seed teacher-forced/u2/u4 models on those carriers.
9. Score the matched action inventory with all nine models and ensembles.
10. Freeze the conditional gameplay pilot.

Only if the final log reports `pilot_allowed=True`, run:

```bash
python -u -m scripts.run_issue_70_parser_repair \
  --run-pilot --device cuda --start-display \
  2>&1 | tee -a data/issue-70-pilot-v6.log
```

Then publish and validate:

```bash
python -u -m scripts.run_issue_70_parser_repair \
  --publish 2>&1 | tee -a data/issue-70-publish-v6.log

python -u -m scripts.run_issue_70_parser_repair \
  --validate 2>&1 | tee -a data/issue-70-validate-v6.log
```

Publication also supports negative gates. A failed parser audit publishes
`parser_not_ready` without requiring world-model training. A failed ranking gate
publishes `not_ready_for_pilot` without gameplay. Neither authorizes #64.
An affirmative perception audit alone is not completion of the whole experiment;
do not publish/validate a positive result before the required downstream stages.

## Training and isolation

Parser inputs are agent RGB images from the existing complete #62 training
lineages. Each shot contributes up to eight uniformly spaced frames, including
its first and final frame. Sampling does not inspect outcomes. Targets use the
synchronized engine presence and projected centers, fixed object kinds, and
available contact/support/stability labels. Unavailable relations/macros are
masked, not treated as false. Every training shot and lineage is included.

The frozen design uses 64×96 images, hidden width 128, seed 7002001, batch size
128, AdamW learning rate 0.001, and 20 from-scratch epochs. The loss combines presence BCE
(weight 4), pig/block count MSE, center regression, kind classification, and
available relation/macro BCE. Those other loss terms have weight one. V6 retains
`4 * balanced_pig_presence_BCE + 4 * balanced_block_presence_BCE`. Each authored
slot's class weights are `N / (2 * N_class)` when both classes occur; a slot with
only one observed class keeps unit weight. Weights are frozen in `parser-loss.json`
from all training shards, not recomputed per minibatch. Every training frame is
still consumed once per epoch; no outcome-conditioned collection/replacement or
oversampling is introduced. This changes the supervised loss and can affect
probability calibration, so the original objective audit remains essential. Probability
temperatures are fixed at one and classification thresholds at 0.5. Calibration
does not fit parser parameters or choose a favorable parser epoch.

Training reads small tensor shards in bounded groups, rather than loading all
images into RAM. Epoch checkpoints retain optimizer state for resume. Already
validated source JSON and the selected PNGs are read directly; no new hashes or
repeated full-image integrity scans are introduced.

The repaired carrier has an explicit v6 identity. Rebuilding uses every shot's
complete h15 grid and each target's immediately preceding real frame for motion.
It does not duplicate h1 windows, which are not used by this experiment. This
keeps the full 3,000-lineage training pool while reducing memory usage.

The nine world models reuse #67's architecture, seeds and 8-million-example
budget per cell. All are trained on the repaired carrier. The baseline is now
the matched, newly trained teacher-forced model, not an old #63 model receiving
incompatible new-parser inputs. The live reference system is labeled
`teacher_forced_cem`; corrected CEM, corrected grid and fixed prior remain.

## Locations and individual stages

All new private artifacts are under
`.local-artifacts/issue-70-parser-repair-v6/`. The rerun cache and scores are in
its `experiment/` subdirectory. Videos go to `data/issue-70-pilot-audit-v6/` and
the public report to `data/runtime_evidence/issue-70/action-design-v6.json`.

Individual modes are available for troubleshooting:

```text
--prepare
--prepare-parser-data
--train-parser
--prepare-rerun
--prepare-endpoints
--audit-objective
--rebuild-carriers
--train-world-models
--score-models
--freeze-pilot
```

`--dry-run` writes no files and exercises parser learning, nine miniature h15
training cells and the ranking/planner paths. After preparation, `--smoke-test
--device cuda` uses one real training lineage to verify label preparation,
parser gradient updates, full h15 carrier derivation and a miniature world-model
training run, without saving weights or data. Neither is evidence that the
retrained parser has passed its full calibration audit.

Exact validation checks parser epoch/frame accounting, checkpoint bindings,
complete training lineage membership, nine world-model reports, then the #70
rankings and any required captured pilot evidence. It does not rerun inference
or hash model files. Results remain exploratory: the already opened old
model-selection data cannot confirm this repair, and no final access is opened.
