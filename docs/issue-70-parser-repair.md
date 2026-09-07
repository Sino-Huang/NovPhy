# Issue 70: rerun with a trained, slot-conditioned parser

The repair workflow now covers data preparation, real parser training, parser
validation, regenerated carriers, matched world-model retraining, and the #70
ranking/gameplay rerun. It keeps the old parser, checkpoints and v1 results intact.
No new game capture is needed for training.

## Start the repair

Activate the `novphy` conda environment and source `env.sh`, then run:

```bash
python -u -m scripts.run_issue_70_parser_repair \
  --run-repair --device cuda \
  2>&1 | tee -a data/issue-70-parser-repair-v2.log
```

This is the long command. It is deliberately left to the operator. It reports
stage numbers, per-lineage progress and ETA, parser epoch/frame/loss progress,
world-model example budgets, and per-model/state/candidate scoring progress.
It does not launch gameplay automatically. Rerunning the command resumes cached
data, completed parser epochs, complete carrier lineages and trained model cells.
An interrupted parser epoch repeats that epoch using its fixed shuffle seed.
Initial data preparation is CPU/disk work; CUDA is used during training and
parser/model inference, so an initially idle GPU is expected.

The stages are:

1. Freeze the repair plan against the accepted #62 training release.
2. Prepare bounded parser-training shards from all 3,000 training lineages.
3. Train the v2 parser for 20 epochs and save a separately versioned checkpoint.
4. Prepare a new #70 rerun bound to that parser and nine future model checkpoints.
5. Parse the existing #68 calibration endpoints into a separate cache.
6. Apply the unchanged objective audit. **Stop here if it fails.**
7. Rebuild complete h15 training carriers with the repaired parser.
8. Train the three-seed teacher-forced/u2/u4 models on those carriers.
9. Score the matched action inventory with all nine models and ensembles.
10. Freeze the conditional gameplay pilot.

Only if the final log reports `pilot_allowed=True`, run:

```bash
python -u -m scripts.run_issue_70_parser_repair \
  --run-pilot --device cuda --start-display \
  2>&1 | tee -a data/issue-70-pilot-v2.log
```

Then publish and validate:

```bash
python -u -m scripts.run_issue_70_parser_repair \
  --publish 2>&1 | tee -a data/issue-70-publish-v2.log

python -u -m scripts.run_issue_70_parser_repair \
  --validate 2>&1 | tee -a data/issue-70-validate-v2.log
```

Publication also supports negative gates. A failed parser audit publishes
`parser_not_ready` without requiring world-model training. A failed ranking gate
publishes `not_ready_for_pilot` without gameplay. Neither authorizes #64.

## Training and isolation

Parser inputs are agent RGB images from the existing complete #62 training
lineages. Each shot contributes up to eight uniformly spaced frames, including
its first and final frame. Sampling does not inspect outcomes. Targets use the
synchronized engine presence and projected centers, fixed object kinds, and
available contact/support/stability labels. Unavailable relations/macros are
masked, not treated as false. Every training shot and lineage is included.

The frozen design uses 64×96 images, hidden width 128, seed 7002001, batch size
128, AdamW learning rate 0.001, and 20 epochs. The loss combines presence BCE
(weight 4), pig/block count MSE, center regression, kind classification, and
available relation/macro BCE. All other loss terms have weight one. Probability
temperatures are fixed at one and classification thresholds at 0.5. Calibration
does not fit parser parameters or choose a favorable parser epoch.

Training reads small tensor shards in bounded groups, rather than loading all
images into RAM. Epoch checkpoints retain optimizer state for resume. Already
validated source JSON and the selected PNGs are read directly; no new hashes or
repeated full-image integrity scans are introduced.

The repaired carrier has an explicit v2 identity. Rebuilding uses every shot's
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
`.local-artifacts/issue-70-parser-repair-v2/`. The rerun cache and scores are in
its `experiment/` subdirectory. Videos go to `data/issue-70-pilot-audit-v2/` and
the public report to `data/runtime_evidence/issue-70/action-design-v2.json`.

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
