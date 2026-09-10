# #70: v5 task-presence fine-tuning

## Diagnosis

The v4 negative audit is reproduced and preserved: count-discriminating regret
0.740728, top-3 0.617284, and zero all-tied states. The image-blind collapse is
resolved, but task-count accuracy remains insufficient. The audit gate is correct.

On calibration endpoints, actual zero pigs (22 candidates) gave mean predicted
count 0.69096; actual one pig (2,378 candidates) gave 0.84841. As a diagnostic
ablation only, replacing predicted pig counts with truth while retaining predicted
blocks reduces regret to 0.17284 and raises top-3 to 0.88889. These oracle values
are never used by the objective, candidate selection, or planner. Block-count and
generalization errors also remain; fixing pig imbalance cannot guarantee a pass.

An inference pass over all 43,752 existing training frames found only 1,556
pig-absent examples (3.6%). The parser predicts mean 0.5023 pigs on these training
examples, versus 0.9769 when pigs are present. Thus the weakness exists even on
training data; it is not solely a calibration-distribution issue. Sampled raw
engine targets and endpoint PNGs agree on pig absence (destroyed, no body).

## Implemented repair

- V5 initializes from the completed v4 training checkpoint, retaining architecture,
  18 slots, 236-value carriers and fitted input-normalization buffers. It uses a
  fresh optimizer, seed 7002001, learning rate 0.0001, and 10 fine-tuning epochs.
- Add `4 * balanced_pig_presence_BCE + 4 * balanced_block_presence_BCE` to the
  existing losses. Each slot's positive/negative class receives total equal
  weight using `N/(2*N_class)`, computed once from the training shards and frozen
  in `parser-loss.json`. If only one class occurs, it keeps unit weight.
- No changed dataset membership, oversampling, outcome-conditioned replacement,
  new captures, score rounding, oracle scoring, or calibration-fitted parameters.
  Every original training frame is consumed once per epoch.
- Log separate present/absent accuracy, mean predicted presence and count MAE on
  128 uniformly spaced training lineages each epoch. These are training diagnostics,
  explicitly not an independent test. Retain the prior image-sensitivity guard.
- Calibration logs now include prediction/MAE grouped by actual pig/block count,
  saved in `experiment/parser-count-diagnostics.json`. The objective, regret/top-3
  gates, candidate inventory and final-evaluation restrictions are unchanged.
- V5 has separate artifacts and parser/carrier identities. The v4 data manifest
  is reused read-only; it references v3 shards. Keep both v3 and v4 directories.
  No new hashes or image-corpus integrity scans are introduced.

Reweighting can change probability calibration. It is a targeted supervised
training repair, not a claim of perfectly calibrated expected counts. The
unchanged calibration audit must still judge whether this objective is usable.

## Verification

A bounded three-epoch training-only fine-tune increased pig-absence accuracy on
the fixed diagnostic sample from 55.1% to 89.9%, while present accuracy was 97.3%
(previously 99.1%). Mean predicted presence on absent pigs fell from 0.4754 to
0.1429. This is evidence of improved minority-class learning, not calibration
success. No new calibration evaluation was used to select fine-tuning settings.

78 focused tests passed, including equal total rare/common class gradients,
training-only class-weight caching, source-bound initialization with normalization
preservation, checkpoint/resume, and existing parser/planning tests. The no-write
dry run and real CUDA parser/carrier/world-model smoke passed. A temporary real
one-epoch run verified fine-tuning, checkpoint reload, exact training-accounting
validation and resume without retraining. Temporary weights were removed.

The v5 plan and reused data manifest are prepared. Full fine-tuning, calibration
and any later world-model training remain operator work. #70 stays open.

## Commands

```bash
python -u -m scripts.run_issue_70_parser_repair \
  --run-repair --device cuda \
  2>&1 | tee -a data/issue-70-parser-repair-v5.log
```

This includes class-inventory progress, epoch/frame/loss and class diagnostics,
then endpoint/audit progress. Only after `pilot_allowed=True`:

```bash
python -u -m scripts.run_issue_70_parser_repair \
  --run-pilot --device cuda --start-display \
  2>&1 | tee -a data/issue-70-pilot-v5.log
```

Review `data/issue-70-pilot-audit-v5/`, then publish and validate:

```bash
python -u -m scripts.run_issue_70_parser_repair \
  --publish 2>&1 | tee -a data/issue-70-publish-v5.log
python -u -m scripts.run_issue_70_parser_repair \
  --validate 2>&1 | tee -a data/issue-70-validate-v5.log
```

A completed negative audit/ranking gate skips gameplay and can still be published.
An exception instead needs investigation before continuing. No #64 authorization
or final access follows from this exploratory repair.
