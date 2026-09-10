# #70: v6 CNN implementation and verification

The deadline-scoped plan was posted before code changes:
https://github.com/Sino-Huang/NovPhy/issues/70#issuecomment-5570973339

Implementation is complete locally. `ConvSlotVisualPredicateParser` uses the
declared learned RGB CNN (16/32/64 channels; strides 1/2/2; GroupNorm/LeakyReLU;
4x6 pooling; 128-feature projection) with the existing slot/predicate heads.
It has 290,702 trainable parameters for the full 18-slot configuration. The old
MLP classes/checkpoints remain unchanged. No pretrained downloads are needed.

The runner now defaults to v6 artifacts, from-scratch CNN training for 20 epochs,
and the existing training-only balanced losses. It reuses the existing tensor
shards. All state/carrier shapes and audit thresholds remain unchanged.

`--run-perception` executes preparation, training, endpoint parsing and Stage A,
then stops even when Stage A passes. It does not rebuild carriers, train the
nine world models, or launch Unity gameplay. The operator must explicitly use
`--run-repair` to continue the expensive downstream stages after a passing audit.
Progress includes architecture/parameter count, epoch/frame/loss, estimated
training time remaining, per-class diagnostics and endpoint progress.

## Verification

- 82 focused tests passed, including learned convolution gradients, interface
  shapes, independent presence learning from images, strict MLP/CNN checkpoint
  incompatibility, read-only data reuse and exposure checks.
- Tests verify both a failed-audit stop and a perception-only stop after a
  passing audit, plus CLI routing and historical comparison reporting.
- Public no-write dry run and real CUDA parser/carrier/world-model smoke passed.
- A temporary real two-epoch run over all 43,752 prepared training frames per
  epoch passed an intentional epoch-boundary interruption, optimizer resume,
  checkpoint reload, training-accounting validation and completed-run resume.
  Temporary weights were removed; the production 20-epoch run was not performed.
- The two-epoch check took about 31 seconds including label inventory and reload
  checks, with peak allocated CUDA memory 494,289,408 bytes. Actual epoch times
  were roughly 11–12 seconds on this machine; these are smoke measurements, not
  a guarantee for the full run or downstream training.

The CNN was still undertrained after two epochs; these tests establish that the
training/inference workflow functions, not improved calibration accuracy. The
full fixed schedule and unchanged audit must determine whether it is useful.
No calibration-result-driven architecture/threshold sweep was performed.

## Run next

The v6 plan and reused data manifest are prepared. Keep prior artifact directories
(the v5 manifest references v3 shards; the v5 audit is the historical reference).

```bash
python -u -m scripts.run_issue_70_parser_repair \
  --run-perception --device cuda \
  2>&1 | tee -a data/issue-70-parser-repair-v6.log
```

Inspect the ending audit and these files under
`.local-artifacts/issue-70-parser-repair-v6/experiment/`:

- `objective-audit.json`
- `parser-count-diagnostics.json`
- `backbone-comparison.json`

The comparison retains the v5 and CNN metrics and explicitly marks itself as
exploratory, not a matched causal backbone ablation (initialization/schedules differ).

Only if perception passes and the operator elects to continue:

```bash
python -u -m scripts.run_issue_70_parser_repair \
  --run-repair --device cuda \
  2>&1 | tee -a data/issue-70-downstream-v6.log
```

The completed parser/endpoint work resumes from cache. Subsequent pilot,
publication and validation commands are in `docs/issue-70-parser-repair.md`.
A failed perception gate still permits negative publication/validation without
downstream training. A passing perception gate alone is not completion of #70.
No final evaluation or #64 authorization is opened; #70 remains open.
