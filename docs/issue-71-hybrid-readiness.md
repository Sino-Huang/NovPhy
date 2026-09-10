# #71: CNN-compatible hybrid readiness

This is an implementation/training readiness check, not a positive experiment
result. #72 remains blocked until the operator's full run is inspected and its
readiness publication validates. #64/#65 are not authorized.

## Audit and minimal change

The selected #70 corrected checkpoint has a 236-value, 18-slot CNN carrier.
Direct tensor comparison against its seeded initialization confirms that
`micro_adapter`, `macro_adapter`, `micro_head`, and `macro_head` never changed.
Its short-unroll recipe trained continuous h15 only. The #15 hybrid used a
different, 197-value, 15-slot oracle-symbol carrier. Neither is a trained
CNN-compatible adaptive hybrid checkpoint.

The old gameplay wrapper also selects one pair and builds a symbolic request
from the initial observation before its rollout loop. Reusing that request
does not refresh symbols as the predicted carrier changes. Historical code and
evidence are left intact; the new runtime does not use that wrapper.

Reuse the frozen #70 CNN, its carrier codec, accepted #62 training RGB/labels,
and the existing `PairConditioner` / `FiLMBlock` shared-predictor implementation.
No new game collection, encoder training, CEM change, or final data access.

Necessary new components:

- A fixed-slot ordered soft-predicate interface for contact/directed support
  and the two macro state predicates. Small learned carrier readouts replace
  unavailable legacy weights. Availability is supervised with masks; unavailable
  labels are not negatives. Contacts are symmetric; support remains directed.
- Selected-mode inputs are decoded from the current continuous carrier on
  **every** imagined transition. Symbolic tensors are temporary conditioning
  inputs, never a second persistent rollout state. Engine labels enter loss and
  training diagnostics only. No untrained event/duration head is instantiated;
  this is the declared two-predicate macro-state interface, not evidence of
  predicting macro events or validating skipped paths.
- A small joint controller receives only current carrier, interface action,
  and remaining prediction time (lookahead feature capped at 60 steps). Its
  selection executes exactly one of nine `(1,5,15) × (continuous,micro,macro)`
  paths. It cannot request a step beyond the common endpoint. The fixed control
  uses the **same corrected checkpoint**, continuous h15, with no controller call.

This soft ordered interface is not the old capacity-matched #15/#58 graph
adapter or its reliability gate. It is a new, explicitly identified lightweight
CNN-compatible candidate. It cannot rescue or reinterpret the older hypothesis
tests. Neither joint nonseparability nor useful adaptation is established here.

## Frozen, matched training recipe

Use the existing 3,000 training-role lineages only. One-based inventory indices
divisible by five (600 lineages) are excluded from predictor fitting and used
for controller supervision. Both predictors use the other 2,400. The CNN already
used all 3,000; these are not independent perception/generalization estimates.

Each shot supplies up to four uniformly positioned, contiguous 60-fixed-step
windows, including its beginning and end. Parse the preceding RGB frame to
construct motion exactly as at deployment. Windows never cross shots. Short
terminal windows carry their actual length; incomplete requested-horizon targets
are masked, not silently relabeled. Shards preserve lineage, shot, and offset.

Two predictors start from the same seed (20260908), with identical capacity,
data, masks, sampling, and optimizer exposure: 9,000 updates, batch 64, AdamW
1e-4, weight decay 1e-4, clip 1. A group of three fitting lineages visits all nine
pairs before advancing. Every fitting lineage visits every pair; each pair has
1,000 updates. Both arms see the same four local transitions and symbolic
supervision. The corrected arm adds recursive four-step MSE and the declared
carrier-bound penalty; the teacher-forced arm does not. Their compute differs
and is disclosed. This is not a comparison to the unmatched old #70 models.

Controller labels use the frozen corrected model on its 600 excluded fitting
lineages. Duration-weighted dynamic programming minimizes endpoint carrier MSE
times duration, plus 0.0001 times executed linear MACs relative to continuous
h15, plus continuation value. Exact ties use the frozen pair ordering. Only
complete trained horizons are eligible; no artificial mode-balancing labels.

Train 1,800 controller updates, generate one bounded closed-loop aggregation
round, then train another 1,800 on pooled initial/aggregation labels. Aggregation
replaces contexts at actually visited predicted states, retains observed
contexts at other positions, and recomputes DP against unchanged aligned true
targets. Those targets never enter the deployed controller or predictor API.
Use the first window of each controller-training shot, selected without outcomes.

The 24-lineage readiness diagnostics are training-role diagnostics only. Report
pair usage, single-mode behavior, masked symbolic confusion counts, finite
225-step execution and same-endpoint recursive errors. No quality threshold is
tuned here. #72 must separately test useful adaptation and scientific advancement.

## Execution, memory, and evidence

```bash
python -u -m scripts.run_issue_71_hybrid_readiness --dry-run

python -u -m scripts.run_issue_71_hybrid_readiness \
  --smoke-test --device cuda

python -u -m scripts.run_issue_71_hybrid_readiness \
  --run --device cuda \
  2>&1 | tee -a data/issue-71-readiness.log

python -u -m scripts.run_issue_71_hybrid_readiness \
  --validate --device cuda \
  2>&1 | tee -a data/issue-71-validate.log
```

Initialize the `novphy` environment and source `env.sh` first, as usual.
No display or Unity process is needed. `--run` prepares data, trains both
predictors, trains/aggregates the controller, and publishes readiness. Individual
`--prepare`, `--train`, `--train-controller`, and `--publish` stages are available.

Preparation logs each lineage/shot and ETA; training logs every 90 updates and
checkpoints optimizer/sampler state. Completed lineages/models/label shards and
controller rounds are reused. An interrupted controller fitting round restarts
that small round; completed rounds are retained. Rerun the same command to resume.

Only one source capture JSON and a 32-image parser batch are loaded during
preparation. Training holds three lineage shards and one minibatch, not the
image corpus. Three bounded real-lineage measurements took 1.2–2.6 seconds per
lineage; tensor shards were about 0.55–1.1 MB. A brief batch-64 CUDA benchmark
took about 0.03 seconds/update. Observed peak process RSS was about 1.8 GiB;
peak allocated CUDA memory was about 85 MiB. These small warm-cache measurements
suggest roughly **1–3 hours overall**, dominated by preparation, with substantial
uncertainty from shot count and storage contention. Allow several GB of shard
storage. The foreground ETA is more informative once the full run starts.

Artifacts: `.local-artifacts/issue-71-hybrid-readiness-v1/`:

- `plan.json`: source/training/role contracts and exact relevant implementation
  text, existing Git revision, and dirty-worktree disclosure. No new hashes or
  image-integrity pass. Historical source IDs are retained unchanged.
- `teacher_forced.pt`, `corrected.pt`, `controller.pt`: trained state and bindings.
- `readiness.json`: machine-readable readiness, parameter-gradient/exposure
  checks, checkpoint identities, pair traces, diagnostics and compute accounting.
- `smoke.json`: isolated tiny real-data check, explicitly not production evidence.

The linear-MAC accounting includes selected readouts/adapters/trunk/controller;
it is not a full FLOP or end-to-end perception cost claim. Wall time is measured
for preparation/training. #72 must calibrate complete deployment cost before a
matched-compute claim. No engine-derived physical-violation metric is invented
for predicted carriers.

`--validate` recomputes deterministic readiness evidence from the trained
checkpoints and compares it exactly with publication. It rejects incompatible
dimensions, unfinished training, missing task gradients, changed frozen source,
and nonfinite/ineffective pair execution. Single-mode controllers and poor finite
predictions are reported honestly, not silently balanced or called superiority.

Current implementation is local and uncommitted, including prerequisite #70
changes. The source snapshot makes the local implementation identifiable but is
not an archived Git release. Do not close #71 or start #72 merely because the
smoke passed; inspect the full run and complete the repository handoff first.

## Verification completed before handoff

- 115 focused tests passed: 19 new readiness tests plus predictor, temporal
  carrier, CNN parser and #70 parser-repair regressions.
- Public no-write dry run passed against the real source artifacts.
- Public CUDA smoke passed on a real captured training shot: 61 RGB frames,
  18 optimizer updates covering all nine pairs, task gradients reaching every
  deployed parameter tensor, controller labeling/distillation/aggregation-label
  generation, exclusive pair execution, and exact checkpoint reload.
- Full-window preparation was separately exercised on three real lineages,
  including a two-shot lineage. No production shards or model fits were started.
- Resume publication from a completed optimizer checkpoint and the file-backed
  controller-label/aggregation/training/reload path are regression tested.
- `git diff --check` passed. No full-suite claim; no commit/push performed.

Readiness and comparison results are still pending the operator's long run.
