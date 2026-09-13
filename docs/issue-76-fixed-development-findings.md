# Fixed-policy development evaluator: synthetic preflight

The complete calibration matrix has now been scored under the separately
frozen execution plan. Independent validation is not complete: its strict CPU
comparison stopped on a small CPU/CUDA numerical discrepancy. No calibration
choice is frozen and no model-selection prediction has been opened. All previous
failed training qualifications and the unmet #76 gameplay and advancement
requirements remain unchanged. The earlier metadata and synthetic preflight
retain their original `score_execution_authorized=false` dispositions.

## Implemented and checked

The evaluator now supports batched, uninterrupted fixed-pair curves at all five
frozen physical endpoints; separate exact observed-context local predictions;
masked engine-target field metrics; unavailable assignments; and cumulative
per-member prediction failures. It does not expose a controller, optimizer,
collector or fresh-data role. An unchanged-carrier reference is explicitly not
a gameplay action prior.

Calibration selection requires the complete ordered model/member matrix and
common endpoint availability. It chooses one recipe/pair per arm across seeds
using equal-family, equal-seed endpoint error, disqualifies prediction failures,
and breaks ties by the frozen recipe/pair order. These are offline prediction
controls, not strongest gameplay policies.

Seventeen tests passed, including independent single-member rollout agreement,
exact first-shot indexing, persistent failures, engine masks/counts, exact work
counters, equal-family selection, retention of an adverse seed, and rejection
of incomplete or incorrectly bound inputs.

## Bounded synthetic CUDA smoke

The source-bound [smoke plan](../data/issue-76-fixed-development/smoke-plan.json)
was written before execution. It used 50 synthetic records, seed 760940001,
batch size 50, and the first paired hybrid/pure checkpoints in the frozen
inventory. All nine hybrid and three pure policies reached all five exact
endpoints with finite predictions. No real development record was loaded.

The [smoke report](../data/issue-76-fixed-development/smoke-report.json) records:

- All 48 saved checkpoint files and six parameter-midpoint instances passed
  source, completion, capacity, finite-state and completed-budget checks.
- Checkpoint validation: 9.704 seconds measured wall time; 645.3 MiB peak RSS.
- Synthetic scoring phase: 2.740 seconds measured wall time; 1,305.8 MiB peak
  RSS and 18.2 MiB peak allocated CUDA memory.
- 57,000 recursive and 3,000 local transitions; 109,224,480,000 linear MACs.
  There were no controller calls, optimizer updates or new captures.

The phase times are conservative wall-time charges, not CPU process time,
CUDA kernel time, or batch-one deployment latency. MACs are not full FLOPs.
An earlier read-only completion check additionally took 8.451 seconds wall time;
it also read no development records. The detailed source checkpoint receipts
are retained at
`.local-artifacts/issue-76-fixed-development-v1/smoke-checkpoints.json`.

## Calibration execution and unresolved verification

The [execution plan](../data/issue-76-fixed-development/execution-plan.json) was
committed and pushed before calibration access. Its ledger preserves 141
canonical #76 cost receipts, including failed/nonselected branches, and excludes
seven exact public receipt mirrors from double counting. These receipts record
86,112.187 seconds of prior job wall-time charges, not kernel hours or full FLOPs;
unmetered earlier engineering and prior-issue work are not represented as zero.

All 54 calibration model instances completed in 67.748 seconds of phase wall
time, retaining 50 assignments and 45 exact endpoints per seed. This is 2,700
assigned model/member rows. Peak RSS was 1,339.0 MiB and peak allocated CUDA
memory was 18.1 MiB. The three preparation phases took 1.084, 1.089 and 1.100
seconds. Original scores are retained without a rerun.

Validation first exposed and corrected a JSON object-key ordering assumption;
see the [technical correction](issue-76-fixed-development-json-correction.md).
All 28 tests now pass. The corrected independent CPU validator completed the
first three cells, then stopped on repaired-representation/pure seed 760930002,
member `issue-76-development-265`, fixed-50 continuous, elapsed native step 6000:

- Stored block-count absolute error: 0.0340266227722168.
- Independent CPU reconstruction: 0.0339810848236084.
- Independent CUDA reconstruction with separate NumPy field formulas:
  0.03402674198150635, within the original tolerance of the stored score.
- Maximum CPU/CUDA carrier-component difference at that batched point:
  0.00013303756713867188.

The strict CPU failure is retained; no numerical tolerance was widened. The
same-backend check establishes the cause at this one point, not validation of
the entire matrix. The cumulative validation budget retains 18.188 seconds,
including the failed attempts. The focused arithmetic diagnostic additionally
took 1.060 seconds wall time. Its compact evidence is in the
[calibration status receipt](../data/issue-76-fixed-development/calibration-status.json).

Resolve the cross-backend numerical verification requirement before freezing
calibration choices. Only then score model selection, validate the full matrix,
and report the predeclared descriptive paired uncertainty. No favorable
offline result can replace the adaptive gameplay/strong-baseline advancement
requirements.

Passing this implementation smoke establishes neither development transfer nor
adaptive gameplay competence. It does not authorize fresh evaluation or #64/#65.
