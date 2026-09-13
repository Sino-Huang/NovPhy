# Fixed-policy development evaluator: synthetic preflight

No development predictions have been scored under this protocol. The metadata
inventory and synthetic preflight both retain `score_execution_authorized=false`.
All previous failed training qualifications and the unmet #76 gameplay and
advancement requirements remain unchanged.

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

## Remaining before development scoring

Finish the persistent role-execution driver and complete deduplicated parent
cost ledger, then freeze the full source-bound execution plan. Evaluate the
entire calibration matrix and freeze its common choices before opening new
model-selection predictions. Validate the complete output inventories,
numerical results, access ordering and descriptive paired uncertainty.

Passing this implementation smoke establishes neither development transfer nor
adaptive gameplay competence. It does not authorize fresh evaluation or #64/#65.
