# Issue 111: deterministic scoring harness

Parent: #108. Shared rules of #85 apply. Infrastructure only: no published record, runner or
disposition changes. Historical runners are hash-bound by their frozen plans
(`runner_sha256_at_freeze`) and stay untouched.

| Piece | Location |
| --- | --- |
| Harness module | `world_model/planning/scoring_harness.py` |
| Regression tests | `tests/test_scoring_harness.py` |

## Contract for new scoring runners (from #112 on)

1. **Deterministic scoring.** Wrap every `--run` scoring phase and every `--validate` in
   `deterministic_scoring()` (cuDNN benchmark off, cuDNN deterministic on, TF32 off for
   convolutions and matmuls). It restores the previous flags on exit and yields
   `SCORING_POLICY`; write that dict into every scoring record. Training may autotune,
   but runs in a separate process (a separate mode invocation) from scoring.
2. **Replication tolerances.** Carrier guards use `carrier_replication(reference, fresh)`:
   non-motion values within `PARSE_TOLERANCE` (1e-3); motion values within
   2 × tolerance / elapsed, with elapsed read from the reference carrier's column 1.
   Endpoint guards use `endpoint_replication(reference_rows, fresh_rows)` on
   `PARSED_QUANTITIES` (pig presence, pig displacement, tie-free cost), never on the
   1000×-weighted count.
3. **Scoring default.** `EndpointCosts.row` emits both costs; `PRIMARY_COST` is the tie-free
   cost (pig presence − 0.1 × pig displacement), `SECONDARY_COST` the count cost.
   `tie_free` also accepts tensors, so a differentiable loss can use the same definition.

## Evidence

- `EndpointCosts.row` equals `run_decision_chain_attribution.Costs.row` (the #99/#100 scorer)
  on random carriers, and `carrier_replication` reproduces the #100 v2 revised-G1 statistics
  on fit shots (carrier column 1 equals the #100 label `elapsed` exactly).
- Regression test: one process scores a parse → 225-step Δ = 1 rollout → endpoint costs
  twice, then runs an autotuned TF32 training warm-up, then scores again; all three score
  sets are bit-identical. With the context manager replaced by a no-op, the test fails;
  in a larger probe of the same pipeline (64-frame parse, 32 candidates) the post-warm-up
  tie-free cost moved by up to 0.078.
