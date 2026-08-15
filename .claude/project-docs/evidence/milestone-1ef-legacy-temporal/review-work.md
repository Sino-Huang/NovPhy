# F1-F4 and Review-Work Consolidation

Reviewed commit: `95e532e164d9d49a0cecdb9514b9abca8be1e24a`
Diff base: `55d01fd38519f00bf27bda673d03e2668bc184ab`
Overall verdict: **FAIL / REQUEST_CHANGES**

The review wave is complete. The committed scientific evidence is internally
honest and the existing repaired artifacts validate, but the exact reviewed
source cannot regenerate those artifacts end to end and the final plan's
reproducibility acceptance thresholds were not met.

## F1-F4

| Gate | Verdict | Decisive evidence |
| --- | --- | --- |
| F1 plan compliance | FAIL | Aggregate metrics exceed `rtol <= 1e-2`; best-pair agreement is `0.9695291753971118 < 0.99`. The miss is recorded honestly, but the plan explicitly requires both thresholds to pass. |
| F2 code quality | FAIL | `write_frontier_input` emits legacy `{"states": ...}` bytes while `canonical_frontier_rows` requires the seven-field `temporal_frontier_input_v1` schema. |
| F3 real manual QA | PASS | Both 556,959-state/1,670,877-score trees validate; the repaired canonical 460-byte primary input renders nonblank JSON/Markdown/SVG/PDF; one raw label and one regime aggregate recompute correctly. |
| F4 scope fidelity | PASS | Protected roots/player remain unchanged; runs/checkpoints remain ignored; continuous-only scope and `symbolic_supervision_unavailable` exclusions are explicit. |

F3 validates the committed/ignored repaired artifact. It does not clear F2:
fresh end-to-end CLI QA regenerated a 312,762,045-byte legacy input and then
failed with exit 2, `frontier input must use the closed canonical schema`.

## Review-Work Lanes

| Lane | Verdict | Summary |
| --- | --- | --- |
| Goal and constraints | PASS | Requested runs, counts, provenance, unavailable metrics, protected-root receipts, and honest negative scientific outcome are present. |
| Code quality | FAIL | Real frontier producer/consumer integration is broken. Bootstrap membership also includes the derived global frontier instead of regime frontiers only. |
| Security | FAIL | `torch.load(..., weights_only=False)` permits checkpoint pickle execution before payload checks. |
| Runtime QA | FAIL | Isolated validation/rendering passes, but the real end-to-end frontier command fails at the schema boundary. |
| Context mining | FAIL | Independently confirms the reproducibility miss. Its separate archived-`todo:8` identity concern is unrelated project history and is not carried as a blocker here. |

Review-work requires all lanes to pass, so its aggregate verdict is **FAIL**.

## Blocking Findings

1. **HIGH - real frontier generation is not integrated.**
   `world_model/training/real_data.py:267-280` writes an incompatible,
   unbounded materialization. `world_model/training/frontier.py:189-225`
   requires digest-bound provenance. The authoritative artifact succeeded only
   after the repair recorded in `artifact-repair.log`.

2. **HIGH - checkpoint deserialization crosses the trust boundary unsafely.**
   `world_model/training/grid_run.py:206-263` calls unrestricted
   `torch.load` before schema/config validation. A crafted checkpoint can run a
   pickle reducer before rejection; a digest calculated from the same supplied
   file is not authentication.

3. **PLAN ACCEPTANCE - reproducibility thresholds failed.**
   `acceptance-manifest.json` records
   `aggregate_metrics_within_rtol_1e-2=false`, best-pair agreement
   `0.9695291753971118`, and `overall_pass=false`. Thresholds were not weakened.

## Nonblocking Follow-Up Findings

- `world_model/training/frontier.py:164-173` includes `global` in bootstrap
  frontier membership. Regime-intersection verdict behavior is redundant under
  the fixed compute-cost ordering, but reported membership frequency can be
  inflated by global-only membership and needs a regression.
- Resume validation does not fully bind shard geometry/topology before reuse.
- Per-shot preprocessing retains tensors proportional to the largest shot,
  although decode/model call batches are bounded.
- Generated SVG/evidence whitespace makes `git diff --check` nonzero.
- Fixture `all` is a train/score/validate smoke with only 18 states; absence of
  a frontier plot is not a blocker because the frontier contract requires at
  least 100 states per regime.

## Reports

- `f1-plan-compliance.md`
- `f2-code-quality.md`
- `f3-manual-qa.md`
- `f4-scope-fidelity.md`
- `.omo/evidence/review-work-goal.md`
- `.omo/evidence/review-work-code.md`
- `.omo/evidence/review-work-security.md`
- `.omo/evidence/review-work-qa.md`
- `.omo/evidence/todo8-context-review.md`

No production fix, threshold change, GPU rerun, protected-root write, or commit
was performed during this review-only wave.
