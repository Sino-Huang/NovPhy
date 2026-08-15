# Todo 8 source gate report

Date: 2026-08-09

## Verdict

- codeQualityStatus: WATCH
- recommendation: REQUEST_CHANGES
- ready for atomic Todo 8 commit: No. Address the two MEDIUM findings below and rerun the focused gate. No CRITICAL or HIGH correctness failure was found.

## Scope and skill perspectives

Reviewed the uncommitted Todo 8 Python/source diff only:

- `world_model/training/grid_run.py`
- `world_model/training/real_data.py`
- `world_model/training/scoring_artifacts.py`
- `world_model/training/scoring_torch.py`
- `tests/test_world_model_grid_run.py`
- `tests/test_world_model_scoring.py`

The `omo:remove-ai-slops` and `omo:programming` SKILL.md files were read before evaluating maintainability and tests. The diff does not add tautological/deletion-only/prompt-prose tests, untyped escapes, broad exception handling, or needless input parsing/normalization. It does add a single-purpose adapter seam that is used by the real scorer, so it is not needless abstraction. The two findings below are a missing behavioral regression and an avoidable unbounded intermediate allocation, respectively.

## Findings

### CRITICAL

None.

### HIGH

None.

### MEDIUM

1. `world_model/training/scoring_torch.py:72-105` is not fully batch-bounded despite its class contract. Inference tensors are sliced to `batch_size` at lines 91-95, but lines 82-89 first stack context and target tensors for every nonterminal state of a shot. A valid catalog has no maximum shot-frame count at this boundary, so a long shot can allocate two full per-state image tensors before the bounded inference loop. Build each CPU context/target stack inside the `offset` loop, or introduce and validate an explicit catalog shot bound. This is a runtime-resource/regression risk, not a demonstrated numerical mismatch.

2. `world_model/training/scoring_torch.py:72-119` has no committed test exercising `TorchCatalogPredictor` or `score_real_checkpoint`. The new tests cover CUDA checkpoint loading (`tests/test_world_model_grid_run.py:193-220`) and interleaved artifact-manifest ordering (`tests/test_world_model_scoring.py:210-225`), but neither detects a changed target-clamp calculation, cache key, shot grouping, nor the batch bound above. Add a temporary PNG catalog regression that invokes the real predictor with terminal-clamped states and instruments `encode`/`encode_target` batch sizes. Assert numerical equivalence to the prior per-example contract within an explicit floating-point tolerance and max batch size.

### LOW

None.

## Verified behavior

- `RealPhaseData.build` still derives target-aware delta-15 motion from `min(context + 15, terminal)` and calibrates P50/P90 from calibration-only states. Existing real adapter test passed and compares every resulting regime assignment to the original state-by-state projection.
- The scorer cache uses the requested delta and the same terminal clamp as the effective-delta contract. A fresh synthetic 17-frame probe compared every requested/effective regime batch against a direct per-example reference: `semantic_match=True`, maximum absolute difference `1.043081283569336e-07`, and observed inference batch maximum `4` with configured `batch_size=4`.
- The manifest digest repair iterates partitions in the same canonical order used by the shard writer/validator (`world_model/training/scoring_artifacts.py:132-139`). The new interleaved-input artifact test passed.
- CUDA is available here. The added CUDA checkpoint round-trip test passed, exercising `torch.set_rng_state(payload["torch_rng"].cpu())` at `world_model/training/grid_run.py:258`.

## Commands and results

- `python -m unittest -v tests.test_world_model_scoring tests.test_world_model_grid_run tests.test_world_model_real_data tests.test_world_model_grid_data` -> PASS, 33 tests in 1.526s.
- Fresh synthetic `TorchCatalogPredictor` probe (17-frame, terminal-clamped states, `batch_size=4`) -> PASS; semantic result above.
- `python -m py_compile world_model/training/grid_run.py world_model/training/real_data.py world_model/training/scoring_artifacts.py world_model/training/scoring_torch.py tests/test_world_model_grid_run.py tests/test_world_model_scoring.py` -> PASS.
- `git diff --check` -> PASS.
- `uv run --with ruff ruff check` on all six scoped files -> reports 7 existing violations. Re-running Ruff against `HEAD` content reports the same 7 violations; `real_data.py` is clean. This gate is not newly broken by Todo 8, but the repository remains lint-nonclean in these files.
- `basedpyright` was unavailable.

## Adversarial classes

- Stale artifacts: PASS. Existing resume/checkpoint/spec/shard rejection coverage ran; the new interleaved partition case validates the repaired canonical manifest digest.
- Dirty worktree: PASS as an audit condition. The worktree contains concurrent `.omo`, `.claude`, and player artifacts; none were modified. The proposed atomic commit must include only the six scoped Todo 8 files after the findings are repaired.
- Misleading success: DETECTED and avoided. An initial shell wrapper printed green test output but exited nonzero because zsh reserves the variable name `status`; the focused suite was rerun directly and exited 0. This report relies on the direct exit-0 run.
- Interruptions/hung runs: PARTIALLY MITIGATED. The prior decode storm is addressed by shot-level reuse and the fresh probe completed quickly, but full-shot CPU materialization remains unbounded (MEDIUM 1); no long-shot adversarial regression exists.

