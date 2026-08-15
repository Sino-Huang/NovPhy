# F2 Code Quality Review

Reviewed commit: `95e532e164d9d49a0cecdb9514b9abca8be1e24a`
Diff base: `55d01fd38519f00bf27bda673d03e2668bc184ab`

The required `remove-ai-slops` and `programming` skill perspectives were loaded and applied. The diff has no material deletion-only or tautological tests, and no obvious untyped escape hatch in the reviewed Python additions. The focused tests cover the fixture path and validator behavior, but do not cover the real-catalog frontier integration.

## Commands and results

- `python -m unittest tests.test_world_model_frontier tests.test_world_model_grid_run tests.test_world_model_real_data tests.test_world_model_scoring` -> PASS (31 tests).
- `python -m unittest tests.test_world_model_data tests.test_world_model_grid_artifacts tests.test_world_model_grid_data tests.test_world_model_model tests.test_world_model_pair_grid tests.test_world_model_training` -> PASS (257 tests).
- `python scripts/run_jepa_pair_grid.py all --fixture --device cpu --steps 18 --batch-size 4 --output-dir <temp>` -> PASS (exit 0; 54 states/162 scores).
- `python -m pytest ...` -> unavailable: this environment has no `pytest` module.
- `git diff --check` -> whitespace warnings only in generated SVG/evidence files; no source-code whitespace errors.

## Findings

### CRITICAL

None.

### HIGH

- `world_model/training/real_data.py:267-280` writes `frontier_input.json` as `{"states": rows}` and omits the required checkpoint, score-manifest, score-spec, and state digests. `scripts/plot_jepa_pair_frontier.py:15-18` unconditionally passes this file to `canonical_frontier_rows`, which requires the closed seven-field `temporal_frontier_input_v1` schema and canonical bytes. Consequently the real `frontier` command, and the `all` command's real frontier phase, always fail with `frontier input must use the closed canonical schema` after successfully scoring. This violates the required real exhaustive scoring/frontier flow and has no real-path regression test. **Blocker.**

### MEDIUM

- `world_model/training/scoring_artifacts.py:93-112` resume validation binds only checkpoint and score-spec digests. It does not compare the existing manifest's `shard_size`, state/score counts, catalog/partition identity, or the complete expected shard list before reusing files. A resume with a changed shard size or a manifest containing stale/extra/reordered shard entries can be accepted/re-written rather than failing closed, contrary to the artifact contract's stale/duplicate/reordered resumability requirement. Existing tests cover tampered bytes but not changed shard geometry or stale manifest topology.
- `scripts/plot_jepa_pair_frontier.py:34` catches broad `Exception` around plotting. This is a minor maintainability issue under the programming/remove-ai-slops perspectives; it is at a CLI boundary and re-raises as `FrontierError`, so it is not itself a release blocker.

### LOW

- Generated `frontier.svg` contains extensive trailing whitespace, reported by `git diff --check`; this is evidence artifact noise rather than runtime impact.

## Verdict

`codeQualityStatus: BLOCK`

`recommendation: REQUEST_CHANGES`

Blocker: fix `write_frontier_input` to emit the canonical digest-bound frontier input consumed by `canonical_frontier_rows`, then add and run a real-path integration test (or an equivalent fixture of the real command) proving `frontier` succeeds. Re-run F2 after that fix; resumability topology checks should also be added before approval.
