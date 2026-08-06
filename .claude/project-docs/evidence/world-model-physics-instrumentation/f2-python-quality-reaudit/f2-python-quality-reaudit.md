# F2 Python Quality Reaudit

Verdict: **BLOCK / REQUEST_CHANGES**

Scope reviewed: `scripts/collect_rollouts.py`, `scripts/rollout_artifacts.py`,
`scripts/physics_rollout_persistence.py`, and `tests/test_collect_rollouts.py`.
This is a read-only audit of product and test code. The worktree was already dirty;
only this evidence directory was created.

## CRITICAL

### F2-PY-001: stale `.tmp` symlink escapes the output root and then crashes resume

`cleanup_incomplete_physics_attempts` intentionally skips symlinks at
`scripts/collect_rollouts.py:940-947`, but `collect_rollouts` later accepts the
same path with `shot_dir.mkdir(..., exist_ok=True)` at line 1547. For a directory
symlink, subsequent persistence writes follow the link. Validation eventually
returns an invalid result, after which line 1707 calls `shutil.rmtree` on the
symlink and raises `OSError`.

Fresh reproduction: [manual-fixture-verdict.log](manual-fixture-verdict.log)
shows `cleanup_removed=[]`, `tmp_symlink_retained=true`, an external target
changing from `[]` to `frames`, `metadata.json`, `physics_events.jsonl`, and
`physics_state.jsonl`, then `Cannot call rmtree on a symbolic link`.

Required correction: reject/remove the stale symlink as a distinct unsafe state
before any capture directory use, and make recovery fail closed without following
or recursively deleting external targets. Add a behavioral regression that proves
the external target remains empty and the collector returns a controlled error.

## HIGH

### F2-PY-002: direct physics collection prints success for an episode its consumer rejects

For `physics_capture_v1`, `collect_rollouts` writes
`capture_source="scripts.collect_rollouts"` and
`replay_mode="same-episode-varied-trials"` at
`scripts/collect_rollouts.py:1721-1740`. The consumer at
`scripts/rollout_artifacts.py:211-212` requires
`capture_source="capture_physics_rollout"` and
`replay_mode="fresh-engine-per-rollout"`.

Fresh reproduction: [manual-fixture-final2.log](manual-fixture-final2.log)
creates an accepted capture and a valid-final resume collision, then gets
`invalid_episode_contract` from `validate_rollout_episode`. The public caller can
therefore emit a manifest and `rollout_count: 1` that cannot enter the validation
pipeline.

Required correction: either make the direct collector write a contract-valid
physics episode or prohibit that mode before capture. Add a test that invokes the
real consumer on every reported-success physics manifest.

## MEDIUM

### F2-PY-003: implementation-mirroring tests add unnecessary maintenance burden

`tests/test_collect_rollouts.py:341-383` asserts that validation does not call
`Path.read_bytes` or `Path.read_text`. These tests pin an implementation detail,
not an observable contract, and would fail under behavior-preserving refactors.
The boundary contract is already represented by the malicious oversized-sidecar
test at lines 323-339. This is a `remove-ai-slops` overfit finding.

### F2-PY-004: collector remains an oversized mixed-responsibility module

`scripts/collect_rollouts.py` measures 1950 pure LOC and now contains persistence,
recovery, validation dispatch, promotion, and legacy desktop collection concerns.
The consulted `remove-ai-slops` perspective flags source files over 250 pure LOC;
the `programming` perspective also rejects needless untyped growth. This is not
the source of the two runtime failures, but it makes them more likely and harder
to test in isolation.

## LOW

No additional low-severity findings.

## Verification

Ran:

```
python -m unittest -v tests.test_collect_rollouts.PhysicsCapturePersistenceTests
python -m py_compile scripts/collect_rollouts.py scripts/rollout_artifacts.py scripts/physics_rollout_persistence.py
git diff --check -- scripts/collect_rollouts.py scripts/rollout_artifacts.py scripts/physics_rollout_persistence.py tests/test_collect_rollouts.py
python - <<'PY'  # TemporaryDirectory adversarial fixture; full invocation in manual-fixture-final2.log
...
PY
```

The focused suite passed 22 tests. It covers normal final-shot reuse, malformed
request-70 data, sidecar bounds, ordinary stale-directory cleanup, symlinked
sidecars, and validator swap races. It does **not** cover a stale tmp *directory
symlink* or consumer acceptance of an emitted physics manifest; both gaps produced
the findings above.

Skill-perspective check: ran. `remove-ai-slops` finds the brittle
implementation-mirroring tests and oversized collector; `programming` finds the
same raw-dict/untyped expansion and broad legacy module pattern. No prompt tests
were present. Prompt injection is N/A because all inputs were local source and
deterministic filesystem fixture data.

Cleanup receipt: [cleanup-receipt.json](cleanup-receipt.json). All fixture roots
used `TemporaryDirectory`; the protected-root assertion deliberately failed for
F2-PY-001 and is recorded in [manual-fixture-verdict.log](manual-fixture-verdict.log).
