# F2 Persistence Lane Manual QA

Verdict: **confirmed**. The repaired persistence behavior passed the exact three focused tests twice and independent temporary-root probes. No production or canonical project files were changed by this lane; the worktree had pre-existing tracked modifications, whose tracked status remained unchanged.

## surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
| --- | --- | --- | --- | --- | --- |
| f2-persistence-run1 | F2 atomic persistence and recovery | Python unittest | `python -m unittest tests.test_collect_rollouts.PhysicsCapturePersistenceTests.test_second_collection_reuses_valid_completed_shot_without_bridge_calls tests.test_collect_rollouts.PhysicsCapturePersistenceTests.test_fixed_name_state_sidecar_symlink_outside_shot_is_rejected tests.test_collect_rollouts.PhysicsCapturePersistenceTests.test_completed_shot_with_nondirectory_frames_is_quarantined` | PASS, 3/3, exit 0 | `focused-run1` |
| f2-persistence-run2 | F2 atomic persistence and recovery, repeated | Python unittest | same exact invocation as `f2-persistence-run1` | PASS, 3/3, exit 0 | `focused-run2` |
| f2-direct-temporary-root | bounded sidecar memory, root confinement, valid-final idempotent resume, stale cleanup, malformed input, invalid completed-shot quarantine | Python CLI | inline Python probe using `TemporaryDirectory`, `capture_physics_rollout`, `collect_rollouts`, `recover_physics_capture_attempts`, `validate_physics_shot_artifact`, `load_physics_capture` | PASS, all seven probe cases, exit 0 | `direct-temporary-root-probe` |
| f2-dirty-worktree | preserve pre-existing dirty worktree | Git CLI | `git -C /mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4 status --short --branch` before/after; compare tracked entries | PASS, tracked status unchanged | `dirty-worktree-check` |
| f2-diff-hygiene | changed persistence surface remains whitespace-clean | Git CLI | `git -C /mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4 diff --check -- scripts/collect_rollouts.py scripts/physics_capture_parsing.py scripts/rollout_artifacts.py scripts/rollout_validation_types.py tests/test_collect_rollouts.py src/webui/bridge.py` | PASS, exit 0 | `scoped-diff-check` |

## adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
| --- | --- | --- | --- | --- | --- |
| f2-malformed-input | request-70/persistence contract | malformed input | malformed JSON is rejected with a contract error and cannot produce success metadata | PASS | `direct-temporary-root-probe` |
| f2-sidecar-bound | F2 bounded sidecar memory | resource exhaustion | a sidecar of `MAX_TOTAL_BYTES + 1` is rejected before parsing | PASS; 1,048,577 bytes rejected against 1,048,576-byte limit | `direct-temporary-root-probe` |
| f2-root-confinement | F2 artifact confinement | symlink/path escape | fixed-name sidecar symlink outside shot root is rejected | PASS | `direct-temporary-root-probe` |
| f2-cancel-resume-stale | F2 resume/recovery | cancel/resume and stale state | repeated cleanup removes `shot_000.tmp`; malformed completed shot moves to `invalid_attempts/...` | PASS | `direct-temporary-root-probe` |
| f2-hung-command | command supervision | hung command | `timeout ... 1s sleep 5` terminates with exit 124 | PASS on two repetitions | `hung-command-run1`, `hung-command-run2` |
| f2-flake-repeat | repeatability | flaky repeat | focused persistence suite produces identical 3/3 result twice | PASS | `focused-run1`, `focused-run2` |
| f2-misleading-success | result integrity | misleading success | output text alone is not success; command with `OK` but exit 17 is rejected | PASS; recorded as `REJECT_OUTPUT_ONLY_SUCCESS` | `misleading-success` |
| f2-repeated-interruption | supervision/recovery | repeated interruption | repeated bounded interruption has stable exit 124; no residue | PASS | `repeated-interruption`, `hung-command-run1`, `hung-command-run2`, `cleanup-receipt` |
| f2-prompt-injection | persistence lane input surface | prompt injection | not applicable: no prompt-bearing input is consumed by this persistence API | NOT_APPLICABLE | `f2-remediation-persistence-lane-manual-qa` |
| f2-dirty-worktree | workspace guard | dirty worktree | pre-existing tracked modifications are preserved; no production edits are made | PASS | `dirty-worktree-check`, `cleanup-receipt` |

## artifactRefs

| id | kind | description | path |
| --- | --- | --- | --- |
| `focused-run1` | log | exact three persistence tests, first run, 3/3 | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/persistence-lane/focused-run1.log` |
| `focused-run2` | log | exact three persistence tests, repeat, 3/3 | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/persistence-lane/focused-run2.log` |
| `direct-temporary-root-probe` | jsonl log | independent bounded-memory, confinement, idempotence, stale/cancel, malformed, quarantine probes | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/persistence-lane/direct-temporary-root-probe.jsonl` |
| `hung-command-run1` | log | first 1-second timeout of 5-second sleep, exit 124 | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/persistence-lane/hung-command-run1.log` |
| `hung-command-run2` | log | repeated timeout, exit 124 | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/persistence-lane/hung-command-run2.log` |
| `misleading-success` | log | output `OK` with exit 17 rejected as success | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/persistence-lane/misleading-success.log` |
| `repeated-interruption` | log | repeated interruption case mapping | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/persistence-lane/repeated-interruption.log` |
| `dirty-worktree-check` | log | tracked worktree status comparison before/after | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/persistence-lane/dirty-worktree-check.log` |
| `scoped-diff-check` | log | scoped `git diff --check` result | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/persistence-lane/scoped-diff-check.log` |
| `cleanup-receipt` | log | evidence non-empty check, no temporary probe directories, tracked status preservation | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/persistence-lane/cleanup-receipt.txt` |
