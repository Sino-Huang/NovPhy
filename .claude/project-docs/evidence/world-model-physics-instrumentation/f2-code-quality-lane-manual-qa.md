# F2 Code Quality Lane Manual QA

## surfaceEvidence

| Scenario | Criterion | Surface | Exact invocation | Verdict | Artifact refs |
|---|---|---|---|---|---|
| CQ-01 | F2 bounded parsing | terminal/data | `python -m py_compile scripts/collect_rollouts.py scripts/physics_capture_parsing.py scripts/rollout_artifacts.py src/webui/bridge.py tests/test_collect_rollouts.py` | PASS | `pycompile` |
| CQ-02 | F2 implementation regression | terminal/data | `python -m unittest -v tests.test_collect_rollouts.PhysicsCapturePersistenceTests tests.test_webui_bridge.PhysicsCaptureV1Tests tests.test_webui_bridge.PhysicsCaptureV1MalformedEnvelopeTests tests.test_physics_capture_contract.PhysicsCaptureContractTests` | PASS | `focused-tests` |
| CQ-03 | F2 malformed and bounded sidecars | terminal/data | `python <TemporaryDirectory fixture probe>` | PASS | `temp-fixtures` |
| CQ-04 | F2 request-70 timeout handling | terminal/data | `python <real loopback server withholding response; bridge timeout=0.1>` | PASS | `hung` |
| CQ-05 | F2 repeatability | terminal/data | `python -m unittest tests.test_collect_rollouts.PhysicsCapturePersistenceTests tests.test_webui_bridge.PhysicsCaptureV1MalformedEnvelopeTests` twice | PASS | `repeat` |
| CQ-06 | Worktree integrity | terminal/data | `git status --short --branch; git diff --name-only -- <protected scopes>; sha256sum <target files>` | PASS (observation only) | `dirty` |
| CQ-07 | Unity protocol source | source inspection | `sed -n '1,420p' tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsCaptureProtocol.cs; rg -n MaxEnvelopeBytes MaxPngBytes MaxJsonBytes MaxPendingClients RequestReadTimeoutMilliseconds ...` | PASS (static only) | `unity-static` |

## adversarialCases

| Scenario | Criterion | Adversarial class | Expected behavior | Verdict | Artifact refs |
|---|---|---|---|---|---|
| ADV-01 | F2 parser | malformed input | Reject malformed JSON with a typed contract error and no accepted artifact | PASS | `temp-fixtures`, `focused-tests` |
| ADV-02 | F2 parser | truncated input | Reject truncated JSON before producing a capture | PASS | `temp-fixtures` |
| ADV-03 | F2 parser | oversized input | Reject over-limit sidecar before JSONL reader/allocation | PASS | `temp-fixtures`, `focused-tests` |
| ADV-04 | F2 artifact validation | symlink/out-of-root path | Reject symlink and missing/escaping path under shot-root confinement | PASS | `temp-fixtures`, `focused-tests` |
| ADV-05 | F2 collector | stale state | Reuse only a validated completed shot; avoid bridge recapture | PASS | `focused-tests`, `repeat` |
| ADV-06 | F2 collector | dirty worktree | Preserve existing user modifications; no destructive reset or checkout | PASS | `dirty` |
| ADV-07 | F2 bridge | hung command | Bound socket wait and fail closed without leaked server/process | PASS | `hung` |
| ADV-08 | F2 bridge/parser | flaky repeat | Two independent focused runs remain green | PASS | `repeat` |
| ADV-09 | F2 collector | misleading success | Malformed request-70 capture raises and does not write success metadata | PASS | `focused-tests`, `temp-fixtures` |
| ADV-10 | F2 collector | repeated interruption | Temporary interrupted shot is removed on repeated cleanup/resume | PASS | `focused-tests` |
| ADV-11 | F2 workflow | prompt injection | not_applicable: no external prompt/content ingestion surface is exercised by these modules | N/A | `focused-tests` |
| ADV-12 | F2 workflow | cancel/resume | Resume path is exercised through completed-shot reuse and interrupted-temp cleanup | PASS | `focused-tests`, `repeat` |

## artifactRefs

- `pycompile`: `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-code-quality-lane-pycompile.log`
- `diff-check`: `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-code-quality-lane-diff-check.log`
- `focused-tests`: `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-code-quality-lane-focused-tests.log`
- `temp-fixtures`: `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-code-quality-lane-temp-fixtures.log`
- `repeat`: `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-code-quality-lane-repeat.log`
- `hung`: `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-code-quality-lane-hung-command.log`
- `dirty`: `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-code-quality-lane-dirty-worktree.log`
- `unity-static`: `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-code-quality-lane-unity-static.log`
