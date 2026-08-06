# F2 Code Quality Audit

## Verdict

`needs-fix`. The review found one HIGH correctness and bounded-memory defect. The
existing pending-client cap does not bound live silent request coroutines.

## Scope And Method

- Governed plan read directly: `.omo/plans/world-model-physics-instrumentation.md`,
  including F2 and the exact 2019.4.41f2 / immutable 2019.3 baseline requirements.
- Current migration worktree reviewed directly at
  `/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4`.
- Prior claims and reports were treated as leads only. The worktree is dirty with
  implementation and evidence from concurrent work; no unfamiliar edits were changed.
- Skill-perspective check: RAN. `omo:remove-ai-slops` and `omo:programming`
  (including Python guidance) were loaded before maintainability/test review.
  The current diff still violates the remove-ai-slops bounded-resource perspective
  in the socket lifecycle. No additional programming-perspective blocker was found
  in the repaired Python persistence seam: it uses typed immutable records and
  explicit bounded capture limits.

## Findings

### CRITICAL

None.

### HIGH

1. `PhysicsCaptureDirectSocket` bounds only queued clients, not active coroutines.
   [PhysicsCaptureProtocol.cs](/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsCaptureProtocol.cs:322)
   rejects connections only when `clients.Count` reaches four. Each frame,
   [line 336](/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsCaptureProtocol.cs:336)
   immediately dequeues one client and starts a separate `Serve` coroutine. A
   silent client then remains live for up to one second at
   [lines 342-344](/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsCaptureProtocol.cs:342).
   A continuous loopback connection flood can therefore create one additional
   live coroutine per frame while keeping the queue below its cap. This defeats
   bounded memory/resource behavior and can starve legitimate request-70 work.
   The capacity test checks only `PendingClients` and does not exercise clients
   after `Update` has started their coroutines
   ([PhysicsCaptureProtocolTests.cs](/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/tasks/task_template_designer/Assets/Tests/Editor/PhysicsCaptureProtocolTests.cs:267)).
   Fix by applying one shared cap to queued plus in-flight clients and test a
   sustained silent-client flood across multiple `Update` ticks.

### MEDIUM

None.

### LOW

1. Fresh Unity verification is unavailable on this host: the exact
   `2019.4.41f2 (6b23d448b533)` executable aborts before tests because
   `libgconf-2.so.4` is missing. This does not establish a source defect, but it
   means this audit cannot independently confirm the C# focused suite here.

2. `python scripts/verify_physics_player.py --stage sciencebirdsgames/physics-v1`
   did not complete within a fresh bounded 120-second invocation. Treat that as
   non-success and investigate separately in runtime QA; it was not used as
   proof of the source finding above.

## Fresh Verification

| Scenario | Invocation | Observable | Result |
| --- | --- | --- | --- |
| Focused Python regression | `PYTHONDONTWRITEBYTECODE=1 timeout 120s python -m unittest tests.test_collect_rollouts.PhysicsCapturePersistenceTests tests.test_prepare_rollout_dataset.PhysicsLauncherTests tests.test_prepare_rollout_dataset.PhysicsCaptureValidationTests tests.test_webui_bridge.PhysicsCaptureV1Tests tests.test_webui_bridge.PhysicsCaptureV1MalformedEnvelopeTests tests.test_smoke_physics_capture -v` | 48 tests, exit 0 | PASS |
| Static checks | `git diff --check`; `python -m py_compile` on changed collector/bridge/smoke sources; `bash -n` on changed shell scripts | all exit 0 | PASS |
| Real artifact surface | `validate_physics_shot_artifact(.omo/evidence/world-model-physics-instrumentation/todo8-live-adversarial/accepted-shot)` | `state_count=1`, `event_count=0`, state digest `bc6d...2fc2`, empty-event digest | PASS |
| Protection probe | mutate a nested temp active-data file then compare `protected_receipt` | digest changed | PASS |
| Exact Unity socket tests | exact 2019.4.41f2 Unity with `-testFilter PhysicsCaptureProtocolTests` | loader error `libgconf-2.so.4` | BLOCKED_ENVIRONMENT |
| Staged verifier | `timeout 120s python scripts/verify_physics_player.py --stage sciencebirdsgames/physics-v1` | timeout | NOT_PASSING |

## UltraQA Mapping

| Class | Verdict | Evidence |
| --- | --- | --- |
| malformed_input | triggered, PASS for Python envelope/persistence tests; source review finds active socket flood still unbounded | 48-test focused run; HIGH finding |
| prompt_injection | N/A: no prompt/LLM input surface is in the reviewed feature | source scope |
| cancel_resume | triggered, PASS for interrupted `.tmp` recovery | `test_interrupted_tmp_is_removed_on_repeated_resume` |
| stale_state | triggered, PASS for stale smoke-marker and corrupt completed-attempt rejection | launcher/validation tests |
| dirty_worktree | triggered, WATCH: worktree has concurrent dirty implementation/evidence; review was read-only and preserved it | fresh `git status --short` |
| hung_or_long_commands | triggered, WATCH: staged verifier timed out at 120 seconds; Unity could not start due to missing shared library | fresh bounded invocations |
| flaky_tests | checked, PASS: focused suite completed deterministically in 1.022 seconds; Unity suite unavailable to repeat | focused run |
| misleading_success_output | triggered, PASS for Python artifact validation; NOT PASS for unavailable Unity/staged verifier, which are not claimed green | fresh probes |
| repeated_interruptions | N/A: no repeated cancellation/resume occurred in this audit; the two bounded failures completed normally | command outcomes |

## Recommendation

`REQUEST_CHANGES`. Resolve the HIGH socket resource cap and add its adversarial
Unity test before F2 approval. Re-run the exact Unity suite in an environment
with the required shared library; investigate the staged verifier timeout in the
runtime-QA lane.
