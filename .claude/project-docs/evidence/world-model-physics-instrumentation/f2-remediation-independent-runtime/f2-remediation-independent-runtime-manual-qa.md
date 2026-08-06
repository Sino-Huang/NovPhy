# F2 remediation independent runtime QA

Verdict: **confirmed** for the requested F2 behaviors, with a disclosed Unity process-exit residual risk and one unrelated broad-suite fixture failure. The three persistence scenarios passed twice; malformed/cancel-resume/stale-state probes passed; Unity 2019.4.41f2 produced passing XML twice for the silent-client capacity/timeout test and once for request 70. Unity exits 134 after saving XML during known CEF shutdown.

## surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
| --- | --- | --- | --- | --- | --- |
| f2-persistence-run1 | F2 atomic persistence: valid completed shot is idempotently reused; fixed-name sidecar symlink is rejected; nondirectory frames quarantine | Python CLI | `python -m unittest tests.test_collect_rollouts.PhysicsCapturePersistenceTests.test_second_collection_reuses_valid_completed_shot_without_bridge_calls tests.test_collect_rollouts.PhysicsCapturePersistenceTests.test_fixed_name_state_sidecar_symlink_outside_shot_is_rejected tests.test_collect_rollouts.PhysicsCapturePersistenceTests.test_completed_shot_with_nondirectory_frames_is_quarantined` | PASS (3/3) | `python-focused-run1` |
| f2-persistence-run2 | same as above, repeated for flake detection | Python CLI | same exact invocation | PASS (3/3) | `python-focused-run2` |
| f2-prepare | F2 launcher/prepare contract remains green | Python CLI | `python -m unittest tests.test_prepare_rollout_dataset` | PASS (38/38) | `python-prepare-suite` |
| f2-bridge-legacy | request 38 and request 62 framing preserved | Python CLI | `python -m unittest tests.test_webui_bridge.PhysicsCaptureV1Tests.test_legacy_request_38_and_62_fixture_bytes_remain_unchanged tests.test_webui_bridge.PhysicsCaptureV1Tests.test_recorder_backed_shoot_preserves_request_38_framing_and_consumes_ground_truth` | PASS (2/2) | `python-bridge-38-62` |
| f2-unity-silent-run1 | Unity queued + in-flight silent clients hold global capacity until timeout, then normal request reaches `WaitForEndOfFrame` | Unity 2019.4.41f2 EditMode | `timeout --signal=TERM --kill-after=20s 180s env DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1 LD_LIBRARY_PATH="/tmp/opencode/unity-2019.4-libssl1.1/root/usr/lib/x86_64-linux-gnu:/tmp/opencode/unity-2019.3.4f1-libs/root/usr/lib/x86_64-linux-gnu:/tmp/opencode/unity-2019.3.4f1-libs/root/usr/lib" /home/sukai/.local/share/novphy-unity/2019.4.41f2-6b23d448b533/editor/Editor/Unity -batchmode -nographics -projectPath /mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/tasks/task_template_designer -runTests -testPlatform EditMode -testFilter PhysicsCaptureProtocolTests.SilentInFlightClientsHoldCapacityUntilTimeoutThenReleaseIt -testResults .../unity-silent-inflight-run1.xml -logFile .../unity-silent-inflight-run1.log` | PASS XML 1/1; process 134 after CEF shutdown | `unity-silent-inflight-run1-xml`, `unity-silent-inflight-run1-log` |
| f2-unity-silent-run2 | same Unity scenario repeated | Unity 2019.4.41f2 EditMode | same invocation with `run2` result/log paths | PASS XML 1/1; process 134 after CEF shutdown | `unity-silent-inflight-run2-xml`, `unity-silent-inflight-run2-log` |
| f2-unity-request70 | normal request-70 envelope and same render-frame batch | Unity 2019.4.41f2 EditMode | same pinned Unity command with `-testFilter PhysicsCaptureProtocolTests.Request70EnvelopeContainsActualPngAndSameRenderFrameBatch` | PASS XML 1/1; process 134 after CEF shutdown | `unity-request70-xml`, `unity-request70-log` |
| f2-syntax | scoped Python syntax and whitespace checks | CLI | `python -m py_compile scripts/collect_rollouts.py scripts/physics_capture_parsing.py scripts/rollout_artifacts.py scripts/rollout_validation_types.py src/webui/bridge.py tests/test_collect_rollouts.py tests/test_webui_bridge.py`; `git diff --check --` scoped F2 files | PASS | `py-compile`, `diff-check` |

## adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
| --- | --- | --- | --- | --- | --- |
| f2-malformed-envelope | request 70 protocol bounds and malformed input rejection | malformed input | malformed magic/version/length/JSON/PNG/render-frame envelopes reject and disconnect | PASS (9/9) | `adversarial-malformed-envelope` |
| f2-cancel-resume-stale | interrupted temporary sidecars removed on repeated resume; corrupt completed attempts quarantined; malformed request has no success metadata | cancel/resume + stale state | cleanup/quarantine and no false success | PASS (3/3) | `adversarial-cancel-resume-stale` |
| f2-hung-bound | command supervision | hung command | a 5s sleep under a 1s timeout exits 124 | PASS | `adversarial-hung-timeout` |
| f2-flake-repeat | repeated behavior | flaky tests | focused persistence and Unity tests give same result on repeat | PASS; Python 3/3 twice, Unity XML 1/1 twice | `python-focused-run1`, `python-focused-run2`, `unity-silent-inflight-run1-xml`, `unity-silent-inflight-run2-xml` |
| f2-misleading-success | result artifact versus process status | misleading success | passing Unity XML must disclose nonzero process status | PASS as disclosure; XML is valid, process exits 134 in CEF shutdown | `unity-silent-inflight-run1-log`, `unity-silent-inflight-run2-log`, `unity-request70-log` |
| f2-repeated-interruption | repeated interruption | repeated interruption | not applicable to this bounded synchronous test surface; timeout probe covered supervision | NOT_APPLICABLE (one-line reason) | `adversarial-hung-timeout` |
| f2-prompt-injection | untrusted test/data content | prompt injection | no prompt text is consumed as instructions | NOT_APPLICABLE (no prompt-bearing input surface) | `f2-remediation-independent-runtime-manual-qa` |
| f2-dirty-worktree | workspace guard | dirty worktree | observe and preserve pre-existing modifications; do not overwrite production/canonical paths | PASS (status captured; no production edits) | `cleanup-receipt` |

## artifactRefs

| id | kind | description | path |
| --- | --- | --- | --- |
| python-focused-run1 | log | focused persistence tests, first run | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/python-focused-run1.log` |
| python-focused-run2 | log | focused persistence tests, repeat | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/python-focused-run2.log` |
| python-prepare-suite | log | prepare suite 38/38 | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/python-prepare-suite.log` |
| python-focused-full-run1 | log | broader collector module, 80/81 with one unrelated fixture failure | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/python-focused-full-run1.log` |
| python-focused-full-run2 | log | repeated broader collector module, same 80/81 result | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/python-focused-full-run2.log` |
| python-bridge-38-62 | log | request 38/62 compatibility tests 2/2 | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/python-bridge-38-62.log` |
| py-compile | log | scoped py_compile | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/py-compile.log` |
| diff-check | log | scoped git diff whitespace check | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/diff-check.log` |
| unity-silent-inflight-run1-xml | XML | Unity exact filter result, 1/1 passed | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/unity-silent-inflight-run1.xml` |
| unity-silent-inflight-run1-log | log | Unity run 1, includes CEF SIGSEGV after results | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/unity-silent-inflight-run1.log` |
| unity-silent-inflight-run2-xml | XML | Unity exact filter repeat, 1/1 passed | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/unity-silent-inflight-run2.xml` |
| unity-silent-inflight-run2-log | log | Unity run 2, includes CEF SIGSEGV after results | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/unity-silent-inflight-run2.log` |
| unity-request70-xml | XML | request 70 envelope test, 1/1 passed | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/unity-request70-run.xml` |
| unity-request70-log | log | request 70 Unity log and shutdown status | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/unity-request70-run.log` |
| adversarial-malformed-envelope | log | malformed envelope suite 9/9 | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/adversarial-malformed-envelope.log` |
| adversarial-cancel-resume-stale | log | cancellation/resume/stale-state probes 3/3 | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/adversarial-cancel-resume-stale.log` |
| adversarial-hung-timeout | log | bounded hung-command probe, exit 124 | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/adversarial-hung-timeout.log` |
| cleanup-receipt | log | process/port/temp/core/protected-root checks | `/mnt/array/sukaih/Project/NovPhy/.omo/evidence/world-model-physics-instrumentation/f2-remediation-independent-runtime/cleanup-receipt.txt` |

## Residual risk

The full `tests.test_collect_rollouts` module was also run twice (81 tests each) and has one repeatable failure in `CollectRolloutsTest.test_known_dataset_artifacts_classify_gameplay_and_reported_menu_shots` at line 1362; this is outside the three F2 persistence scenarios and is retained as evidence in `python-focused-full-run1.log` and `python-focused-full-run2.log`. Unity's XML is authoritative for the test result, but the editor's CEF shutdown SIGSEGV (exit 134) means callers should not treat the process status alone as a clean pass.
