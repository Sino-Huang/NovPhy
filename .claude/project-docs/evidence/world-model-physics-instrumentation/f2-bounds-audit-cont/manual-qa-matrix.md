# F2 Bounded Sidecar Manual QA

| Scenario | Invocation | Fresh observable | Verdict |
| --- | --- | --- | --- |
| Required focused suite | `python -m unittest -v tests.test_collect_rollouts.PhysicsCapturePersistenceTests tests.test_webui_bridge.PhysicsCaptureV1MalformedEnvelopeTests tests.test_physics_capture_contract.PhysicsCaptureContractTests` | 46 tests passed, exit 0 | PASS |
| Valid sidecars | TemporaryDirectory Python fixture | `states=2 events=9` | PASS |
| Truncated JSONL | TemporaryDirectory Python fixture | `rejected=malformed_json` | PASS |
| Oversized sidecar | TemporaryDirectory fixture with `_read_jsonl` trap | `rejected-before-reader=invalid_value` | PASS |
| Out-of-root symlink | Persisted TemporaryDirectory artifact fixture | `symlink artifact is forbidden` | PASS |
| Post-check replacement | Focused persistence test | state and frame swap test passed | PASS |
| Streaming SHA | TemporaryDirectory descriptor instrumentation | 3 updates, largest 7874 bytes, under 65536 bytes | PASS |
| Malformed envelope | Focused bridge test | overflow envelope test passed and disconnected bridge | PASS |
| Repeated interruption | Focused persistence test | interrupted temporary attempt cleanup passed | PASS |
| Flaky repeat | Contract plus bridge malformed tests, run twice | 24 tests passed on each run | PASS |
| Hung command | `timeout 3s python -c 'import time; time.sleep(10)'` | timeout exit 124 | PASS |
| Static checks | `python -m py_compile ... && git diff --check` | exit 0 | PASS |

No dry-run was used for the Python fixture or unit invocations. The fixture emitted non-empty terminal logs for every probe.
