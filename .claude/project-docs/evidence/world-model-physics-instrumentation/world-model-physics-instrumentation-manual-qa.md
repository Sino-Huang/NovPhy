# Manual QA Matrix

Goal: `world-model-physics-instrumentation`  
Required source: `55d6ec93cafc77807342cd7573283f1ed20ca691`  
Observed source: `55d6ec93cafc77807342cd7573283f1ed20ca691`

## surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
| --- | --- | --- | --- | --- | --- |
| F3-S1 | F3 / task 8 accepted staged-player smoke | staged player + temporary output root | `timeout --signal=TERM --kill-after=30s 900s python scripts/smoke_physics_capture.py --stage sciencebirdsgames/physics-v1 --output-dir .omo/evidence/world-model-physics-instrumentation/final-wave-exact/f3-smoke-output --report .omo/evidence/world-model-physics-instrumentation/final-wave-exact/f3-smoke.json` | PASS | `A1`, `A2` |
| F3-S2 | F3 / legacy request 62 and request 38 compatibility | Python bridge test surface | `python -m unittest tests.test_webui_bridge.PhysicsCaptureV1Tests.test_legacy_request_38_and_62_fixture_bytes_remain_unchanged tests.test_webui_bridge.PhysicsCaptureV1Tests.test_recorder_backed_shoot_preserves_request_38_framing_and_consumes_ground_truth -v` | PASS | `A3` |
| F3-S3 | F3 / archive and staged provenance | published `sciencebirdsgames/physics-v1` archive | `sha256sum sciencebirdsgames/physics-v1/novphy-physics-player-2019.4.41f2.tar.gz`; `(cd sciencebirdsgames/physics-v1 && sha256sum -c archive.sha256)` | PASS | `A4` |
| F3-S4 | F3 / accepted artifact contract | accepted `shot_001` validator | `python -c 'from pathlib import Path; from scripts.rollout_artifacts import validate_physics_shot_artifact; print(validate_physics_shot_artifact(Path(".omo/evidence/world-model-physics-instrumentation/final-wave-exact/f3-smoke-output/shot_001")))'` | PASS | `A5`, `A6` |
| F3-S5 | F3 / smoke-driver regression surface | Python smoke test suite | `python -m unittest tests.test_smoke_physics_capture -v` | PASS | `A7` |

## adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
| --- | --- | --- | --- | --- | --- |
| F3-A1 | F3 / staged provenance | stale state | A changed source/archive identity must not be accepted as the required staged player. | PASS | `A1`, `A4` |
| F3-A2 | F3 / protected roots | dirty worktree / protected mutation | Smoke execution must leave canonical project, production player, and active cohort digests unchanged. | PASS | `A2` |
| F3-A3 | F3 / verifier negative path | malformed input | A wrong expected archive SHA must exit nonzero. | PASS | `A8` |
| F3-A4 | F3 / repeatability | flaky tests | The smoke-driver regression suite must complete with all 11 tests passing. | PASS | `A7` |
| F3-A5 | F3 / bounded execution | hung or long command | The recorded live smoke invocation must complete under its 900-second TERM/KILL bound. | PASS | `A2` |
| F3-A6 | F3 / interruption handling | repeated interruptions | No interruption or resumable publication operation was part of this run; no interruption behavior is claimed. | NOT_APPLICABLE | `A2` |
| F3-A7 | F3 / input boundary | prompt injection | No untrusted instruction-bearing input or model-routing surface was exercised. | NOT_APPLICABLE | `A2` |

## artifactRefs

| id | kind | description | path |
| --- | --- | --- | --- |
| A1 | JSON | Exact-head F3 gate record with staged provenance, request 62/38/70 observations, and accepted-shot result. | `.omo/evidence/world-model-physics-instrumentation/final-wave-exact/f3.json` |
| A2 | JSON | Machine-produced live smoke report; accepted shot, sidecar counts, render-frame alignment, and protected-root before/after digests. | `.omo/evidence/world-model-physics-instrumentation/final-wave-exact/f3-smoke.json` |
| A3 | log | Fresh legacy request 38/62 compatibility test output. | `.omo/evidence/world-model-physics-instrumentation/manual-qa-legacy-tests.log` |
| A4 | log | Fresh HEAD/tree and archive SHA-256 plus archive receipt verification. | `.omo/evidence/world-model-physics-instrumentation/manual-qa-provenance.log` |
| A5 | log | Fresh `validate_physics_shot_artifact` result for accepted `shot_001`. | `.omo/evidence/world-model-physics-instrumentation/manual-qa-shot-validator.log` |
| A6 | log | Fresh JSONL/PNG invariant inspection for the accepted shot. | `.omo/evidence/world-model-physics-instrumentation/manual-qa-json-invariants.log` |
| A7 | log | Fresh smoke-driver regression suite output (11 tests). | `.omo/evidence/world-model-physics-instrumentation/manual-qa-smoke-tests.log` |
| A8 | log | Fresh negative archive-SHA verifier invocation and nonzero exit. | `.omo/evidence/world-model-physics-instrumentation/manual-qa-negative-sha.log` |
