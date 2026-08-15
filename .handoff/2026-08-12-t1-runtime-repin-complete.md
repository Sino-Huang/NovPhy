# T1 runtime re-pin complete — success handoff

## Completed

- T1 implementation: `d5be336be778103ac2ae883d4d946a1df3eaf540`.
- Closure records began from branch `physics-unity-2019.4` at HEAD `4bde6262e99c4c8d38bfdccc1af3df240d63b159`.
- Closure records commit: `5a34ec9e0a9f4f45ae7060de65c43a6d322d444c` (`docs(runtime-gate): record accepted T1 re-pin`).
- Preserved C# RED: 25 discovered, 23 passed, exactly two ordering failures; XML sha256 `787cba1af6aae45c0dcb6ac14dde56ab09efe32a5597711d2ad6ffc78a5456db`; source restored sha256 `d6bc41af198c986e8ce371131c617f2c0d125b88f02a30388bd33cdcc6d3a2cd`. It was not repeated.
- Python regression: 3/3 passed. Full per-class EditMode: 59/59 passed. Mutation proof: 9/9 RED; smoke source restored sha256 `ba3c8772534a5e535c1ba7d16faa1427dae342e6a2f7a8e9017f5449e2d05aea`.
- The first mutation attempt was externally killed at 120 seconds. Its one-line residue was recorded, classified EASY, restored exactly, and the single allowed retest passed.
- Deterministic builds A and B exited 0; 151 provenance files compared; zero drift; archive sha256 `de59061350f78f79420d76ec33f1c506aa17c1cfc25d197cdd2f5f770874e838`.
- Exactly one full smoke accepted. Report: `.claude/project-docs/evidence/world-model-physics-instrumentation/task-8-smoke.json`. Output: `.claude/project-docs/evidence/runtime-repin-gate-20260810/session-5-full-smoke/`.
- Smoke evidence: four collisions; selected contact id `contact:361:-632:0:-646|world:static:-466:-466:0`; relative speed `10.6197062`; request sequence `1 -> 2` under one capture id; stable listener; zero recorder refusals; port clear; temporary clone removed; protected roots unchanged.
- Conditional re-pin completed only after acceptance. Exactly `archive.sha256`, `novphy-physics-player-2019.4.41f2.tar.gz`, and `unity-build.log` were replaced under `sciencebirdsgames/physics-v1/`. The archive, receipt, build A, and smoke provenance all bind to `de59061350f78f79420d76ec33f1c506aa17c1cfc25d197cdd2f5f770874e838`.
- Status: `repin_complete`.

## Authority / Limits

- Publication was not authorized and did not occur.
- Cohort collection was not authorized and did not occur.
- No runtime publication occurred. Repository closure commits are finalized separately and do not publish the runtime artifact.
- The accepted smoke cannot be repeated under the consumed one-shot authorization.
- The separate same-step duplicate-ingestion finding remains outside this T1 closure; no result or approval was manufactured for it.

## Next Plan Action

The dependency-ready next action is an explicit owner decision on whether to authorize publication of the re-pinned archive. This is next on the critical path because the runtime gate and conditional re-pin are complete, while publication remains intentionally blocked by authority rather than evidence.

Required input: explicit publication authorization naming the re-pinned candidate. Hard boundaries: do not publish or collect a cohort before that authorization; repository Git finalization is limited to the closure records and is not runtime publication. Acceptance criterion for the decision step: the owner explicitly authorizes or declines publication of archive sha256 `de59061350f78f79420d76ec33f1c506aa17c1cfc25d197cdd2f5f770874e838`.

Smallest first inspection:

```bash
/usr/bin/sha256sum sciencebirdsgames/physics-v1/novphy-physics-player-2019.4.41f2.tar.gz
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -c 'import json; print(json.load(open(".claude/project-docs/evidence/world-model-physics-instrumentation/task-8-smoke.json"))["provenance"]["archive_sha256"])'
```

Both must print `de59061350f78f79420d76ec33f1c506aa17c1cfc25d197cdd2f5f770874e838` before any separately authorized publication workflow begins.
