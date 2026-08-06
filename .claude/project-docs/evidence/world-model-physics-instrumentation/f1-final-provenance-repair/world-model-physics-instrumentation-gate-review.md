# F1 Final Provenance Repair Gate Review

recommendation: APPROVE

blockers: none

originalIntent: Deliver a separately staged Unity 2019.4.41f2 physics-capture player and opt-in supervision pipeline with exact engine/RGB alignment, authoritative sidecars, preserved legacy behavior, reproducible provenance, and immutable canonical player/project/dataset roots.

desiredOutcome: The exact final source commit and rebuilt archive are mutually bound; ignored Unity package files and the player wrapper cannot drift outside provenance; packaging and verification fail closed; all plan outputs and guardrails retain evidence.

userOutcomeReview: PASS. Exact HEAD `fdb1bd5e8c9677a3c4ef15d0b8ddec74d55d71a9` and tree `964a40fd02c0109224e854c984304d7bbc1bc731` were reproduced. The two rebuilt archives and published archive are byte-identical at SHA-256 `a056271134ff8ccb9ba2b790606b7e29cc3729fa2d16f3b26af37234eeafb63b`. The receipt, provenance, both ignored package digests, tracked wrapper identity, 151 payload hashes, clean preflight, static verifier, tamper rejection, and protected-root receipts agree.

Checked artifacts:

- `.omo/plans/world-model-physics-instrumentation.md`
- canonical `.omo/start-work/ledger.jsonl`
- `.omo/evidence/world-model-physics-instrumentation/final-wave-exact-final-native/fdb1bd5e/DoneClaim.json`
- `.omo/evidence/world-model-physics-instrumentation/final-wave-exact-final-native/fdb1bd5e/reproducibility.json`
- `.omo/evidence/world-model-physics-instrumentation/final-wave-exact-final-native/fdb1bd5e/build-1-provenance.json`
- `.omo/evidence/world-model-physics-instrumentation/final-wave-exact-final-native/fdb1bd5e/build-2-provenance.json`
- `.omo/evidence/world-model-physics-instrumentation/final-wave-exact-final-native/fdb1bd5e/published-smoke.json`
- `sciencebirdsgames/physics-v1/archive.sha256`
- `sciencebirdsgames/physics-v1/novphy-physics-player-2019.4.41f2.tar.gz`
- `scripts/package_physics_player.py`, `scripts/verify_physics_player.py`
- `tests/test_package_physics_player.py`, `tests/test_verify_physics_player.py`
- `f1-verdict.json`, `cleanup-receipt.json`

Direct remove-ai-slops/overfit pass: no deletion-only test, requested-removal-only test, prompt-prose assertion, tautological assertion, output-derived expected value, or unnecessary production parsing/normalization was found in the provenance repair scope. The tests assert observable publication ordering, digest binding, fail-closed behavior, and archive contents. Fixture helpers mirror some filesystem setup, but this does not violate an F1 criterion.

Direct programming pass: package inputs are parsed at the boundary, failures are explicit, final archive/receipt replacement is atomic and tested under interruption, the real CLI is exercised with bounded invocations, and deterministic identities are externally recomputed. The older code-review artifact explicitly documents both required skill perspectives but predates the final repairs; its stale high findings are covered by later commits and are not accepted as current evidence.

Exact evidence gaps: none for the reviewed commit/archive. The initial 120-second full protection digest attempt timed out and was excluded; a bounded 300-second rerun completed and reproduced all three expected hashes. After the clean preflight and artifact snapshot, a concurrent uncommitted change appeared in `scripts/prepare_rollout_dataset.py`; it is not part of reviewed HEAD/tree or the rebuilt archive, and a future packaging run must restore or commit a deliberate final state before preflight can pass again.
