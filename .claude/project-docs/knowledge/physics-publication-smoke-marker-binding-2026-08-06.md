# Physics Publication Smoke-Marker Binding

The published physics archive and the smoke marker named by the public collection command form one provenance unit. After rebuilding or promoting a different archive, update and re-run all three together:

1. Generate a fresh successful smoke marker for the published archive.
2. Point `docs/data_contracts/physics_capture_v1.md` and `scripts/verify_physics_capture_docs.py` at that marker.
3. Run `python scripts/verify_physics_capture_docs.py docs`, `python -m unittest tests.test_verify_physics_capture_docs -v`, and `resolve_physics_capture_provenance` against the documented archive/marker pair.

A retained runtime approval report is not automatically a collection smoke marker. The marker must satisfy `resolve_physics_capture_provenance`: it must be newer than the staged archive, have `status=passed`, contain the expected capture contract/player/protocol versions and SHA-256 fields, and bind `archive_sha256` to the staged bytes.

On 2026-08-06, archive `1c2a1bbcea87175150451ad8981e7f28ca09195be98f7da4cb8af577d431fef4` passed static, reproducibility, runtime, and cleanup checks, but Todo 9 still designated `task-8-smoke.json` for archive `c7f9fa4c98480c1c1c8e580cb00454beda4fed4bf28a4822d31c561997906992`. The collector rejected that marker as stale and the docs verifier rejected its digest.

The final remediation changed documentation verification authority selection. `scripts/verify_physics_capture_docs.py` now starts from `final-published-runtime/done-claim.json`, confines its receipt/report/accepted-shot pointers to that evidence directory, and verifies the DoneClaim, publication receipt, final smoke, staged `archive.sha256`, and archive bytes as one digest chain. Historical `task-8-smoke.json` remains evidence for its original publication and must not override a newer complete final DoneClaim.
