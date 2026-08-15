# Project Docs — migrated from `.omo` (2026-08-07)

This directory consolidates the durable documents from the project's work-tracking folders (`.omo`,
`.sisyphus`) into one Claude Code-managed location. The source folders have been removed; this is the
single source of truth for plans, knowledge, evidence, and the work ledger.

## Layout

| Path | Contents | Origin |
|---|---|---|
| `plans/` | Work plans: `world-model-physics-instrumentation.md` (14/14 complete, F1-F4 APPROVE), `world-model-data-pipeline.md` (14/14 complete), `rollout-menu-static-shot-bug.md` (historical) | `.omo/plans/` |
| `knowledge/` | Five compact current guides plus 21 preserved source notes and `knowledge-compression-manifest.json`. Start with `00-current-gates-and-recovery.md`, then use `10` rollout, `20` physics/Unity, `30` world-model data, and `40` engineering/provenance. Original notes remain intact for source-level detail. | `.omo/knowledges/` (both branches, deduped), compressed index added 2026-08-10 |
| `drafts/` | Original plan drafts (physics instrumentation, data pipeline) | `.omo/drafts/` |
| `research/` | ULW research synthesis (protocol, binary parity, observation manifest, cause-disappearance, etc.) | `.omo/ulw-research/` |
| `notepads/` | Per-work `decisions.md`, `learnings.md`, `issues.md`, `problems.md` (flattened as `<work>-<file>.md`) | `.omo/notepads/` |
| `evidence/` | Text audit trail (2027 files, ~37 MB): per-todo done-claims, verdicts, receipts, XML/JSON/log evidence, method-proposal slides, final-wave verdicts, publication authority | `.omo/evidence/` from both branches, deduped by sha256 |
| `ledger.jsonl` | Consolidated work event ledger (144 valid lines) | `.omo/start-work/ledger.jsonl` (both branches, merged) |
| `archive/` | `boulder.json` (work-state snapshot), stale session handoff, `dropped-bulk-inventory.json` | `.omo/` top level |

## Evidence trim note

The physics worktree's `.omo/evidence` was 3.5 GB. Under the approved "trim bulk, keep text evidence"
rule, 33 files (3.5 GB) were dropped: player archives (reproducible from source; the published archive
lives in `sciencebirdsgames/physics-v1`), 3 GB of active-data-root TSV listings, and giant (>5 MB)
hash manifests and `*.normalized` duplicates — whose conclusions are recorded in the retained small
verdict JSONs. Every dropped file's sha256 is recorded in `archive/dropped-bulk-inventory.json`.

## Code that references this directory

Patched 2026-08-07 to point here instead of `.omo`:
- `scripts/build_physics_player.sh` — `--migration-provenance` → `evidence/world-model-physics-instrumentation/task-2-migration-provenance.json`
- `scripts/verify_physics_capture_docs.py` — publication authority → `evidence/world-model-physics-instrumentation/final-published-runtime`
- `scripts/smoke_physics_capture.py` — default report → `evidence/world-model-physics-instrumentation/task-8-smoke.json`
- `scripts/package_physics_player.py` — product-diff exclude `:(exclude).omo` → `:(exclude).claude`
- `docs/data_contracts/physics_capture_v1.md` — evidence path references
