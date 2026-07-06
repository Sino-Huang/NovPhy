# Decisions

## 2026-07-05 Start Work Context
- Execute Wave 1 first: Task 1 validator, Task 2 metadata instrumentation, Task 3 knowledge documentation.
- Use search/analyze mode before implementation: launched explore/librarian agents for existing image primitives, metadata graph, and Pillow image-diff APIs.
- Implementation is delegated only; Atlas verifies and marks plan checkboxes after each verified completion.

## 2026-07-05 Wave 1 Dispatch Decisions
- Task 1 should implement the validator in `scripts/collect_rollouts.py` unless a tiny helper module is clearly cleaner; avoid new dependencies.
- Task 2 should keep state evidence additive/backward-compatible and avoid relying on protocol state as authoritative.
- Task 3 is documentation-only; WebUI files must remain unchanged.

## 2026-07-05 WebUI Boundary Decision
- WebUI is comparison evidence for collector behavior, not a route rewrite target. Startup/connect recovery uses protocol transitions, `/api/shot` does not re-prepare per shot, and `/api/load-level` plus `/api/restart` currently use `load_next_available_level()` / protocol `53`.
- Collector fixes must not default to protocol `51` or `52`; those bridge methods exist, but local WebUI notes say explicit load/restart paths can hang in this Unity build.
